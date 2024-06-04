import json
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.functional import pad

from model.bottleneck_resnet import BottleneckResnet3d
from model.layers import (
    BERT,
    IPA,
    FeedForward,
    NewRayAttention,
    QKRayAttention,
    RayAttention,
)
from model.pocketwise_models import PocketWiseModel, get_all_pocketwise_models, padstack
from model.seq_models import SeqModel, get_all_seq_models


def get_all_detection_models():
    return {
        "nolayer": NoLayerDetectionModel,
        "rayatt": RayAttDetectionModel,
        "qkrayatt": QKRayAttDetectionModel,
        "newrayatt": NewRayAttDetectionModel,
        "ipa": IPADetectionModel,
        "bert": BERTDetectionModel,
        "seq-rayatt": SeqRayAttDetectionModel,
        "seq-qkrayatt": SeqQKRayAttDetectionModel,
        "seq-newrayatt": SeqNewRayAttDetectionModel,
        "seq-ipa": SeqIPADetectionModel,
        "seq-bert": SeqBERTDetectionModel,
    }


def get_detection_model(
    model_config: Dict[str, Any], trn_dataset_config: Dict[str, Any]
) -> Union["VoxelBasedDetectionModel", "SeqBasedDetectionModel"]:
    model_name = model_config["model"]
    all_models = get_all_detection_models()
    if model_name in all_models:
        model = all_models[model_name](model_config, trn_dataset_config)
    else:
        raise Exception(f'model "{model_name}" not implemented')
    return model


def load_json(filename):
    with open(filename, "r") as reader:
        return json.load(reader)


def get_initialized_pocketwise_seg_model(
    seg_config, trn_dataset_config
) -> PocketWiseModel:
    experiment, version, when = (
        seg_config["experiment"],
        seg_config["version"],
        seg_config["when"],
    )
    dataset_name = trn_dataset_config["dataset"]
    fold = trn_dataset_config.get("fold", None)
    if fold is None:
        subdir_name = dataset_name
    elif type(fold) == int:
        subdir_name = f"{dataset_name}_fold{fold}"
    else:
        subdir_name = f"{dataset_name}_{fold}"

    def get_ckpt_file():
        if when == "last":
            return f"./logs/{experiment}/{version}/{subdir_name}/last.ckpt"
        elif when == "best":
            ckpt_pattern = Path(
                f"./logs/{experiment}/{version}/{subdir_name}/epoch=*.ckpt"
            )
            globbed = list(ckpt_pattern.parent.glob(ckpt_pattern.name))
            if len(globbed) == 1:
                return str(globbed[0])
            elif len(globbed) == 0:
                raise Exception(f"No file of pattern {ckpt_pattern}")
            else:
                raise Exception(f"More than one file of pattern {ckpt_pattern}")

    model_config_file = f"./logs/{experiment}/{version}/model_config.json"
    model_config = load_json(model_config_file)
    model_name = model_config["model"]

    if not model_name.startswith("seq"):
        all_models = get_all_pocketwise_models()
    else:
        all_models = get_all_seq_models()
    model = all_models[model_name](model_config, exclude_final_ff=True)
    model: Union[PocketWiseModel, SeqModel]

    if not seg_config.get("no_transfer", False):
        ckpt_file = get_ckpt_file()
        state_dict = torch.load(ckpt_file, map_location="cpu")["state_dict"]
        model_state_dict = {
            name.replace("model.", "", 1): param
            for name, param in state_dict.items()
            if not name.startswith("model.final_ff.")
        }
        model.load_state_dict(model_state_dict)
    else:
        print("No transfer!")

    return model


class VoxelBasedDetectionModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config, trn_dataset_config):
        super().__init__()
        seg_config = config["seg"]
        self.seg_model = get_initialized_pocketwise_seg_model(
            seg_config, trn_dataset_config
        )

    def seg_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("seg_model"):
                yield param

    def other_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("seg_model"):
                yield param

    @abstractmethod
    def process_further(self, x, R, t, mask):
        pass

    def forward(self, lens, cat_grids, R, t):
        assert len(cat_grids.shape) == 5
        N, L, _, _ = R.shape
        assert t.shape == (N, L, 3)

        xs = self.seg_model.process_grids(cat_grids)
        xs = torch.split(xs, lens)

        x = padstack(xs)
        mask = padstack([torch.ones(n, dtype=bool, device=x.device) for n in lens])

        x = self.seg_model.get_last_hidden(x, R, t, mask)

        out = self.process_further(x, R, t, mask)
        return out


class DecayFunction(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(self.c_init_val))

    @abstractmethod
    def decay_func(self, r):
        pass

    def forward(self, t):
        assert len(t.shape) == 3 and t.shape[2] == 3
        r = torch.sqrt(torch.sum(t**2, dim=-1))
        return self.decay_func(r)

    @property
    @abstractmethod
    def c_init_val(self):
        pass


class ExpDecayFunction(DecayFunction):
    @property
    def c_init_val(self):
        return 0.0

    def decay_func(self, r):
        return torch.exp(-self.c * r)


class NoLayerDetectionModel(VoxelBasedDetectionModel):
    def __init__(self, config, trn_dataset_config):
        super().__init__(config, trn_dataset_config)
        model_config = config["model_config"]

        self.hidden_dim = self.seg_model.hidden_dim
        assert self.hidden_dim == self.seg_model.hidden_dim

        self.final_ff = nn.Linear(self.hidden_dim, 1)
        self.reduction = model_config["reduction"]
        assert self.reduction in [
            "mean",
            "sum",
            "self-weighted-sum",
            "maxpool",
            "avgpool",
            "exp-decay-mean",
            "exp-decay-sum",
        ]
        if self.reduction == "self-weighted-sum":
            self.to_weight = nn.Linear(self.hidden_dim, 1)
            self.normalize_weight = nn.Softmax(-1)
        elif self.reduction.startswith("exp-decay-"):
            self.decay = ExpDecayFunction()

    def process_further(self, x, R, t, mask):
        if self.reduction == "mean":
            x = self.final_ff(x)[:, :, 0]
            return torch.mean(x, dim=1)
        elif self.reduction == "sum":
            x = self.final_ff(x)[:, :, 0]
            return torch.sum(x, dim=1)
        elif self.reduction == "self-weighted-sum":
            weight = self.normalize_weight(self.to_weight(x)[:, :, 0])
            x = self.final_ff(x)[:, :, 0]
            return torch.sum(x * weight, dim=1)
        elif self.reduction == "maxpool":
            x = torch.max(x, dim=1).values
            return self.final_ff(x)[:, 0]
        elif self.reduction == "avgpool":
            x = torch.mean(x, dim=1)
            return self.final_ff(x)[:, 0]
        elif self.reduction.startswith("exp-decay-"):
            x = self.final_ff(x)[:, :, 0]
            w = self.decay(t)
            if self.reduction == "exp-decay-mean":
                return torch.mean(w * x, dim=1)
            elif self.reduction == "exp-decay-sum":
                return torch.sum(w * x, dim=1)
            else:
                raise Exception()
        else:
            raise Exception()


class PocketBasedDetectionModel(VoxelBasedDetectionModel):
    def __init__(self, config, trn_dataset_config):
        super().__init__(config, trn_dataset_config)
        model_config = config["model_config"]

        core_config = model_config[self.core_name]
        self.hidden_dim = core_config.get("hidden_dim", 128)
        assert self.hidden_dim == self.seg_model.hidden_dim

        self.use_center_relpos = model_config.get("use_center_relpos", False)
        if self.use_center_relpos:
            self.embed_center_relpos = nn.Linear(3, self.hidden_dim, bias=False)
            # nn.init.zeros_(self.embed_center_relpos.weight)
            # weight zero could make the effect of this negligible

        setattr(self, self.core_name, self.core_module_cls(**core_config))

        weight_strategy = config.get("layer_weight_strategy", None)
        if weight_strategy == "copy_last_seg_layer":
            core_module = getattr(self, self.core_name)
            assert hasattr(self.seg_model, self.core_name)
            seg_core_module = getattr(self.seg_model, self.core_name)
            # core_module:RayAttention
            # seg_core_module:RayAttention
            for unit in core_module.units:
                unit.load_state_dict(seg_core_module.units[-1].state_dict())
            print("Successfully loaded the last layers")

        self.final_ff = nn.Linear(self.hidden_dim, 1)
        self.reduction = model_config["reduction"]
        assert self.reduction in [
            "mean",
            "sum",
            "self-weighted-sum",
            "maxpool",
            "avgpool",
            "exp-decay-mean",
            "exp-decay-sum",
        ]
        if self.reduction == "self-weighted-sum":
            self.to_weight = nn.Linear(self.hidden_dim, 1)
            self.normalize_weight = nn.Softmax(-1)
        elif self.reduction.startswith("exp-decay-"):
            self.decay = ExpDecayFunction()

    @property
    @abstractmethod
    def core_name(self):
        pass

    @property
    @abstractmethod
    def core_module_cls(self):
        pass

    def process_further(self, x, R, t, mask):
        if self.use_center_relpos:
            # assumption: t has been relative to the pocket center.
            center_relpos = torch.matmul(torch.inverse(R), (-t).unsqueeze(-1)).squeeze(
                -1
            )
            x = x + self.embed_center_relpos(center_relpos)

        core_module = getattr(self, self.core_name)
        x = core_module(x, R, t, mask)
        if self.reduction == "mean":
            x = self.final_ff(x)[:, :, 0]
            return torch.mean(x, dim=1)
        elif self.reduction == "sum":
            x = self.final_ff(x)[:, :, 0]
            return torch.sum(x, dim=1)
        elif self.reduction == "self-weighted-sum":
            weight = self.normalize_weight(self.to_weight(x)[:, :, 0])
            x = self.final_ff(x)[:, :, 0]
            return torch.sum(x * weight, dim=1)
        elif self.reduction == "maxpool":
            x = torch.max(x, dim=1).values
            return self.final_ff(x)[:, 0]
        elif self.reduction == "avgpool":
            x = torch.mean(x, dim=1)
            return self.final_ff(x)[:, 0]
        elif self.reduction.startswith("exp-decay-"):
            x = self.final_ff(x)[:, :, 0]
            w = self.decay(t)
            if self.reduction == "exp-decay-mean":
                return torch.mean(w * x, dim=1)
            elif self.reduction == "exp-decay-sum":
                return torch.sum(w * x, dim=1)
            else:
                raise Exception()
        else:
            raise Exception()


class RayAttDetectionModel(PocketBasedDetectionModel):
    @property
    def core_name(self):
        return "rayatt"

    @property
    def core_module_cls(self):
        return RayAttention


class QKRayAttDetectionModel(PocketBasedDetectionModel):
    @property
    def core_name(self):
        return "qkrayatt"

    @property
    def core_module_cls(self):
        return QKRayAttention


class NewRayAttDetectionModel(PocketBasedDetectionModel):
    @property
    def core_name(self):
        return "newrayatt"

    @property
    def core_module_cls(self):
        return NewRayAttention


class IPADetectionModel(PocketBasedDetectionModel):
    @property
    def core_name(self):
        return "ipa"

    @property
    def core_module_cls(self):
        return IPA


class BERTDetectionModel(PocketBasedDetectionModel):
    @property
    def core_name(self):
        return "bert"

    @property
    def core_module_cls(self):
        return BERT


# seq models


class SeqBasedDetectionModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config, trn_dataset_config):
        super().__init__()
        model_config = config["model_config"]
        seg_config = config["seg"]
        self.seg_model = get_initialized_pocketwise_seg_model(
            seg_config, trn_dataset_config
        )

        self.hidden_dim = self.seg_model.hidden_dim
        self.use_center_relpos = model_config.get("use_center_relpos", False)
        if self.use_center_relpos:
            self.embed_center_relpos = nn.Linear(3, self.hidden_dim, bias=False)
            # nn.init.zeros_(self.embed_center_relpos.weight)
            # weight zero could make the effect of this negligible

        core_config = model_config[self.core_name]
        assert self.hidden_dim == core_config["hidden_dim"]
        setattr(self, self.core_name, self.core_module_cls(**core_config))

        self.final_ff = nn.Linear(self.hidden_dim, 1)
        self.reduction = model_config["reduction"]
        assert self.reduction in ["mean"]

    def seg_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("seg_model"):
                yield param

    def other_parameters(self):
        for name, param in self.named_parameters():
            if not name.startswith("seg_model"):
                yield param

    def forward(self, x, R, t, mask):
        x = self.seg_model.get_last_hidden(x, R, t, mask)

        if self.use_center_relpos:
            # assumption: t has been relative to the pocket center.
            center_relpos = torch.matmul(torch.inverse(R), (-t).unsqueeze(-1)).squeeze(
                -1
            )
            x = x + self.embed_center_relpos(center_relpos)

        core_module = getattr(self, self.core_name)
        x = core_module(x, R, t, mask)

        if self.reduction == "mean":
            x = self.final_ff(x)[:, :, 0]
            return torch.mean(x, dim=1)
        else:
            raise Exception()

    @property
    @abstractmethod
    def core_name(self):
        pass

    @property
    @abstractmethod
    def core_module_cls(self):
        pass


class SeqRayAttDetectionModel(SeqBasedDetectionModel):
    @property
    def core_name(self):
        return "rayatt"

    @property
    def core_module_cls(self):
        return RayAttention


class SeqQKRayAttDetectionModel(SeqBasedDetectionModel):
    @property
    def core_name(self):
        return "qkrayatt"

    @property
    def core_module_cls(self):
        return QKRayAttention


class SeqNewRayAttDetectionModel(SeqBasedDetectionModel):
    @property
    def core_name(self):
        return "newrayatt"

    @property
    def core_module_cls(self):
        return NewRayAttention


class SeqIPADetectionModel(SeqBasedDetectionModel):
    @property
    def core_name(self):
        return "ipa"

    @property
    def core_module_cls(self):
        return IPA


class SeqBERTDetectionModel(SeqBasedDetectionModel):
    @property
    def core_name(self):
        return "bert"

    @property
    def core_module_cls(self):
        return BERT
