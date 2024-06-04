from abc import ABCMeta, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import pad

from model.bottleneck_resnet import BottleneckResnet3d
from model.layers import (
    BERT,
    IPA,
    NewRayAttention,
    QKRayAttention,
    RayAttention,
)


def get_all_pocketwise_models():
    return {
        "nolayer": NoLayerPocketWiseModel,
        "rayatt": RayAttPocketWiseModel,
        "qkrayatt": QKRayAttPocketWiseModel,
        "newrayatt": NewRayAttPocketWiseModel,
        "ipa": IPAPocketWiseModel,
        "bert": BERTPocketWiseModel,
    }


def get_pocketwise_model(model_config: dict[str, Any]) -> "PocketWiseModel":
    model_name = model_config["model"]
    all_models = get_all_pocketwise_models()
    if model_name in all_models:
        model = all_models[model_name](model_config)
    else:
        raise Exception(f'model "{model_name}" not implemented')
    return model


def padstack(l: list[torch.Tensor], pad_value=0) -> torch.Tensor:
    n = len(l[0].shape)
    max_lens = [max(t.shape[i] for t in l) for i in range(n)]
    padded_l = [
        pad(
            x,
            tuple(
                (max_lens[n - 1 - i] - x.shape[n - 1 - i]) * j
                for i in range(n)
                for j in [0, 1]
            ),
            mode="constant",
            value=pad_value,
        )
        for x in l
    ]
    return torch.stack(padded_l, dim=0)


class PocketWiseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config, exclude_final_ff=False):
        super().__init__()
        if not exclude_final_ff:
            self.final_ff = nn.Linear(self.hidden_dim, 1)

    def forward(self, lens, cat_grids, R, t):
        assert len(cat_grids.shape) == 5
        N, L, _, _ = R.shape
        assert t.shape == (N, L, 3)

        xs = self.process_grids(cat_grids)
        xs = torch.split(xs, lens)

        x = padstack(xs)
        mask = padstack([torch.ones(n, dtype=bool, device=x.device) for n in lens])

        x = self.get_last_hidden(x, R, t, mask)
        out = self.final_ff(x)
        assert len(out.shape) == 3 and out.shape[-1] == 1
        return out[:, :, 0]

    @abstractmethod
    def process_grids(self, grids: torch.Tensor):
        pass

    @abstractmethod
    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        pass

    @property
    @abstractmethod
    def hidden_dim(self):
        pass

    @abstractmethod
    def load_weights(self, src_model_name, state_dict):
        pass


class BottleneckResnetPocketWiseModel(PocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        super().__init__(config, exclude_final_ff=exclude_final_ff)
        self.bottleneck_resnet = BottleneckResnet3d(
            last_channel=18,
            channels=[64, 128, 256, 512],
            strides=[2, 2, 2, 1],
            units=[2, 2, 2, 2],
        )

    def process_grids(self, grids: torch.Tensor):
        return self.bottleneck_resnet(grids)


class NoLayerPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        self._hidden_dim = model_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)
        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.act_fn = nn.ReLU()

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        x = self.dropout(x)
        x = self.dropout(self.act_fn(self.first_ff(x)))
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name == "nolayer"
        self.load_state_dict(state_dict)


class RayAttPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        rayatt_config = model_config["rayatt"]
        self._hidden_dim = rayatt_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)

        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.first_layernorm = nn.LayerNorm(self.hidden_dim)

        self.rayatt = RayAttention(**rayatt_config)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        N, L, D = x.shape
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        x = self.dropout(x)
        x = self.first_layernorm(self.dropout(self.first_ff(x)))
        x = self.rayatt(x, R, t, mask)
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name in ["nolayer", "rayatt"]
        if src_model_name == "nolayer":
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("rayatt."):
                    refined_state_dict[name] = weight
                elif name.startswith("first_layernorm."):
                    refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        elif src_model_name == "rayatt":
            if state_dict.keys() == self.state_dict().keys():
                self.load_state_dict(state_dict)
                return
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("rayatt."):
                    if not name in refined_state_dict:
                        refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        else:
            raise Exception()

        self.load_state_dict(refined_state_dict)


class QKRayAttPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        qkrayatt_config = model_config["qkrayatt"]
        self._hidden_dim = qkrayatt_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)

        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.first_layernorm = nn.LayerNorm(self.hidden_dim)

        self.qkrayatt = QKRayAttention(**qkrayatt_config)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        N, L, D = x.shape
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        x = self.dropout(x)
        x = self.first_layernorm(self.dropout(self.first_ff(x)))
        x = self.qkrayatt(x, R, t, mask)
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name in ["nolayer", "qkrayatt"]
        if src_model_name == "nolayer":
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("qkrayatt."):
                    refined_state_dict[name] = weight
                elif name.startswith("first_layernorm."):
                    refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        elif src_model_name == "qkrayatt":
            if state_dict.keys() == self.state_dict().keys():
                self.load_state_dict(state_dict)
                return
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("qkrayatt."):
                    if not name in refined_state_dict:
                        refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        else:
            raise Exception()

        self.load_state_dict(refined_state_dict)


class NewRayAttPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        newrayatt_config = model_config["newrayatt"]
        self._hidden_dim = newrayatt_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)

        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.first_layernorm = nn.LayerNorm(self.hidden_dim)

        self.newrayatt = NewRayAttention(**newrayatt_config)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        N, L, D = x.shape
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        x = self.dropout(x)
        x = self.first_layernorm(self.dropout(self.first_ff(x)))
        x = self.newrayatt(x, R, t, mask)
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name in ["nolayer", "newrayatt"]
        if src_model_name == "nolayer":
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("newrayatt."):
                    refined_state_dict[name] = weight
                elif name.startswith("first_layernorm."):
                    refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        elif src_model_name == "newrayatt":
            if state_dict.keys() == self.state_dict().keys():
                self.load_state_dict(state_dict)
                return
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("newrayatt."):
                    if not name in refined_state_dict:
                        refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        else:
            raise Exception()

        self.load_state_dict(refined_state_dict)


class IPAPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        ipa_config = model_config["ipa"]
        self._hidden_dim = ipa_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)

        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.first_layernorm = nn.LayerNorm(self.hidden_dim)

        self.ipa = IPA(**ipa_config)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        N, L, D = x.shape
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        x = self.dropout(x)
        x = self.first_layernorm(self.dropout(self.first_ff(x)))
        x = self.ipa(x, R, t, mask)
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name in ["nolayer", "ipa"]
        if src_model_name == "nolayer":
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("ipa."):
                    refined_state_dict[name] = weight
                elif name.startswith("first_layernorm."):
                    refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        elif src_model_name == "ipa":
            if state_dict.keys() == self.state_dict().keys():
                self.load_state_dict(state_dict)
                return
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("ipa."):
                    if not name in refined_state_dict:
                        refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        else:
            raise Exception()

        self.load_state_dict(refined_state_dict)


class BERTPocketWiseModel(BottleneckResnetPocketWiseModel):
    def __init__(self, config, exclude_final_ff=False):
        model_config = config["model_config"]
        bert_config = model_config["bert"]
        self._hidden_dim = bert_config.get("hidden_dim", 128)
        super().__init__(config, exclude_final_ff=exclude_final_ff)

        dropout_p = model_config.get("dropout_p", 0.1)
        self.dropout = nn.Dropout(p=dropout_p)

        self.first_ff = nn.Linear(512, self.hidden_dim)
        self.first_layernorm = nn.LayerNorm(self.hidden_dim)

        self.bert = BERT(**bert_config)

    @property
    def hidden_dim(self):
        return self._hidden_dim

    def get_last_hidden(
        self, x: torch.Tensor, R: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ):
        N, L, D = x.shape
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)
        x = self.dropout(x)
        x = self.first_layernorm(self.dropout(self.first_ff(x)))
        x = self.bert(x, R, t, mask)
        return x

    def load_weights(self, src_model_name, state_dict):
        assert src_model_name in ["nolayer", "bert"]
        if src_model_name == "nolayer":
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("bert."):
                    refined_state_dict[name] = weight
                elif name.startswith("first_layernorm."):
                    refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        elif src_model_name == "bert":
            if state_dict.keys() == self.state_dict().keys():
                self.load_state_dict(state_dict)
                return
            refined_state_dict = {
                name: weight
                for name, weight in state_dict.items()
                if not name.startswith("final_ff.")
            }
            for name, weight in self.state_dict().items():
                if name.startswith("bert."):
                    if not name in refined_state_dict:
                        refined_state_dict[name] = weight
                elif name.startswith("final_ff."):
                    refined_state_dict[name] = weight
        else:
            raise Exception()

        self.load_state_dict(refined_state_dict)
