from abc import ABCMeta, abstractmethod
from typing import Any

import torch.nn as nn
from utils.preprocess import convert_residue

from model.layers import IPA, NewRayAttention, QKRayAttention, RayAttention


def get_all_seq_models():
    return {
        "seq-rayatt": SeqRayAttModel,
        "seq-qkrayatt": SeqQKRayAttModel,
        "seq-newrayatt": SeqNewRayAttModel,
        "seq-ipa": SeqIPAModel,
    }


def get_seq_model(model_config: dict[str, Any]) -> "SeqModel":
    model_name = model_config["model"]
    all_models = get_all_seq_models()
    if model_name in all_models:
        model = all_models[model_name](model_config)
    else:
        raise Exception(f'model "{model_name}" not implemented')
    return model


class SeqModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config, exclude_final_ff=False):
        super().__init__()
        model_config = config["model_config"]
        core_config = model_config[self.core_name]
        self.hidden_dim = core_config["hidden_dim"]

        vocab = sorted(list(set(convert_residue.values())))
        self.embed = nn.Embedding(len(vocab), self.hidden_dim)

        setattr(self, self.core_name, self.core_module_cls(**core_config))

        if not exclude_final_ff:
            self.final_ff = nn.Linear(self.hidden_dim, 1)

    def get_last_hidden(self, x, R, t, mask):
        N, L = mask.shape
        assert x.shape == (N, L)
        assert R.shape == (N, L, 3, 3)
        assert t.shape == (N, L, 3)

        x = self.embed(x)
        core_module = getattr(self, self.core_name)
        x = core_module(x, R, t, mask)

        return x

    def forward(self, x, R, t, mask):
        x = self.get_last_hidden(x, R, t, mask)

        return self.final_ff(x)[:, :, 0]

    @property
    @abstractmethod
    def core_name(self):
        pass

    @property
    @abstractmethod
    def core_module_cls(self):
        pass


class SeqRayAttModel(SeqModel):
    @property
    def core_name(self):
        return "rayatt"

    @property
    def core_module_cls(self):
        return RayAttention


class SeqQKRayAttModel(SeqModel):
    @property
    def core_name(self):
        return "qkrayatt"

    @property
    def core_module_cls(self):
        return QKRayAttention


class SeqNewRayAttModel(SeqModel):
    @property
    def core_name(self):
        return "newrayatt"

    @property
    def core_module_cls(self):
        return NewRayAttention


class SeqIPAModel(SeqModel):
    @property
    def core_name(self):
        return "ipa"

    @property
    def core_module_cls(self):
        return IPA
