from typing import List, Tuple, Dict, Any, Union
import torch 
import torch.nn as nn 
from abc import ABCMeta, abstractmethod 
from torch.nn.functional import pad
from copy import deepcopy

from model.bottleneck_resnet import BottleneckResnet3d


def get_all_residuewise_models():
    return {
        'nolayer': NoLayerResidueWiseModel
    }
    
def get_residuewise_model(model_config:Dict[str, Any]) -> 'ResidueWiseModel':
    model_name = model_config['model']
    all_models = get_all_residuewise_models()
    if model_name in all_models:
        model = all_models[model_name](model_config)
    else:
        raise Exception(f'model "{model_name}" not implemented')
    return model

def padstack(l: List[torch.Tensor], pad_value=0) -> torch.Tensor:
    n = len(l[0].shape)
    max_lens = [max(t.shape[i] for t in l)  for i in range(n)]
    padded_l = [pad(x, tuple((max_lens[n-1-i] - x.shape[n-1-i]) * j for i in range(n) for j in [0, 1]), mode='constant', value=pad_value) for x in l]
    return torch.stack(padded_l, dim=0)


class ResidueWiseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
    
    @abstractmethod
    def load_weights(self, src_model_name, state_dict):
        pass 


class NoLayerResidueWiseModel(ResidueWiseModel):
    def __init__(self, config):
        super().__init__(config)
        model_config = config['model_config']
        hidden_dim = model_config.get('hidden_dim', 128)
        dropout_p = model_config.get('dropout_p', 0.1)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.bottleneck_resnet = BottleneckResnet3d(last_channel=18, channels=[64, 128, 256, 512], strides=[2, 2, 2, 1], units=[2, 2, 2, 2])
        self.first_ff = nn.Linear(512, hidden_dim)
        self.act_fn = nn.ReLU()
        self.final_ff = nn.Linear(hidden_dim, 1)
        
    def forward(self, grids:torch.Tensor):
        assert len(grids.shape) == 5
        x = self.bottleneck_resnet(grids)
        x = self.dropout(x)
        x = self.dropout(self.act_fn(self.first_ff(x)))
        assert len(x.shape) == 2
        return self.final_ff(x)[:, 0] #(batch,)
    
    def load_weights(self, src_model_name, state_dict):
        assert src_model_name == 'nolayer'
        self.load_state_dict(state_dict)