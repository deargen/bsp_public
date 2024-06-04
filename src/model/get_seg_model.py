from model.residuewise_models import get_residuewise_model
from model.pocketwise_models import get_pocketwise_model
from model.seq_models import get_seq_model

from pathlib import Path
import json, torch 


def get_seg_model(model_config, dataset_type):
    model_name = model_config['model']

    if dataset_type == 'residue':
        model = get_residuewise_model(model_config)
    elif dataset_type == 'pocket':
        if not model_name.startswith('seq'):
            model = get_pocketwise_model(model_config)
        else:
            model = get_seq_model(model_config)
    else:
        raise Exception(f'No dataset type "{dataset_type}"')
    
    weight_from = model_config.get('weight_from', None)
    if weight_from is not None:
        experiment, version, when = weight_from
        weight_src_model_config = get_model_config(experiment, version)
        weight_src_model_name = weight_src_model_config['model']
        ckpt_file = get_ckpt_file(experiment, version, when)
        state_dict = torch.load(ckpt_file, map_location='cpu')['state_dict']
        state_dict = {name.replace('model.', '', 1): param for name, param in state_dict.items()}
        print(f'Loading pretrained weights from ({experiment}, {version})')
        model_name = model_config['model']
        print(f'({model_name} <- {weight_src_model_name})')
        model.load_weights(weight_src_model_name, state_dict)

    return model

def get_model_config(experiment, version):
    model_config_file = f'./logs/{experiment}/{version}/model_config.json'
    with open(model_config_file, 'r') as f:
        return json.load(f)
          
def get_ckpt_file(experiment, version, when):
    if when == 'last':
        return f'./logs/{experiment}/{version}/scPDB/last.ckpt'
    elif when == 'best':
        ckpt_pattern = Path(f'./logs/{experiment}/{version}/scPDB/epoch=*.ckpt')
        globbed = list(ckpt_pattern.parent.glob(ckpt_pattern.name))
        if len(globbed) == 1:
            return str(globbed[0])
        elif len(globbed) == 0:
            raise Exception(f'No file of pattern {ckpt_pattern}')
        else:
            raise Exception(f'More than one file of pattern {ckpt_pattern}')
        
        
def test_get_model():
    pass


            
        