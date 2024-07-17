import json
from pathlib import Path

import torch

from model.detection_models import get_detection_model
from model.pocketwise_models import get_pocketwise_model
from model.residuewise_models import get_residuewise_model
from model.seq_models import get_seq_model


def load_json(filename):
    with open(filename, "r") as reader:
        return json.load(reader)


def get_train_dataset_config(experiment, version, fold=None):
    foldname = get_foldname(fold)
    dataset_config_file = (
        f"./logs/{experiment}/{version}/scPDB_{foldname}/dataset_config.json"
    )
    return load_json(dataset_config_file)["trn"]


def get_all_scPDB_folds(experiment, version):
    dir = Path(f"./logs/{experiment}/{version}")
    folds = []
    for f in dir.glob("scPDB_*"):
        fold = f.stem[len("scPDB_") :]
        folds.append(fold)
    return folds


def get_train_config(experiment, version, fold=None):
    foldname = get_foldname(fold)
    train_config_file = (
        f"./logs/{experiment}/{version}/scPDB_{foldname}/train_config.json"
    )
    return load_json(train_config_file)


def get_foldname(fold):
    if isinstance(fold, int):
        return f"fold{fold}"
    return fold


def get_model(
    experiment, version, when, fold=None, device="cuda:0", return_model_name=False
):
    """
    Load model on device from the given checkpoint
    """

    foldname = get_foldname(fold)

    def get_ckpt_file():
        if when == "last":
            return f"./logs/{experiment}/{version}/scPDB_{foldname}/last.ckpt"
        elif when == "best":
            ckpt_pattern = Path(
                f"./logs/{experiment}/{version}/scPDB_{foldname}/epoch=*.ckpt"
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
    model_config["weight_from"] = (experiment, version, when)
    model_name = model_config["model"]

    train_dataset_config = get_train_dataset_config(experiment, version, fold=fold)
    dataset_type = train_dataset_config["type"]
    train_config = get_train_config(experiment, version, fold=fold)
    detection = train_config.get("detection")
    if dataset_type == "pocket":
        if detection:
            model = get_detection_model(model_config, train_dataset_config)
        else:
            if not model_name.startswith("seq"):
                model = get_pocketwise_model(model_config)
            else:
                model = get_seq_model(model_config)
    elif dataset_type == "residue":
        model = get_residuewise_model(model_config)
    model.to(device=device)
    model.eval()

    ckpt_file = get_ckpt_file()
    state_dict = torch.load(ckpt_file, map_location=device)["state_dict"]
    model_state_dict = {
        name.replace("model.", "", 1): param for name, param in state_dict.items()
    }

    model.load_state_dict(model_state_dict)

    if return_model_name:
        return model, model_name

    return model
