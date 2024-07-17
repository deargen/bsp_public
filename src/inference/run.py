from argparse import ArgumentParser
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from inference.dataset import PocketwiseDatasetForInference
from inference.prepare_data import DataPreparation
from model import get_all_scPDB_folds, get_model
from prepare_data import read_pocket_center_file
from torch.utils.data import DataLoader
from tqdm import tqdm


def _int_or_inf(x):
    if x == "inf":
        return float("inf")
    return int(x)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, required=True)
    parser.add_argument("-pc", "--pocket_center_filename", type=str)
    parser.add_argument("-c", "--cache_dir", type=Path)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--top_n", type=_int_or_inf, default=5)
    parser.add_argument("--residue_thres", type=float, default=0.5)
    parser.add_argument("-mv", "--model_version", type=str, default="main")
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + exp(-x))


if __name__ == "__main__":
    args = parse_args()
    train_log_dir = Path("logs") / args.model_version
    if not train_log_dir.exists():
        raise FileNotFoundError(train_log_dir)

    if args.input_path.is_dir():
        pdb_files = list(args.input_path.glob("*.pdb"))
        assert len(pdb_files) > 0, f"No pdb files found in {args.input_path}"
        if len(pdb_files) == 0:
            raise Exception(f"No files found with pattern '{args.file_pattern}'")
    else:
        # Input is a hdf5 file
        if not args.input_path.exists():
            raise FileNotFoundError(args.input_path)

    print("Loading BSD models..")
    bsd_models = []
    bsd_folds = get_all_scPDB_folds(args.model_version, "bsd")
    for fold in tqdm(bsd_folds):
        model = get_model(args.model_version, "bsd", "best", fold, device=args.device)
        model.eval()
        bsd_models.append(model)

    print("Loading BRI models..")
    bri_models = []
    bri_folds = get_all_scPDB_folds(args.model_version, "bri")
    for fold in tqdm(bri_folds):
        model = get_model(args.model_version, "bri", "best", fold, device=args.device)
        model.eval()
        bri_models.append(model)

    seq_based = False  # TODO: change this based on the model

    if args.input_path.is_dir():
        pdb_input_path = args.input_path
        h5_cache_file_name = f"{args.input_path.name}_inference.h5"
        h5_cache_dir = args.cache_dir
        h5_cache_sub_dir_name = args.input_path.name
        pocket_center_file = (
            None
            if args.pocket_center_filename is None
            else args.input_path / args.pocket_center_filename
        )
    else:
        if args.pocket_center_filename is not None:
            raise ValueError(
                "pocket_center_file should be None if inferencing from a precomputed cache"
            )
        pocket_center_file = None
        pdb_input_path = None
        h5_cache_file_name = args.input_path.name
        h5_cache_dir = args.input_path.parent
        h5_cache_sub_dir_name = None

    with DataPreparation(
        pdb_input_path,
        output_dir=h5_cache_dir,
        output_file_name=h5_cache_file_name,
        sub_dir_name=h5_cache_sub_dir_name,
        pocket_center_file=pocket_center_file,
    ) as dp:
        if dp.need_preparation():
            dp.prepare()
        else:
            pass
        assert dp.cache_file.exists()

        if pocket_center_file is not None:
            pocket_centers_dict = read_pocket_center_file(pocket_center_file)
        else:
            pocket_centerse_dict = dp.pocket_centers_dict
        manual_pocket_names = {
            prot_name: list(d.keys()) for prot_name, d in pocket_centers_dict.items()
        }
        dataset = PocketwiseDatasetForInference(
            dp.cache_file, manual_pocket_names=manual_pocket_names, seq_based=seq_based
        )
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        code_to_pocket_name_to_score = {}
        code_to_pocket_name_to_center = {}
        code_to_pocket_name_to_ca_idx_to_score = {}
        print("Predicting..")
        code_to_pocket_name_to_ca_idx_to_residue_name = {}
        for batch in tqdm(dataloader):
            # TODO: Output resnames as well
            if seq_based:
                (
                    code_list,
                    pocket_name_list,
                    pocket_center_list,
                    x,
                    R,
                    t,
                    mask,
                    ca_idxs_list,
                ) = batch
                raise NotImplementedError
            else:
                (
                    code_list,
                    pocket_name_list,
                    pocket_center_list,
                    lens,
                    cat_grids,
                    R,
                    t,
                    ca_idxs_list,
                    residue_names_list,
                ) = batch
                for code, pocket_name, pocket_center, ca_idxs, residue_names in zip(
                    code_list,
                    pocket_name_list,
                    pocket_center_list,
                    ca_idxs_list,
                    residue_names_list,
                    strict=True,
                ):
                    code_to_pocket_name_to_center.setdefault(code, {})[pocket_name] = (
                        pocket_center
                    )
                    for ca_idx, residue_name in zip(
                        ca_idxs, residue_names, strict=True
                    ):
                        code_to_pocket_name_to_ca_idx_to_residue_name.setdefault(
                            code, {}
                        ).setdefault(pocket_name, {})[ca_idx] = residue_name

                cat_grids = cat_grids.to(args.device)
                R = R.to(args.device)
                t = t.to(args.device)
                bsd_outs = []
                bri_out_lists = []
                with torch.no_grad():
                    for bsd_model in bsd_models:
                        out = bsd_model(lens, cat_grids, R, t).cpu().numpy()
                        bsd_outs.append(out)
                    for bri_model in bri_models:
                        out = bri_model(lens, cat_grids, R, t)
                        assert len(out) == len(lens)
                        bri_out_list = [
                            x[:n].cpu().numpy() for x, n in zip(out, lens, strict=True)
                        ]
                        bri_out_lists.append(bri_out_list)

                bsd_out = np.mean(bsd_outs, axis=0)
                bri_out_list = [np.mean(x, axis=0) for x in zip(*bri_out_lists)]

            assert bsd_out.shape == (len(pocket_name_list),)
            for bri_out, ca_idxs in zip(bri_out_list, ca_idxs_list, strict=True):
                assert bri_out.shape == (len(ca_idxs),)

            for code, pocket_name, score in zip(
                code_list, pocket_name_list, bsd_out, strict=True
            ):
                code_to_pocket_name_to_score.setdefault(code, {})[pocket_name] = (
                    sigmoid(score)
                )
            for code, pocket_name, bri_out, ca_idxs in zip(
                code_list, pocket_name_list, bri_out_list, ca_idxs_list, strict=True
            ):
                for ca_idx, score in zip(ca_idxs, bri_out, strict=True):
                    code_to_pocket_name_to_ca_idx_to_score.setdefault(
                        code, {}
                    ).setdefault(pocket_name, {})[ca_idx] = sigmoid(score)

    # write csv
    def _get_csv_lines():
        csv_lines = []
        for code, pocket_name_to_score in code_to_pocket_name_to_score.items():
            sorted_pocket_names = sorted(
                pocket_name_to_score.keys(), key=lambda x: -pocket_name_to_score[x]
            )
            num_select = min(args.top_n, len(sorted_pocket_names))
            selected_pocket_names = sorted_pocket_names[:num_select]

            for pocket_rank, pocket_name in enumerate(selected_pocket_names, start=1):
                pocket_center = code_to_pocket_name_to_center[code][pocket_name]
                pocket_score = pocket_name_to_score[pocket_name]

                ca_idx_to_score = code_to_pocket_name_to_ca_idx_to_score[code][
                    pocket_name
                ]
                ca_idx_to_residue_name = code_to_pocket_name_to_ca_idx_to_residue_name[
                    code
                ][pocket_name]
                residue_name_to_ca_idx = {
                    v: k for k, v in ca_idx_to_residue_name.items()
                }
                try:
                    sorted_residue_names = sorted(
                        ca_idx_to_residue_name.values(),
                        key=lambda residue_name: (
                            int(residue_name.split("_")[1]),
                            residue_name.split("_")[0],
                        ),
                    )
                except ValueError:
                    sorted_residue_names = sorted(ca_idx_to_residue_name.values())
                selected_residue_names = [
                    residue_name
                    for residue_name in sorted_residue_names
                    if ca_idx_to_score[residue_name_to_ca_idx[residue_name]]
                    >= args.residue_thres
                ]
                score_sorted_residue_names = sorted(
                    selected_residue_names,
                    key=lambda residue_name: -ca_idx_to_score[
                        residue_name_to_ca_idx[residue_name]
                    ],
                )

                for residue_name in selected_residue_names:
                    ca_idx = residue_name_to_ca_idx[residue_name]
                    residue_score = ca_idx_to_score[ca_idx]
                    residue_rank = score_sorted_residue_names.index(residue_name) + 1
                    csv_line = [
                        code,
                        pocket_name,
                        round(pocket_center[0], 2),
                        round(pocket_center[1], 2),
                        round(pocket_center[2], 2),
                        round(pocket_score, 2),
                        pocket_rank,
                        residue_name,
                        round(residue_score, 2),
                        residue_rank,
                    ]
                    csv_lines.append(csv_line)
        return csv_lines

    csv_lines = _get_csv_lines()

    df = pd.DataFrame(
        csv_lines,
        columns=[
            "code",
            "pocket_name",
            "pocket_center_x",
            "pocket_center_y",
            "pocket_center_z",
            "pocket_score",
            "pocket_rank_within_code",
            "residue_name",
            "residue_score",
            "residue_rank_within_pocket",
        ],
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    """ 
    code, pocket_name, pocket_score, pocket_rank_within_code, residue_name, residue_score, residue_rank_within_pocket 
    """
