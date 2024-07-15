from argparse import ArgumentParser
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from inference.dataset import PocketwiseDatasetForInference
from inference.prepare_data import DataPreparation
from model import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, required=True)
    parser.add_argument("-c", "--cache_dir", type=Path)
    parser.add_argument("-o", "--output_path", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--residue_thres", type=float, default=0.5)
    return parser.parse_args()


def sigmoid(x):
    return 1 / (1 + exp(-x))


if __name__ == "__main__":
    args = parse_args()

    if args.input_path.is_dir():
        pdb_files = list(args.input_path.glob("*.pdb"))
        assert len(pdb_files) > 0, f"No pdb files found in {args.input_path}"
        if len(pdb_files) == 0:
            raise Exception(f"No files found with pattern '{args.file_pattern}'")
    else:
        # Input is a hdf5 file
        if not args.input_path.exists():
            raise FileNotFoundError(args.input_path)

    folds = [1, 2, 3, 4, 5]

    print("Loading BSD models..")
    bsd_models = []
    for fold in tqdm(folds):
        model = get_model("main", "bsd", "best", fold, device=args.device)
        model.eval()
        bsd_models.append(model)

    print("Loading BRI models..")
    bri_models = []
    for fold in tqdm(folds):
        model = get_model("main", "bri", "best", fold, device=args.device)
        model.eval()
        bri_models.append(model)

    seq_based = False  # TODO: change this based on the model

    if args.input_path.is_dir():
        h5_cache_file_name = f"{args.input_path.name}_inference.h5"
        sub_dir_name = args.input_path.name
        with DataPreparation(
            args.input_path,
            output_dir=args.cache_dir,
            output_file_name=h5_cache_file_name,
            sub_dir_name=sub_dir_name,
        ) as dp:
            if dp.need_preparation():
                dp.prepare()
            else:
                pass
            assert dp.cache_file.exists()
            h5_cache_file = dp.cache_file
    else:
        h5_cache_file = args.input_path

    dataset = PocketwiseDatasetForInference(h5_cache_file, seq_based=seq_based)
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    code_to_pocket_num_to_score = {}
    code_to_pocket_num_to_center = {}
    code_to_pocket_num_to_ca_idx_to_score = {}
    print("Predicting..")
    code_to_pocket_num_to_ca_idx_to_residue_name = {}
    for batch in tqdm(dataloader):
        # TODO: Output resnames as well
        if seq_based:
            (
                code_list,
                pocket_num_list,
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
                pocket_num_list,
                pocket_center_list,
                lens,
                cat_grids,
                R,
                t,
                ca_idxs_list,
                residue_names_list,
            ) = batch
            for code, pocket_num, pocket_center, ca_idxs, residue_names in zip(
                code_list,
                pocket_num_list,
                pocket_center_list,
                ca_idxs_list,
                residue_names_list,
                strict=True,
            ):
                code_to_pocket_num_to_center.setdefault(code, {})[pocket_num] = (
                    pocket_center
                )
                for ca_idx, residue_name in zip(ca_idxs, residue_names, strict=True):
                    code_to_pocket_num_to_ca_idx_to_residue_name.setdefault(
                        code, {}
                    ).setdefault(pocket_num, {})[ca_idx] = residue_name

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

        assert bsd_out.shape == (len(pocket_num_list),)
        for bri_out, ca_idxs in zip(bri_out_list, ca_idxs_list, strict=True):
            assert bri_out.shape == (len(ca_idxs),)

        for code, pocket_num, score in zip(
            code_list, pocket_num_list, bsd_out, strict=True
        ):
            code_to_pocket_num_to_score.setdefault(code, {})[pocket_num] = sigmoid(
                score
            )
        for code, pocket_num, bri_out, ca_idxs in zip(
            code_list, pocket_num_list, bri_out_list, ca_idxs_list, strict=True
        ):
            for ca_idx, score in zip(ca_idxs, bri_out, strict=True):
                code_to_pocket_num_to_ca_idx_to_score.setdefault(code, {}).setdefault(
                    pocket_num, {}
                )[ca_idx] = sigmoid(score)

    # write csv
    def _get_csv_lines():
        csv_lines = []
        for code, pocket_num_to_score in code_to_pocket_num_to_score.items():
            sorted_pocket_nums = sorted(
                pocket_num_to_score.keys(), key=lambda x: -pocket_num_to_score[x]
            )
            selected_pocket_nums = sorted_pocket_nums[: args.top_n]

            for pocket_rank, pocket_num in enumerate(selected_pocket_nums, start=1):
                pocket_center = code_to_pocket_num_to_center[code][pocket_num]
                pocket_score = pocket_num_to_score[pocket_num]

                ca_idx_to_score = code_to_pocket_num_to_ca_idx_to_score[code][
                    pocket_num
                ]
                ca_idx_to_residue_name = code_to_pocket_num_to_ca_idx_to_residue_name[
                    code
                ][pocket_num]
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
                        pocket_num,
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
            "pocket_num",
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
    code, pocket_num, pocket_score, pocket_rank_within_code, residue_name, residue_score, residue_rank_within_pocket 
    """
