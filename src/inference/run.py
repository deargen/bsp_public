from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from inference.dataset import PocketwiseDatasetForInference
from inference.prepare_data import DataPreparation
from model import get_model
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=Path, required=True)
    parser.add_argument("-c", "--cache_dir", type=Path)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pdb_files = list(args.input_dir.glob("*.pdb"))
    assert len(pdb_files) > 0, f"No pdb files found in {args.input_dir}"
    if len(pdb_files) == 0:
        raise Exception(f"No files found with pattern '{args.file_pattern}'")

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

    output_file_name = f"{args.input_dir.name}_inference.h5"
    sub_dir_name = args.input_dir.name
    with DataPreparation(
        args.input_dir,
        output_dir=args.cache_dir,
        output_file_name=output_file_name,
        sub_dir_name=sub_dir_name,
    ) as dp:
        if dp.need_preparation():
            dp.prepare()
        else:
            pass
        assert dp.cache_file.exists()

        dataset = PocketwiseDatasetForInference(dp.cache_file, seq_based=seq_based)
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        code_to_pocket_idx_to_score = {}
        code_to_pocket_idx_to_ca_idx_to_score = {}
        print("Predicting..")
        for batch in tqdm(dataloader):
            if seq_based:
                code_list, pocket_idx_list, x, R, t, mask, ca_idxs_list = batch
                raise NotImplementedError
            else:
                code_list, pocket_idx_list, lens, cat_grids, R, t, ca_idxs_list = batch
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

            assert bsd_out.shape == (len(pocket_idx_list),)
            for bri_out, ca_idxs in zip(bri_out_list, ca_idxs_list, strict=True):
                assert bri_out.shape == (len(ca_idxs),)

            for code, pocket_idx, score in zip(
                code_list, pocket_idx_list, bsd_out, strict=True
            ):
                code_to_pocket_idx_to_score.setdefault(code, {})[pocket_idx] = score
            for code, pocket_idx, bri_out, ca_idxs in zip(
                code_list, pocket_idx_list, bri_out_list, ca_idxs_list, strict=True
            ):
                for ca_idx, score in zip(ca_idxs, bri_out, strict=True):
                    code_to_pocket_idx_to_ca_idx_to_score.setdefault(
                        code, {}
                    ).setdefault(pocket_idx, {})[ca_idx] = score

        print(code_to_pocket_idx_to_ca_idx_to_score)
        print("-------------------------")
        print(code_to_pocket_idx_to_score)

        # TODO: interpret and save the results and remove the above print statements
