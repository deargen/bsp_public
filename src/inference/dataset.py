import random
from pathlib import Path

import h5py
import numpy as np
import torch
from tfbio_data import make_grid
from torch.nn.functional import pad
from torch.utils.data import Dataset
from utils.random_rotations import get_random_rotation


def padstack(l: list[torch.Tensor], pad_value=0) -> torch.Tensor:  # noqa: E741
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


def padstack_identity(l: list[torch.Tensor]) -> torch.Tensor:  # noqa: E741
    assert l[0].shape[1:] == (3, 3)
    max_len = max(x.shape[0] for x in l)
    padded_l = [
        torch.cat(
            [
                x,
                torch.eye(3, device=x.device)[None, :, :].broadcast_to(
                    max_len - len(x), 3, 3
                ),
            ],
            dim=0,
        )
        for x in l
    ]

    return torch.stack(padded_l, dim=0)


class PocketwiseDatasetForInference(Dataset):
    def __init__(
        self,
        cache_file: str | Path,
        manual_pocket_names: dict[str, list[str]] | None = None,
        mode="test",
        CA_within=17.0,
        grid_rotation=None,
        frame_rotation=None,
        t_relpos=True,
        rdo=None,
        seq_based=False,
    ):
        self.seq_based = seq_based
        cache_file = Path(cache_file)
        if not cache_file.exists():
            raise FileNotFoundError(f"File {cache_file} does not exist")
        self.cache_file = cache_file

        self.CA_within = CA_within

        if mode in ["dev", "test"]:
            assert grid_rotation is None
        self.grid_rotation = grid_rotation
        if mode in ["dev", "test"]:
            assert frame_rotation is None
            assert rdo is None
        self.frame_rotation = frame_rotation

        self.t_relpos = t_relpos

        if rdo is not None:
            assert 0 < rdo < 1
        self.rdo = rdo

        self.manual_pocket_names = manual_pocket_names
        self.pocket_paths = self.get_pocket_paths()

    def get_pocket_paths(self):
        pocket_paths = []
        with h5py.File(self.cache_file, "r") as hdf:
            for code in hdf.keys():
                f = hdf[code]
                num_pockets = f.attrs["num_pockets"]
                manual_pocket_centers = f.attrs.get("manual_pocket_centers", False)
                if manual_pocket_centers:
                    if self.manual_pocket_names is None:
                        raise ValueError(
                            "manual_pocket_centers is True, but manual_pocket_names is None"
                        )
                    if code not in self.manual_pocket_names:
                        raise ValueError(
                            f"manual_pocket_centers is True, but {code} not in manual_pocket_names {self.manual_pocket_names}"
                        )
                    for pocket_name in self.manual_pocket_names[code]:
                        pocket_path = f"/{code}/{pocket_name}"
                        pocket_paths.append(pocket_path)
                else:
                    pocket_num_starts_from = f.attrs.get("pocket_num_starts_from", 0)
                    for pocket_num in range(
                        pocket_num_starts_from, pocket_num_starts_from + num_pockets
                    ):
                        pocket_name = f"pocket_{pocket_num}"
                        assert pocket_name in f
                        pocket_path = f"/{code}/{pocket_name}"
                        pocket_paths.append(pocket_path)
        return pocket_paths

    def __len__(self):
        return len(self.pocket_paths)

    def __getitem__(self, i):
        """
        pocket_path: e.g. '/1a2b/pocket_1'
        """

        if not hasattr(self, "hdf"):
            self.hdf = h5py.File(self.cache_file, "r")

        pocket_path = self.pocket_paths[i]
        _, code, *pocket_name_pieces = pocket_path.split("/")
        pocket_name = "/".join(pocket_name_pieces)
        pocket_gp = self.hdf[pocket_path]
        pocket_center = pocket_gp["center"][:]

        if not self.seq_based:
            return (code, pocket_name, pocket_center), self.getitem_voxel_based(
                pocket_path
            )
        else:
            return (code, pocket_name, pocket_center), self.getitem_seq_based(
                pocket_path
            )

    def getitem_voxel_based(self, pocket_path):
        prot_gp = self.hdf["/".join(pocket_path.split("/")[:2])]
        pocket_gp = self.hdf[pocket_path]
        center = pocket_gp["center"][:]

        pocket_atom_idxs = pocket_gp["potential_atom_idxs"][:]
        pocket_atom_coords = prot_gp["atom_coords"][pocket_atom_idxs]
        pocket_atom_features = prot_gp["atom_features"][pocket_atom_idxs]

        num_CAs = pocket_gp.attrs["num_cas"]
        _potential_ca_idxs = []
        for i in range(num_CAs):
            residue_gp = pocket_gp[f"CA_{i}"]
            t = residue_gp["coord"][:]
            dist_from_center = np.linalg.norm(t - center)
            if dist_from_center < self.CA_within:
                _potential_ca_idxs.append(i)
        if self.rdo is None:
            potential_ca_idxs = _potential_ca_idxs
        else:
            potential_ca_idxs = []
            for i in _potential_ca_idxs:
                if random.uniform(0, 1) >= self.rdo:  # with prob 1-self.rdo
                    potential_ca_idxs.append(i)
            if potential_ca_idxs == []:
                potential_ca_idxs = [random.choice(_potential_ca_idxs)]
        assert (
            len(potential_ca_idxs) >= 1
        )  # Those rare cases num_CAs == 0 are filtered out in the sample file (thus by the sampler)

        grids = []
        ts = []
        Rs = []
        ca_idxs = []
        residue_names = []

        for i in potential_ca_idxs:
            residue_gp = pocket_gp[f"CA_{i}"]

            # local frame
            t = residue_gp["coord"][:]  # in Angstroms
            R = residue_gp["orientation"][:]
            if "residue_name" in residue_gp.attrs:
                residue_name = residue_gp.attrs["residue_name"]
            else:
                # TODO: Change this?
                pocket_name = pocket_path.split("/")[-1]
                residue_name = f"{pocket_name}-res_{i}"
            if self.frame_rotation is not None:
                if not (
                    isinstance(self.frame_rotation, float)
                    and 0 < self.frame_rotation < np.pi
                ):
                    raise Exception(self.frame_rotation)
                assert (
                    isinstance(self.frame_rotation, float)
                    and 0 < self.frame_rotation < np.pi
                )
                R0 = get_random_rotation(1, max_angle=self.frame_rotation)[0]
                R = np.matmul(R, R0)

            # coordinates and features
            sub_idxs = residue_gp["sub_atom_idxs"]
            atom_coords = pocket_atom_coords[sub_idxs]
            atom_features = pocket_atom_features[sub_idxs]

            # translation and rotation
            atom_coords = atom_coords - t[None, :]
            if self.grid_rotation is None:
                atom_coords = np.matmul(
                    atom_coords, R
                )  # np.matmul(R.T, atom_coords.T).T
            elif self.grid_rotation == "random":
                random_rotation = get_random_rotation(1, max_angle=None)[0]
                atom_coords = np.matmul(atom_coords, random_rotation)
            else:
                assert isinstance(self.grid_rotation, float)
                further_rotation = get_random_rotation(1, max_angle=self.grid_rotation)[
                    0
                ]
                atom_coords = np.matmul(atom_coords, np.matmul(R, further_rotation.T))

            # grid features
            grid = torch.from_numpy(
                make_grid(atom_coords, atom_features, grid_resolution=1.0, max_dist=7.5)
            )
            assert grid.dtype == torch.float32

            grids.append(grid)

            # append local frames
            if self.t_relpos:
                t = t - center.astype(np.float32)
            ts.append(
                torch.from_numpy(t * 0.1)
            )  # Angstroms -> nanometers, as in Alphafold
            Rs.append(torch.from_numpy(R))
            residue_names.append(residue_name)

            ca_idxs.append(i)

        grids = torch.stack(grids, dim=0)
        ts = torch.stack(ts, dim=0)
        Rs = torch.stack(Rs, dim=0)

        return grids, Rs, ts, ca_idxs, residue_names

    def collate_fn(self, data_list):
        code_list = [x[0][0] for x in data_list]
        pocket_name_list = [x[0][1] for x in data_list]
        pocket_center_list = [x[0][2] for x in data_list]
        data_list = [x[1] for x in data_list]
        if not self.seq_based:
            lens = [len(x[0]) for x in data_list]
            cat_grids = torch.cat([x[0] for x in data_list], dim=0)

            R = padstack_identity([x[1] for x in data_list])
            t = padstack([x[2] for x in data_list])

            ca_idxs_list = [x[3] for x in data_list]
            residue_names_list = [x[4] for x in data_list]

            return (
                code_list,
                pocket_name_list,
                pocket_center_list,
                lens,
                cat_grids,
                R,
                t,
                ca_idxs_list,
                residue_names_list,
            )
        else:
            # xs, Rs, ts, mask, ca_idxs
            x = padstack([a[0] for a in data_list])
            R = padstack_identity([a[1] for a in data_list])
            t = padstack([a[2] for a in data_list])
            mask = padstack([a[3] for a in data_list])

            ca_idxs_list = [a[4] for a in data_list]

            return (
                code_list,
                pocket_name_list,
                pocket_center_list,
                x,
                R,
                t,
                mask,
                ca_idxs_list,
            )

    def getitem_seq_based(self, pocket_path):
        if self.CA_within != 17:
            raise Exception("Not implemented yet")

        pocket_gp = self.hdf[pocket_path]
        center = pocket_gp["center"][:].astype(np.float32)

        xs = np.array(
            [self.aa_to_idx[chr(aa)] for aa in pocket_gp["aas"][:]], dtype=np.int64
        )  # stored in ascii ordinal
        ts = pocket_gp["ts"][:].astype(np.float32)
        Rs = pocket_gp["Rs"][:].astype(np.float32)
        assert len(xs) == len(ts) == len(Rs)
        n = len(xs)
        ca_idxs = np.arange(n)

        if self.rdo is not None:
            assert isinstance(self.rdo, float)
            selected_idxs = np.random.uniform(0, 1, (n,)) >= self.rdo
            if np.all(~selected_idxs):
                selected_idxs = np.arange(n) == np.random.choice(range(n))
            xs = xs[selected_idxs]
            ts = ts[selected_idxs, :]
            Rs = Rs[selected_idxs, :, :]
            ca_idxs = ca_idxs[selected_idxs]
            n = len(xs)

        assert self.grid_rotation is None

        if self.frame_rotation is not None:
            if not (
                isinstance(self.frame_rtation, float)
                and 0 < self.frame_rotation < np.pi
            ):
                raise Exception(self.frame_rotation)
            assert (
                isinstance(self.frame_rtation, float)
                and 0 < self.frame_rotation < np.pi
            )
            R0 = get_random_rotation(n, max_angle=self.frame_rotation).astype(
                np.float32
            )  # n independent rotations
            Rs = np.matmul(R0, Rs)

        if self.t_relpos:
            ts = 0.1 * (
                ts - center[None, :]
            )  # Angstroms -> nanometers, as in Alphafold
        else:
            ts = 0.1 * ts  # Angstroms -> nanometers, as in Alphafold

        xs, Rs, ts, ca_idxs = (
            torch.from_numpy(xs),
            torch.from_numpy(Rs),
            torch.from_numpy(ts),
            ca_idxs.tolist(),
        )

        mask = torch.ones(
            (len(xs),), dtype=torch.bool, device=xs.device
        )  # TODO: need to check if this is correct

        return xs, Rs, ts, mask, ca_idxs
