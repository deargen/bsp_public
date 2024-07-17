import re
import shutil
import subprocess as sp
import warnings
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import tfbio_data
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue
from openbabel import pybel
from tqdm import tqdm
from utils.frames import to_frame


class DataPreparation:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path | None = None,
        output_file_name: str | None = None,
        sub_dir_name: str | None = None,
        gridSize=16,
        voxelSize=1.0,
        CA_within=17.0,
    ):
        pdb_files = list(input_dir.glob("*.pdb"))
        assert len(pdb_files) > 0, f"No pdb files found in {input_dir}"
        self.pdb_files = pdb_files
        self.provided_output_dir = output_dir
        self.provided_output_file_name = output_file_name
        self.sub_dir_name = sub_dir_name

        self.gridSize = gridSize
        self.voxelSize = voxelSize
        self.CA_within = CA_within

        self.cache_file: Path | None = None

    def __enter__(self):
        if self.provided_output_dir is None:
            self.tempdir = TemporaryDirectory(".")
            self.output_dir = Path(self.tempdir.name)
        else:
            self.tempdir = None
            self.output_dir = self.provided_output_dir
            self.output_dir.mkdir(exist_ok=True, parents=True)
        output_file_name = (
            "inference_cache.h5"
            if self.provided_output_file_name is None
            else self.provided_output_file_name
        )

        self.cache_file = self.output_dir / output_file_name

        self.sub_dir = (
            self.output_dir
            if self.sub_dir_name is None
            else self.output_dir / self.sub_dir_name
        )
        self.sub_dir.mkdir(exist_ok=True, parents=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def need_preparation(self):
        return not self.cache_file.exists()

    def prepare(self):
        if self.cache_file.exists():
            raise FileExistsError(f"File {self.cache_file} already exists")

        with h5py.File(self.cache_file, "w") as f:
            errored = False
            for pdb_file in tqdm(self.pdb_files):
                try:
                    prot_name = pdb_file.stem
                    copied_pdb_file = self.sub_dir / pdb_file.name
                    shutil.copy(pdb_file, copied_pdb_file)
                    sp.call(["fpocket", "-f", str(copied_pdb_file)])
                    fpocket_out_dir = self.sub_dir / f"{copied_pdb_file.stem}_out"
                    if not fpocket_out_dir.exists():
                        raise FileNotFoundError(
                            f"Directory {fpocket_out_dir} does not exist"
                        )
                    pocket_centers = get_pocket_centers(fpocket_out_dir)

                    featurizer = tfbio_data.Featurizer(save_molecule_codes=False)
                    pdb_parser = PDBParser(PERMISSIVE=True, QUIET=True)

                    f.create_group(prot_name)
                    prot_gp = f[prot_name]

                    mols = list(pybel.readfile("pdb", str(copied_pdb_file)))
                    assert len(mols) == 1
                    mol = mols[0]
                    mol.removeh()

                    atom_coords, atom_features = featurizer.get_features(mol)
                    atom_coords = atom_coords.astype(np.float32)
                    atom_features = atom_features.astype(np.float32)

                    prot_gp["atom_coords"] = atom_coords.astype(np.float16)
                    prot_gp["atom_features"] = atom_features.astype(np.float16)

                    prot_gp.attrs["num_pockets"] = len(pocket_centers)

                    CA_info_list = get_CA_info_list(pdb_parser, pdb_file)

                    for i, pocket_center in enumerate(pocket_centers, start=0):
                        prot_gp.create_group(f"pocket_{i}")
                        pocket_gp = prot_gp[f"pocket_{i}"]
                        pocket_gp["center"] = pocket_center

                        num_cas = 0
                        potential_atom_idxs_list = []
                        for t, R, heavy_coords, residue_name in CA_info_list:
                            dist = np.linalg.norm(pocket_center - t)
                            if dist > self.CA_within:
                                continue
                            pocket_gp.create_group(f"CA_{num_cas}")
                            CA_gp = pocket_gp[f"CA_{num_cas}"]

                            CA_gp["coord"] = t
                            CA_gp["orientation"] = R
                            CA_gp.attrs["residue_name"] = residue_name

                            potential_atom_idxs = get_potential_atom_idxs(
                                t,
                                atom_coords,
                                gridSize=self.gridSize,
                                voxelSize=self.voxelSize,
                            )
                            potential_atom_idxs_list.append(potential_atom_idxs)

                            num_cas += 1
                        pocket_gp.attrs["num_cas"] = num_cas
                        if len(potential_atom_idxs_list) == 0:
                            unioned_potential_atom_idxs = np.array([], dtype=np.int)
                        else:
                            unioned_potential_atom_idxs = np.unique(
                                np.concatenate(potential_atom_idxs_list)
                            )
                        pocket_gp["potential_atom_idxs"] = (
                            unioned_potential_atom_idxs.astype(np.uint16)
                        )

                        for i, potential_atom_idxs in enumerate(
                            potential_atom_idxs_list
                        ):
                            CA_gp = pocket_gp[f"CA_{i}"]
                            sub_atom_idxs = get_sub_idxs(
                                unioned_potential_atom_idxs, potential_atom_idxs
                            )
                            CA_gp["sub_atom_idxs"] = sub_atom_idxs.astype(np.uint16)
                except Exception as e:
                    raise e
                    errored = True
                    break
        if errored:
            self.cache_file.unlink()


def get_potential_atom_idxs(
    CA_coord: np.ndarray, atom_coords: np.ndarray, gridSize=16, voxelSize=1.0
):
    # TODO: unit-test this
    assert CA_coord.shape == (3,)
    assert atom_coords.shape[1] == 3
    max_dist = (gridSize / 2) * voxelSize * np.sqrt(3)
    dists = get_dists_1d(atom_coords, CA_coord)
    return np.where(dists <= max_dist)[0]


def get_pocket_centers(fpocket_output_dir: Path):
    fpocket_pocket_dir = fpocket_output_dir / "pockets"
    pocket_pqr_files = list(fpocket_pocket_dir.glob("pocket*_vert.pqr"))
    pocket_pqr_files.sort(
        key=lambda file: int(next(re.finditer("\d+", file.name)).group())
    )
    pocket_centers = np.array([get_center(file) for file in pocket_pqr_files])
    return pocket_centers


def get_center(pqr_file):
    """
    Originally from DeepPocket/get_centers.py
    """
    with open(pqr_file, "r") as f:
        centers = []
        masses = []
        for line in f:
            if line.startswith("ATOM"):
                center = list(
                    map(
                        float,
                        re.findall(
                            "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                            " ".join(line.split()[5:]),
                        ),
                    )
                )[:3]
                mass = float(line.split()[-1])
                centers.append(center)
                masses.append(mass)
        centers = np.asarray(centers)
        masses = np.asarray(masses)
        xyzm = (centers.T * masses).T
        xyzm_sum = xyzm.sum(axis=0)  # find the total xyz*m for each element
        cg = xyzm_sum / masses.sum()
    return cg


def get_CA_info_list(pdb_parser: PDBParser, protein_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pdb_parser.get_structure("", protein_file)[0]
    t_list, R_list, heavy_coords_list, residue_names = [], [], [], []
    for chain in model.child_list:
        for residue in chain:
            residue: Residue
            if "CA" not in residue:
                continue
            if "N" not in residue:
                continue
            if "C" not in residue:
                continue
            N_coord = residue["N"].coord
            CA_coord = residue["CA"].coord
            C_coord = residue["C"].coord
            frame = to_frame(N_coord, CA_coord, C_coord)
            R_list.append(frame["R"])
            t_list.append(frame["t"])
            heavy_coords = np.stack(
                [atom.coord for atom in residue if atom.element != "H"]
            )
            heavy_coords_list.append(heavy_coords)

            residue_name = f"{residue.parent.id}/{residue.resname}_{residue.id[1]}"
            residue_names.append(residue_name)

    return list(zip(t_list, R_list, heavy_coords_list, residue_names))


def get_sub_idxs(arr: np.ndarray, subarr: np.ndarray):
    """
    arr: [1, 2, 3, 5, 6, 7]
    subarr: [1, 5, 7]
    return: [0, 3, 5]
    """
    assert len(arr.shape) == len(arr.shape) == 1
    assert len(arr) >= len(subarr)
    subarr_mask = np.any(arr[:, None] == subarr[None, :], axis=1)
    return np.where(subarr_mask)[0]


def get_dists_1d(coords1: np.ndarray, coord2: np.ndarray):
    return np.sqrt(np.sum((coords1 - coord2[None, :]) ** 2, axis=1))


def get_dists_2d(coords1: np.ndarray, coords2: np.ndarray):
    return np.sqrt(np.sum((coords1[:, None, :] - coords2[None, :, :]) ** 2, axis=2))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=Path, required=True)
    parser.add_argument("--cache_dir", "-c", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with DataPreparation(args.input_dir, output_dir=args.cache_dir) as dp:
        dp.prepare()
