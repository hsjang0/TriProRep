"""
Code adapted from Structtokenbench (https://github.com/KatarinaYuan/StructTokenBench.git).
"""

import torch
import numpy as np

import io
from pathlib import Path
from cloudpathlib import CloudPath
from typing import Union

from biotite.structure.io.pdbx import CIFFile, convert
import biotite.structure as bs
from Bio.Data import PDBData

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils import residue_constants as RC
from esm.utils.structure.normalize_coordinates import normalize_coordinates


PathLike = Union[str, Path, CloudPath]
PathOrBuffer = Union[PathLike, io.StringIO]


class WrappedProteinChain(ProteinChain):

    """Enable cif file loading, similar to loading pdb.
    Reference to from_pdb in https://github.com/evolutionaryscale/esm/blob/f342784d6a4a5488bfb6c9548530d9724531c85c/esm/utils/structure/protein_chain.py#L539
    """

    @classmethod
    def from_cif_list(
        cls,
        path: PathOrBuffer,
        chain_id_list: list,
        id: str | None = None,
        is_predicted: bool = False,
    ) -> list:

        atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"])
        ret = []
        for chain_id in chain_id_list:
            try:
                pdb_chain = cls.from_cif(path, chain_id, id, is_predicted, atom_array)
            except:
                print(f"Cannot retrieve from local cluster", id, chain_id)
                pdb_chain = None
            ret.append(pdb_chain)
        
        return ret

    @classmethod
    def from_cif(
        cls,
        path: PathOrBuffer,
        chain_id: str = "detect",
        id: str | None = None,
        is_predicted: bool = False,
        atom_array=None,
    ) -> "ProteinChain":
        """Return a ProteinStructure object from a cif file.
        """

        if id is not None:
            file_id = id
        else:
            match path:
                case Path() | str():
                    file_id = Path(path).with_suffix("").name
                case _:
                    file_id = "null"
        
        if atom_array is None:
            atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"])
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        if not (atom_array.chain_id == chain_id).any():
            atom_array = convert.get_structure(CIFFile.read(path), model=1, 
                                extra_fields=["b_factor"], use_author_fields=False)

        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        entity_id = 1  # Not supplied in PDBfiles

        sequence = "".join(
            (
                r
                if len(r := PDBData.protein_letters_3to1.get(monomer[0].res_name, "X"))
                == 1
                else "X"
            )
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, RC.atom_type_num, 3],
            np.nan,
            dtype=np.float32,
        )
        atom_mask = np.full(
            [num_res, RC.atom_type_num],
            False,
            dtype=bool,
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)
        insertion_code = np.full([num_res], "", dtype="<U4")

        confidence = np.ones(
            [num_res],
            dtype=np.float32,
        )

        for i, res in enumerate(bs.residue_iter(atom_array)):
            chain = atom_array[atom_array.chain_id == chain_id]
            assert isinstance(chain, bs.AtomArray)

            res_index = res[0].res_id
            residue_index[i] = res_index
            insertion_code[i] = res[0].ins_code

            # Atom level features
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    atom_name = "SD"

                if atom_name in RC.atom_order:
                    atom_positions[i, RC.atom_order[atom_name]] = atom.coord
                    atom_mask[i, RC.atom_order[atom_name]] = True
                    if is_predicted and atom_name == "CA":
                        confidence[i] = atom.b_factor

        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
        )
        

    def to_structure_encoder_inputs(
        self,
        device="cpu",
        should_normalize_coordinates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.tensor(self.atom37_positions, dtype=torch.float32, device=device)
        plddt = torch.tensor(self.confidence, dtype=torch.float32, device=device)
        residue_index = torch.tensor(self.residue_index, dtype=torch.long, device=device)

        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)