import os, sys

# Include the path where the "src" of allodb is; w.r.t. the location of this file, it is 2 folders behind
# '__file__' is used as "../../" doesn't work because it uses the location of the script/notebook that imports this module, and not the location of this module
sys.path.append(
    os.path.dirname( os.path.abspath(__file__) )
    .rsplit("/", 2)[0]
    + "/database"
)

from src.utils import Cif as BaseCifFile
from src.cifutils import MMCIF2Dict, CifFileWriter

import tempfile, pickle
from functools import cached_property, cache
from contextlib import contextmanager
import pandas as pd
from tqdm.notebook import tqdm


class CifFile(BaseCifFile):
    """
    Base class to manage .cif files. Inherits from allodb.utils Cif but overrides intialization to read from a file.
    """

    def __init__(self, parent, filename=None, name=None):
        self.filename = filename or parent.filename
        self._name = name or parent._name
        
    @cached_property
    def data(self) -> dict:
        return MMCIF2Dict().parse(self.filename)[self._name]



class Cif:
    """
    Base class to work with a structure from a file: a "minimal" structure which also has a parent "original" file. Mimicks allodb.utils Structure class.
    """
    
    path = "."
    original_cifs_path = "."
    
    def __init__(self, pdb, filename=None, name=None):
        self.entry_id = pdb.lower()
        self._name = name or self.entry_id.upper()
        self.filename = filename or f"{self.path}/{self.entry_id}.cif"
        self.orig_filename = f"{self.original_cifs_path}/{self.entry_id}_updated.cif.gz"

    @cached_property
    def cif(self):
        return CifFile(self)

    @cached_property
    def origcif(self):
        return CifFile(self, filename=self.orig_filename)

    @cached_property
    def atoms(self) -> pd.DataFrame:
        return pd.DataFrame(self.cif.data["_atom_site"], dtype=str)
    
    @cached_property
    def residues(self) -> pd.DataFrame:
        return (
            self.atoms.drop(
                columns=[
                    c for c in self.atoms.columns if c not in [
                        'label_comp_id', 'label_asym_id', 'label_entity_id', 'label_seq_id',
                        'pdbx_PDB_ins_code', 'auth_seq_id', 'auth_comp_id', 'auth_asym_id',
                        'pdbx_PDB_model_num', 'pdbx_label_index', 'pdbx_sifts_xref_db_name',
                        'pdbx_sifts_xref_db_acc', 'pdbx_sifts_xref_db_num',
                        'pdbx_sifts_xref_db_res'
                    ]
                ]
            )
            .drop_duplicates()
        )

    @contextmanager
    def _extended_temp_ciff(self, extra_data: dict):
        """
        To be used as: 'with <obj>._extended_temp_ciff(extra_data) as f:' to work with a cif file with the data of the current object plus the extra_data passed
        """
        data = self.cif.data
        data.update(extra_data)
        try:
            file = tempfile.NamedTemporaryFile("w+", suffix=".cif")
            writer = CifFileWriter(file.name)
            writer.write({self.entry_id.upper(): data})
            yield file
        finally:
            if file:
                file.close()