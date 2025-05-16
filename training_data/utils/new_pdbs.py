import sys, os
# Include the path where the "src" of allodb is; w.r.t. the location of this file, it is 2 folders behind
# '__file__' is used as "../../" doesn't work because it uses the location of the script/notebook that imports this module, and not the location of this module
sys.path.append(
    os.path.dirname( os.path.abspath(__file__) )
    .rsplit("/", 2)[0]
)

from src.utils import Assembly


from .utils import *
import requests, gzip, tempfile


class PDBCif:
    """
    Base class to manage .cif files of new PDBs not in allodb. Mimicks CifFile but retrieves the .cif content from online, and has other utils to manipulate the downloaded information.
    """
    def __init__(self, parent, filename=None, name=None):
        self.filename = filename or parent.filename
        self._name = name or parent._name

        # Download the SIFTS-standardized .cif.gz file from PDBe
        response = requests.get(
            f"https://www.ebi.ac.uk/pdbe/entry-files/{self._name.lower()}_updated.cif.gz"
        )
        assert response.status_code != 404, f"PDB not found (status_code {response.status_code})"
        self._cif_content = response.content
        
    @cached_property
    def data(self):
        with self.ciff() as f:
            data = MMCIF2Dict().parse(f.name)[self._name]
        return data
    
    @contextmanager
    def ciff(self):
        """
        To be used as: 'with <obj>.ciff() as f:' to work with the contents of the cif as a .cif.gz file
        """
        try:
            file = tempfile.NamedTemporaryFile("wb+", suffix=".cif.gz")
            file.write(self._cif_content)
            yield file
        finally:
            if file:
                file.close()

    @property
    def text(self):
        """
        Return the contents of the cif as plain text
        """
        return gzip.decompress(self._cif_content).decode()



class Pdb(Cif):
    """
    Inherits from Cif, to manage structures of new PDBs not in allodb. Uses PdbCif and has extra methods of the original allodb.PDB class to retrieve information from the new PDB.
    """
    
    path = "."
    
    def __init__(self, pdb, filename=None, name=None):
        self.entry_id = pdb.lower()
        self._name = name or self.entry_id.upper()
        self.filename = filename or f"{self.path}/{self.entry_id}.cif"

    @cached_property
    def cif(self):
        return PDBCif(self)

    
    @cached_property
    def _entities(self):
        return (
            pd.DF(self.cif.data["_entity_poly"], dtype=str)
        )

    @cached_property
    def _protein_entities(self):
        return (
            self._entities
            .query("type == 'polypeptide(L)'")
            .entity_id.to_list()
        )    


    @cached_property
    def _assembly(self):
        try:
            ass =  Assembly(self)

            @contextmanager
            def ciff(self):
                try:
                    file = tempfile.NamedTemporaryFile("w+", suffix=".cif")
                    file.write(self.text)
                    yield file
                finally:
                    if file:
                        file.close()

            ass.cif.ciff = ciff.__get__(ass.cif)
            return ass
        except Assembly._AssemblyIsModel:
            return None

    @cached_property
    def _assembly_asyms(self):
        if self.assembly is not None:
            return pd.DF(self.cif.data["_pdbx_struct_assembly_gen"], dtype=str).sort_values("assembly_id")

    @property
    def assembly(self):
        if self._assembly is not None:
            return self._assembly