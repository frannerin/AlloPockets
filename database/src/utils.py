"""
Module with accessory classes and functions to manage protein structures in general

It defines the foundational class Structure and other foundational utils such as the function `simplify_residues` to standardize the annotation/saving of residues from structure and enhance interoperability. It also has the base classes for dealing with assemblies, cif files...
"""

import os, hashlib, io, requests, json, gzip, tempfile

import pandas as pd
pd.DF = pd.DataFrame

# # To deal with CIFs
from .cifutils import MMCIF2Dict, CifFileWriter

# To deal with assemblies
import numpy as np
import biotite
import biotite.structure.io.pdbx as pdbx
if int(biotite.__version__.split(".")[0]) > 0:
    pdbx.PDBxFile = pdbx.CIFFile
    

from functools import lru_cache, cached_property


from . import allodb
from .viz import Viz


property_cache = lambda func: property(lru_cache(maxsize=64)(func)) ## maxsize=128
debug = False
threshold = 0.78 # Threshold to consider overlap/similarity/same



def simplify_residues(
    residues_subset, 
    residues, 
    fields = (["label_asym_id"], ["label_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]), 
        # Possible combinations of columns from the DataFrames to use so that results are standardized (all use the same columns) and the minimum number of columns is used
    starting_field_level=-1
):
    """
    Function to get a "standardized" (a.k.a., interoperable, smallest/using the minimum amount of columns necessary, comparable/to standardize among multiple methods, etc) dictionary with which to obtain the rows of certain residues (residues_subset) from a bigger DataFrame of residues (residues) using the pd.DF.merge method, using only the fields or combinations of fields from the passed list.
    """
    # # of residues to be obtained
    n_residues = len(residues_subset); assert n_residues > 0

    field_level = starting_field_level
    queried_residues = ()
    # Iterate through the fields list starting from the first to try to get the desired residue rows from the big dataframe using the fields combination; if the first column/combination isn't enough to obtain the desired rows, the second combination is used/should be enough
    while len(queried_residues) != n_residues:
        field_level += 1
        # Get the tested fields from the DataFrame of desired residues
        desired_residues = (
            residues_subset[fields[field_level]]
            .drop_duplicates()
        )
        # Use the obtained fields information to query the bigger DataFrame; if the result is correct/number of rows match loop is exited
        queried_residues = residues.merge(desired_residues)
    # Sanity check
    assert len(queried_residues) == n_residues

    # If the second combination of fields is used, ins_code might be all "?" and can be dropped    
    if "pdbx_PDB_ins_code" in desired_residues and desired_residues["pdbx_PDB_ins_code"].unique().tolist() == ["?"]:
        desired_residues.drop("pdbx_PDB_ins_code", axis=1, inplace=True)
    assert desired_residues.notna().all().all()
    # Return the smallest possible dict with which the bigger DataFrame can be used with the .merge method to obtain the subset of rows
    return desired_residues.to_dict(orient="list")



def calculate_common_residues(self_residues, other_residues):
    """
    Function to calculate the % of residues/rows on one DataFrame that are found in another, and vice versa, taking into account only chain ID and residue ID fields.

    TODO: an inner merge would avoid having to do the query for _merge == 'both'?
    """
    # Use pandas.DataFrame.merge method to obtain the union of the two with an indicator column
    union = (
        self_residues.merge(
            other_residues, # merge using on= or selecting only these columns?
            on = ["label_entity_id", "auth_comp_id", "auth_seq_id", "pdbx_PDB_ins_code"],
            how = "outer", indicator = True)
        .query("_merge == 'both'")
    )
    # Return a tuple of the % of residues of self in other and vice versa 
    return len(union) / len(self_residues), \
            len(union) / len(other_residues)


def add_to_group(self_id, self_dict, groups, self_in_other, other_in_self):
    """
    Function to group an element (i.e., site (DF of residues) or chain (DF of residues with the same label_asym_id)) into an existing dictionary of groups, according to the similarities (residues in common) stored in self_in_other and other_in_self, and depending on threshold module variable.

    The group(s) that have at least a member with which the element has a similarity above the threshold are identified to introduce the element, and if the element has similarities above the threshold with members from more than one group, those groups will be merged together and include the element.

    self_id : str or int
        Identifier of the element being processed/grouped, used in the keys of 'groups'
    self_dict : dict
        Dictionary with information/to store information about the element being grouped (e.g., similarity to other elements in the same or other groups)
    groups : list of dicts
        List with each group's dict with their group members as items (with element_id as keys)
    self_in_other : dict
        Dictionary with the % of residues of the element being processed/grouped found in each of the other elements that are already in groups
    other_in_self : dict
        Dictionary with the % of residues of each of the other elements found in the element being processed/grouped
    """
    # If any of the similarities with any other element in any of the dictionaries (i.e., residues of self in other or vice versa) is above the threshold
    if any(
        common > threshold 
        for d in [self_in_other, other_in_self] 
            for common in d.values()
    ):
        # Identify and sort the existing groups (by each group's index in the list) in 'groups', if they have a group member with a similarity with the processed element above the threshold
        commonest_groups = sorted(
            set(
                group_id
                for d in [self_in_other, other_in_self]
                    for id, common in d.items()
                    if common > threshold
                        for group_id, group in enumerate(groups) 
                        if id in group
            ),
            reverse=True # so that when popping if len(commonest_groups) > 1 it starts from the highest id
        )
        # sc assert len(commonest_groups) == len(set(commonest_groups))

        # If there is only one identified group, save the group id to add the element to it
        if len(commonest_groups) == 1: 
            self_group_id = commonest_groups[0]
        # Else, if there are more than 1 identified groups
        elif len(commonest_groups) > 1:
            # Pop the groups to be merged from 'groups' and create a new group dict with their members combined
            # Commonest_groups (ids) were sorted in reverse to start popping from the highest group index id seamlessly
            merged_groups = {
                k: v 
                for d in (
                    groups.pop(group_id) for group_id in commonest_groups
                ) 
                for k,v in d.items()
            }
            # Start the new merged group and save the new merged group id to add the new element to it
            groups.append(merged_groups)
            self_group_id = groups.index(merged_groups)

        # Add the new element to the group with the saved group id
        groups[self_group_id].update({self_id: self_dict})
        # For each of the members of the group to which the element has been saved, assign "not_in_other": False if after the addition of the new member/regrouping they are indeed contained in other group members
        for other_id, other_dict in groups[self_group_id].items():
            if any(
                # at least one other member contains above-threshold residues of the processed member, but is below-threshold-contained in the residues of the processed member
                (
                    other_dict["res_of_self_in_other"][compare_id] > threshold
                    and compare_dict["res_of_self_in_other"][other_id] < threshold
                )
                # comparing each member to all others
                for compare_id, compare_dict in groups[self_group_id].items() 
                if compare_id != other_id
            ):
                other_dict.update({"not_in_other": False})
    # ELSE (if there are no other "similar" elements in any of the groups), start a new group with the element
    else:
        groups.append({self_id: self_dict})

    # Return the updated list of dicts/groups with the new element in it
    return groups



class Cif:
    """
    Base class to manage .cif files of elements (of whole PDBs, chains, sites, assemblies...)

    name : str
        Name that will appear on the first line of the file
    data : dict
        Cif data in dictionary format
    """
    def __init__(self, name, data):
        self._name = name
        self.data = data

    def __hash__(self):
        """
        Define how each Cif object's uniqueness has to be calculated for saving info in cache
        """
        return sum(map(hash, ((k, tuple(v)) for k, v in self.data["_atom_site"].items())))
        
    def _get_or_save_text(self, filename=None, compress=False):
        """
        Private method to obtain the information of the Cif as the text of the .cif (if filename=None; needs to be 'saved' to a fake file to read the contents as string) or save it into a file
        """
        if debug: print("Cif get_or_save")
        # Establish how to get the file stream to write into
        if filename is not None:
            # If a file needs to be saved and a filename has been passed, check consistency and assess compression
            assert ( filename.endswith(".cif") or filename.endswith(".cif.gz") )
            if filename.endswith(".cif.gz"):
                compress = True
            file = lambda: open(filename, "w")
        elif filename is None:
            file = lambda: tempfile.NamedTemporaryFile("w+", suffix=".cif")

        # Open the established file stream and write the contents of the Cif as a .cif file
        with file() as f:
            writer = CifFileWriter(f.name, compress=compress)
            writer.write({self._name: self.data})
            # If filename is None/no saving is requested, return the contents of the .cif file as text string
            if filename is None:
                return f.file.read()

    @property_cache
    def text(self):
        """
        Text string of the context of the .cif file that can be obtained from the Cif
        """
        if debug: print("getting Cif text")
        try:
            # If self._text is defined by a child Class (e.g., PDBCif), return it
            return self._text
        except:
            # Else, use the private method to get the .cif file text string
            return self._get_or_save_text()

    
    def save(self, filename):
        """
        Function to save the contents of the Cif in a .cif file
        """
        return self._get_or_save_text(filename=filename)



class PDBCif(Cif):
    """
    Child class of Cif that defines its own ._text private attribute pointing to the disk-saved original .cif file from the PDB (SIFTS updated .cif.gz file from PDBe) (or is downlaoded if it doesn't exist

    pdb : allodb.PDB object
        PDB object for which to get the PDBe SIFTS-updated .cif.gz file (saved in allodb.datapath or retrieved online otherwise)
    update : bool
        Whether to update the saved .cif.gz file if it has changed w.r.t. the saved .cif.gz file (according to the hash of the saved DB entry (PDB._cif_hash) vs. the hash of the online file) or not
    save : bool
        Whether to save the downloaded .cif.gz file if it cannot be retrieved from allodb.datapath (saved in allodb.datapath); is True if update is True
    """
    def __init__(self, pdb, update, save):
        self._name = pdb.entry_id.upper()
        self._text, self.hash, filename = self._get_cif(
            pdb.entry_id, update, save, db_hash=pdb._cif_hash
        ).values()
        if filename is not None:
            self.filename = filename

    def __hash__(self):
        """
        Define how each PDBCif object's uniqueness has to be calculated for saving info in cache
        """
        return hash(self._text)
            
    @property_cache
    def data(self):
        """
        Cif data in dictionary format. The .data attribute is overriden w.r.t. the parent class and the data dictionary is obtained directly from parsing the saved file, or the .cif-like string saved in the .text attribute
        """
        if debug: print("getting pdbcif data")
        # If the file exist, parse it; else write the contents of .text to a fake file for applying the same parser
        if hasattr(self, "filename"):
            return MMCIF2Dict().parse(self.filename)[self._name]
        else:
            with tempfile.NamedTemporaryFile("w+", suffix=".cif") as f:
                f.write(self.text)
                return MMCIF2Dict().parse(f.name)[self._name]
    
    @staticmethod
    def _download_SIFTS_cif(entry_id):
        """
        Function to download updated SIFTS .cif.gz from PDBe
        """
        print(f"Downloading {entry_id}")
        response = requests.get(
            f"https://www.ebi.ac.uk/pdbe/entry-files/{entry_id.lower()}_updated.cif.gz"
        )
        assert response.status_code != 404, f"PDB not found (status_code {response.status_code})"
        return response.content
        
    @staticmethod
    def _hash(text):
        """
        Helper function to hash magic method

        TODO: don't remember why it had to be done like this
        """
        return hashlib.sha256(bytes(text, "utf8")).hexdigest()

    @classmethod
    def _get_cif(cls, entry_id, update=False, save=False, db_hash=None):
        """
        Helper function to class object initialization, to manage retrieval/reading of .cif.gz file and matching to the database-saved hash, given a PDB ID
        """
        if debug: print("getting pdbcif get_cif")
        save = True if update else save # save regardless of anything if update is True

        # Define filenames and a method to check existence
        cifname = f"{entry_id.lower()}_updated.cif.gz"
        file = os.path.join(allodb.datapath, cifname) # path + SIFTS filename
        exists = lambda: os.path.isfile(file)

        # If file exists, read contents and hash
        if exists():
            with gzip.open(file, "rb") as f:
                cif_text = f.read().decode()
            cif_hash = cls._hash(cif_text)

        # If it doesn't exist or asked to update, download and proceed
        if not exists() or update:
            web_cif_content = cls._download_SIFTS_cif(entry_id)
            web_cif_text = gzip.decompress(web_cif_content).decode()
            web_cif_hash = cls._hash(web_cif_text)

            # If asked to update and file exists, compare web and local hash and print comparison result (if update is not changed to False, the update will happen next)
            if update and exists():
                if cif_hash != web_cif_hash: print("Local file hash doesn't coincide with online file hash")
                elif cif_hash == web_cif_hash:
                    update = False; print("Local and online file hashes coincide")

            # If it doesn't exist or asked to update, save web file and proceed with info/contents of web file
            if not exists() or update:
                cif_text, cif_hash = web_cif_text, web_cif_hash
                if save:
                    with open(file, "wb") as f:
                        f.write(web_cif_content)

        # Print warning if database-saved hash and the hash of the file being used (file saved in the datapath or downloaded online) do not coincide
        if db_hash is not None and db_hash != cif_hash: print(
            "Database-stored cif file hash doesn't coincide with retrieved cif file hash: entry was created with a different (e.g., outdated) cif file version."
            )

        # Return final information of cif
        return {"text": cif_text, "hash": cif_hash, "filename": file if exists() else None}

    


class Structure:
    """
    Base class to work with any type of structures: PDBs, assemblies, minimal pdbs, sites (residues + ligands). Child classes must define one of "_atoms" or "_residues" that defines the information that is in the structure and/or how to get it, while the non-defined one can be obtained from the other.
    """
    @property_cache
    def atoms(self):
        """
        Table with all atoms of structure matching the information of the "_atom_site" block of a cif. If _atoms is not defined, they are retrieved using .residues (._residues) and the information from a complete structure (e.g., PDB or assembly...) contained in the object's ._pdb private attribute ("parent" pdb/structure).
        """
        if debug: print("getting structure atoms")
        try:
            return self._atoms
        except:
            return self.residues.merge(self._pdb.atoms)

    @property_cache
    def residues(self):
        """
        Table with all residues of the structure, using the information of the "_atom_site" block of a cif but only fields with residue ID information that allow to identify individual, unique residues. If _residues is not defined, they are retrieved using .atoms (._atoms) by dropping all atom-specific fields and retaining only fields that characterize the different residues (and dropping duplicate rows).
        """
        if debug: print("getting structure residues")
        try:
            return self._residues
        except:
            # return (
            #     self.atoms.drop(
            #         [
            #             'group_PDB', 'id', 'type_symbol',
            #             'auth_atom_id', 'label_atom_id', 'label_alt_id',
            #             'Cartn_x', 'Cartn_y', 'Cartn_z',
            #             'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge'
            #         ],
            #         axis=1
            #     )
            #     .drop_duplicates()
            # )
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

    @property_cache
    def cif(self):
        """
        Cif-type object with the information of the structure. Will only contain the "_atom_site" block with the information from the atoms attribute.
        """
        if debug: print("getting structure cif")
        return Cif(
            self._name,
            {
                "_entry": {"id": [self._name]},
                "_atom_site": self.atoms.to_dict(orient="list")
            },
        )

    def view(self):
        """
        Visualize the structure of the element.
        """
        return Viz(self)




class Minimal_pdb(Structure):
    """
    Structure-type class to make objects that will only contain the minimum amount necessary of "molecules" (i.e., label_entity_id) from the object from which it is intialized in order to comprise a complete protein chain(s) + the ligand(s) it(they) is(are) directly interacting with.

    TODO: should this be deprecated/deleted?
    """
    def __hash__(self):
        """
        Define how each Minimal_pdb object's uniqueness has to be calculated for saving info in cache
        """
        return hash((self._pdb, sum(map(hash, ((k, tuple(v)) for k, v in self._residues.to_dict(orient="list").items())))))
    
    @classmethod
    def _from_Site(cls, site):
        """
        If it is initialized from a site, the object will only comprehend the ligand(s) of the site and the protein chain(s) it is directly interacting with, discarding the rest of protein chains or other molecules.
        """
        self = cls()
        self._name = "minimal_site_pdb"
        self._pdb = site.pdb

        # _residues is defined with only the modulator(s) residues and the residues of the protein chains that are present in the site
        self._residues = pd.concat((
            site.modulator_residues,
            site.pdb.residues.query(f"label_asym_id in {site.protein_residues['label_asym_id'].unique().tolist()}")
        ))
        
        return self
        
    @classmethod
    def _from_PDB(cls, pdb):
        """
        If it is initialized from a PDB, the object will only comprehend the ligand(s) of all sites of the PDB and the protein chain(s) that interact with them, discarding the rest of protein chains that do not interact with any of the annotated modulators or other molecules.
        """
        self = cls()
        self._name = "minimal_pdb"
        self._pdb = pdb

        # _residues is defined as the merge between all the Minimal_pdb objects' residues attributes originated from each Site of the PDB
        self._residues = pd.concat(
            (site.minimal_site_pdb.residues for site in pdb.sites)
        ).drop_duplicates()
        
        return self

    @classmethod
    def _from_Assembly(cls, assembly):
        """
        Same as ._from_PDB but using the structure of the assembly.
        """
        self = cls()
        self._name = f"{assembly._name}_minimal"
        self._pdb = assembly

        self._residues = pd.concat(
            (site.assembly_site.minimal_site_pdb.residues for site in assembly._pdb.sites if site.assembly_site is not None)
        ).drop_duplicates()
        
        return self





class Assembly(Structure):
    """
    Structure-type object that reconstructs and stores the assembly from the information in the PDB Cif.
    
    The main method _get_assembly provides an ._atoms DataFrame where reptitions of the asymmetric unit have remapped, unique auth_asym_id and label_asym_id fields (chain identifiers), to avoid visualization and other processing problems with common suites for protein/biomolecular structural data and take into account all "repeated" atoms in their transformed coordinates. _get_assembly also generates the ._repetitions attribute that stores a DataFrame with the number of times each chain/molecule (unique label_asym_id) is repeated.

    pdb : PDB
    assembly_id : int
        Assembly ID to reconstruct, as PDBs can have many assemblies and the annotated allosteric modulators might be in any of them
    """
    def __init__(self, pdb, assembly_id=1, **kwargs):
        self._pdb = pdb
        self._atoms, self._repetitions = self._get_assembly(pdb, assembly_id, **kwargs)
        self._name = f"{pdb.entry_id}_assembly{assembly_id}"

    def __hash__(self):
        """
        Define how each Assembly object's uniqueness has to be calculated for saving info in cache
        """
        return hash(self._name)

    def __call__(self, assembly_id):
        """
        Makes all Assembly objects callable to obtain the corresponding Assembly of the parent PDB for an arbitrary assembly_id
        """
        return Assembly(self._pdb, assembly_id)

    @property_cache
    def _entities(self):
        """
        Populated with the parent PDB's ._entities attribute
        """
        if debug: print("getting assembly _entities")
        return self._pdb._entities
    
    @property_cache
    def _protein_entities(self):
        """
        Populated with the parent PDB's ._protein_entities attribute
        """
        if debug: print("getting assembly protein_entities")
        return self._pdb._protein_entities

    @property_cache
    def cif(self):
        """
        Cif-type object with the information of the assembly structure. It inherits the same initialization of .cif from the parent Structure and adds the ._repetitions attribute.
        """
        if debug: print("getting assembly cif")
        cif = super().cif
        if self._repetitions is not None:
            cif.data.update({"_assembly_repetitions": self._repetitions.to_dict(orient="list")})
        return cif

    @cached_property # cannot be @utils.property_cache because PDB is cached w/out taking into account changes in its .sites and this depends on them
    def minimal_pdb(self):
        """
        Minimal_pdb-type object with the information of the assembly structure.
        """
        if debug: print("getting assembly minimal_pdb")
        return Minimal_pdb._from_Assembly(self)
    

    
    class _AssemblyIsModel(Exception):
        """
        TODO: probably could be deleted?
        """
        _message = "The model/asymmetric unit is the same as the biological assembly 1."

    def _get_n_repetitions(self, id, name="label_asym_id"):
        """
        Helper private method to obtain the specific number of repetitions of a chain ID (i.e., label_asym_id) on the assembly (from ._repetitions)
        """
        if debug: print("getting assembly n_repetitions")
        return self._repetitions.loc[lambda x: x[name] == id]["repetition_number"].item() # will fail if _repetitions is None
        
    @classmethod
    def _get_assembly(cls, pdb, assembly_id, **kwargs):
        """
        Main function to get the assembly from the information in the cif and return a DataFrame of all atoms, and repetitions.
        """
        if debug: print("getting assembly get_assembly")
        # Read cif with biotite and get the assembly (default, first/1) atoms StackArray
        with io.StringIO(pdb.cif.text) as f:
            pdbx_file = pdbx.PDBxFile.read(f)
        ass = pdbx.get_assembly(pdbx_file, f"{assembly_id}", **kwargs)
    
        # Get atom_site field from the model's cif and retain only 1 of the 'alt' atom coordinates (if there are two: A, B) (by not taking into account for grouping) for biotite compatibility
        atom_fields = [
            "label_asym_id",
            "auth_asym_id",
            "auth_comp_id",
            "auth_seq_id",
            "pdbx_PDB_ins_code",
            "auth_atom_id",
        ]
        atom_site_df = pdb.atoms.drop_duplicates(atom_fields)

        if ass.shape[1] == len(atom_site_df):
            raise cls._AssemblyIsModel()
            
        # From the model: make a list of residues (resid, auth_asym, label_asym) and initialize a counter
        residues = (
            pdb.residues[
                ["label_asym_id", "auth_asym_id", "auth_comp_id", "auth_seq_id", "pdbx_PDB_ins_code"] # _seq_id is str
            ].replace(["?", ".", None], "") # for compatibility with biotite's empty "ins_code"
            .values # list of residues in the model
        ) 
        residue_counts = {tuple(res): 0 for res in residues}
        
        # Using the model info, create a vector with the label_asym of each atom of the assembly stackarray (biotite loses label_asym)
        label_asym_id_mapping = {
            (auth_asym, resname, resnum, resic): label_asym for label_asym, auth_asym, resname, resnum, resic in residues
        }  # resids-chains to label_asym mapping
        ass_label_asym = np.array(
            tuple(
                label_asym_id_mapping[res]
                for res in zip(ass.chain_id, ass.res_name, ass.res_id.astype(str), ass.ins_code) # res_id as str for compatibility with pdb.residues (all str)
            ),
            dtype=object,
        )  # vector of label_asym_ids for each atom of the assembly stackarray
    
        # Get the ranges of indices of the stackarray that correspond to the different residues to remap their auth_ and label_asym with the repetitons
        resbreaks = (
            (
                np.diff(
                    tuple(
                        zip(
                            np.unique(ass.ins_code, return_inverse=True)[1], # .astype(str)
                            ass.res_id,
                            np.unique(ass.chain_id, return_inverse=True)[1],
                            np.unique(ass_label_asym, return_inverse=True)[1],
                        )
                    ),
                    axis=0,
                )
                != (0, 0, 0, 0)
            )
            .any(axis=1)
            .nonzero()[0]
        )  # atom stackarray indices where the residue changes (breaks between atoms of different residues)
        resranges = tuple(
            zip((-1, *resbreaks), (*resbreaks, len(ass.res_id) - 1))
        )  # indices of the ass atom array for each residue (when it'll be sliced later +1 is summed)
    
        # Store the remapped auth_asym and label_asym to account for chain repetitions in the assembly
        remapped_auth_asym = np.ndarray(
            ass.chain_id.shape, dtype=object
        )  # empty vectors to store them
        remapped_label_asym = np.ndarray(ass.chain_id.shape, dtype=object)
        for atomini, atomend in resranges:
            # Get the info of the current residue
            label_asym = ass_label_asym[atomend]
            auth_asym = ass.chain_id[atomend]
            resname = ass.res_name[atomend]
            resid = ass.res_id[atomend]
            resic = ass.ins_code[atomend]
    
            # Increase the n of repetitions of the residue by 1 in the counter dict and store the repetition n
            residue_counts[
                (label_asym, auth_asym, resname, str(resid), resic) # res_id as str for compatibility with pdb.residues (all str)
            ] += 1  # increase the number of repetitions of the unique residue by 1
            rep = residue_counts[(label_asym, auth_asym, resname, str(resid), resic)] # res_id as str
    
            # And store the chainids of the residue's atoms to identify its repetition number acc. to the nº of times the residue has been repeated
            remapped_auth_asym[atomini + 1 : atomend + 1] = f"{auth_asym}{rep}"
            remapped_label_asym[atomini + 1 : atomend + 1] = f"{label_asym}{rep}"
            
        # Make a DF mimicking part of the cif's atom_site field with all assembly atoms and their new assembly coords
        assdf = pd.DF(
            {
                "label_asym_id": ass_label_asym,
                "auth_asym_id": ass.chain_id,
                "auth_comp_id": ass.res_name,
                "auth_seq_id": ass.res_id, #.astype(str),
                "pdbx_PDB_ins_code": (ic or '?' for ic in ass.ins_code), # '?' for compatibility with the original cif field
                "auth_atom_id": ass.atom_name,
                "Cartn_x": ass.coord[0][:, 0],
                "Cartn_y": ass.coord[0][:, 1],
                "Cartn_z": ass.coord[0][:, 2],
            },
            dtype=str
        )
    
        # Complete the dataframe using pandas' .merge method with the cif's atom_site df
        mergeddf = assdf.merge(
            atom_site_df.drop(
                columns=["Cartn_x", "Cartn_y", "Cartn_z"]
            ),  # drop the columns that will be provided by the remapped df
            on=atom_fields,
            how="left",  # retain only keys from left df
        )[
            atom_site_df.columns
        ]  # reorder columns to match the standard cif atom_site field column order for saving
        # COLUMN "id" (atom_id) IS NOW REPEATED FOR EACH REPETITION OF THE MODEL: SHOULD BE EASY TO RESET IT BUT SO FAR EVERYTHING WORKS 
        # Retrieve nº of repetitions of the chainids using the residue repetition counter
        repetitions = set(
            (auth_asym, label_asym, reps)
            for (label_asym, auth_asym, _, _, _), reps in residue_counts.items()
        )
        
        # If there isn't any repetition, the assembly is a selection of the model and changing the original chainids is unconvenient and not necessary
        if not any(rep[-1] > 1 for rep in repetitions):
            repetitions = None
        else:
            mergeddf.loc[:, ["auth_asym_id", "label_asym_id"]] = tuple(
                zip(remapped_auth_asym, remapped_label_asym)
            )  # substitute auth_asym and label_asym by the remapped
            repetitions = pd.DF(
                repetitions,
                columns=["auth_asym_id", "label_asym_id", "repetition_number"] # repetition_number is int
            )
            
        return mergeddf, repetitions