"""
Module with accessory classes and functions to manage Sites

Defines the base Site class (which inherits from utils.Structure as well) and child classes to manage sites of assemblies, nonredundancy... and the function to get the residues of sites `get_site_res`.
"""

import pymol2

from . import utils


class BaseSite(utils.Structure):
    """
    Base class to work with any type of site: Site, assembly site, nonredundant site... and which is a child from the Structure class. As such, it must define one of ._atoms or ._residues: afaik, usually ._residues (and thus also needs to define a parent .pdb to access it thorugh ._pdb to obtain the atoms); and it also usually needs a .modulator that defines the site.
    """
    
    @property
    def _pdb(self):
        """
        Parent PDB
        """
        return self.pdb
        
    @property
    def _name(self):
        """
        Attribute that returns the class (or child class) name for tasks such as saving/writing a Cif file
        """
        return self.__class__.__name__

    @utils.property_cache
    def residues(self):
        """
        Overrides utils.Structure.residues. Table with all residues of the site, in the same format as .residues from the parent structure

        TODO: why doesn't it override ._residues instead????
        """
        if utils.debug: print("getting basesite residues")
        return self.pdb.residues.merge(utils.pd.DF(self.site, dtype=str))

    @utils.property_cache
    def protein_residues(self):
        """
        Table with all residues of the site that are part of a protein chain,  in the same format as .residues
        """
        if utils.debug: print("getting basesite protein_residues")
        return (
            self.residues
            .query(f"label_entity_id in {self.pdb._protein_entities}") # residues only from protein entities
        )


class BaseAllosite(BaseSite):
    @utils.property_cache
    def modulator_residues(self):
        """
        Table with the residues that correspond to the annotated modulator(s), in the same format as .residues from the parent structure
        """
        if utils.debug: print("getting baseallosite modulator_residues")
        return self.pdb.residues.merge(utils.pd.DF(self.modulator, dtype=str))

    @utils.property_cache
    def protein_residues(self):
        """
        Table with all residues of the site that are part of a protein chain and different from the modulator,  in the same format as .residues
        """
        if utils.debug: print("getting baseallosite protein_residues")
        site_protein_entities = [
            ent_id 
            for ent_id in self.pdb._protein_entities 
            if ent_id not in self.modulator_residues.label_entity_id.unique().tolist()
        ]
        return (
            self.residues
            .query(f"label_entity_id in {site_protein_entities}") # residues only from protein entities of site_protein_entities
            .merge(self.modulator_residues, how="outer", indicator=True) # not the modulator_residues
            .query(f"_merge == 'left_only'").drop("_merge", axis=1)
        )

    @utils.property_cache
    def minimal_site_pdb(self):
        """
        Minimal_pdb-type object initialized from the present site
        """
        if utils.debug: print("getting baseallosite minimal_site_pdb")
        return utils.Minimal_pdb._from_Site(self)



class AssemblySite(BaseAllosite):
    """
    BaseAllosite-type class to obtain and manage the site formed by an annotated modulator(s) in the structure of its PDB assembly
    """
    def __init__(self, assembly, modulator):
        # Process the passed identifiers of the modulator in the "base" annotation/structure to take into account the repetitions and thus the new label_asym_id remapped identifiers, if it's the case
        if assembly._repetitions is None or tuple(modulator.keys()) == ("label_entity_id",):
            self.modulator = modulator
        else:
            self.modulator = {k: [] for k in modulator.keys()}
            for i, res in utils.pd.DF(modulator, dtype=str).iterrows():
                for rep in range(1, assembly._get_n_repetitions(res["label_asym_id"]) + 1):
                    for k in res.index:
                        # only transform label_asym_id with the repetitions because the rest of fields of modulator will be the same, given that they come from utils.simplify_residues
                        self.modulator[k].append(f"{res[k]}{rep}" if k == "label_asym_id" else res[k]) 
        
        self.pdb = assembly

        # Retrieve the (remapped) modulator residues from the assembly structure
        modulator_residues = assembly.residues.merge(utils.pd.DF(self.modulator, dtype=str))
        assert len(modulator_residues) > 0

        # Establish the site with the present object information
        self.site = get_site_res(self)

    @utils.property_cache
    def nonredundant_site_pdb(self):
        """
        Nonredundant_Site_pdb-type object initialized from the present site
        """
        if utils.debug: print("getting assemblysite nonredundant_site_pdb")
        if len(self.protein_residues) > 0:
            return Nonredundant_Site_pdb(self)

    @utils.property_cache
    def nonredundant_site(self):
        """
        Nonredundant site from the nonredundant_site_pdb of the present site
        """
        if utils.debug: print("getting assemblysite nonredundant_site")
        if self.nonredundant_site_pdb is not None:
            return self.nonredundant_site_pdb.nonredundant_site



class Nonredundant_Site_pdb(utils.Structure):
    """
    Structure-type class to obtain a structure with the passed site modulator(s) and the minimum non-redundant amount of protein chains that directly interact with it. Redundancy in a site is defined as different protein chains (different label_asym_id) of the same entity (label_entity_id) interacting with the modulator(s) through the same site in their surface (the same residue names and sequence IDs (% of intersection above the utils.threshold limit)).
    """
    def __hash__(self):
        """
        Define how each Nonredundant_Site_pdb object's uniqueness has to be calculated for saving info in cache
        """
        return hash((self._pdb, self._site))
    
    def __init__(self, site):
        self._name = "nonredundant_site_pdb"
        self._pdb = site.pdb
        self._site = site

        # List to store the label_asym_ids that will be in the final structure        
        self._nonredundant_label_asym_ids = []

        # Establish function to check if a label_asym_id is present in the default assembly of the "parent" pdb of the site; default is simply return True
        is_in_ass = lambda _: True
        if hasattr(site.pdb, "assembly"):
            ass = site.pdb.assembly
            if ass is not None:
                # If the "parent" pdb has the assembly attribute and it's not None (the assembly is different from the model), define the function to check if the label_asym_id is present
                def is_in_ass(asym_id):
                    ## TODO: if the asym_id has repetitions in the assembly it means that it's present in it and the function doesn't have to do the whole process?
                    # Retrieve repetition # (if any) of the label_asym_id in the assembly and establish the asym_ids to query the assembly with
                    if ass._repetitions is not None:
                        ids = (f"{asym_id}{rep}" for rep in range(1, ass._get_n_repetitions(asym_id)+1))
                    else:
                        ids = [asym_id]
                    # Return whether any of the asym_ids is in the assembly
                    return any(i in ass.residues["label_asym_id"].unique() for i in ids)            

        # For each different entity (top-level hierarchy for which the aim is to reduce redundancy) in the residues of the site
        for entity_id, entity_residues in site.protein_residues.groupby("label_entity_id"):
            if utils.debug: print("site protein residues redundancy, entity", entity_id)
            label_asym_ids = entity_residues["label_asym_id"].unique()
            # If there are more than 1 label_asym_id of the entity, check if they are redundant
            # Similarly to grouping sites according to the % similarity of their residues during creation, group chains according to the % similarity of their site-participating residues
            if len(label_asym_ids) > 1:
                # Prepare to process thorugh all of the asyms of the entity
                asyms_groups = []
                asyms = {asym_id: residues for asym_id, residues in entity_residues.groupby("label_asym_id")}
                for asym_id, residues in asyms.items():
                    # Define properties/data of the present asym to store its information and its comparison to other asyms for later grouping
                    in_ass = is_in_ass(asym_id)
                    asym_dict = lambda common: {"in_ass": in_ass, "not_in_other": True,
                                                "res_of_self_in_other": common}
                    # If there still aren't any asyms groups, save asym as first group
                    if len(asyms_groups) == 0: 
                        asyms_groups.append({asym_id: asym_dict(common={})})
                        if utils.debug: print("asyms_groups", asyms_groups)
                        continue
                    # Else, calculate the similarities to other groups' members (utils.calculate_common_residues) and use them to group the asym (utils.add_to_group)
                    else:
                        self_in_other, other_in_self = {}, {}
                        for i, group in enumerate(asyms_groups):
                            for other_asym_id, other_asym in tuple(group.items()):
                            # Calculate the % of residues in common (from different chains: different _asym_id)
                                other_residues = asyms[other_asym_id]
                                res_of_self_in_other, res_of_other_in_self = utils.calculate_common_residues(residues, other_residues)
                                self_in_other.update({other_asym_id: res_of_self_in_other})
                                other_in_self.update({other_asym_id: res_of_other_in_self})
                                asyms_groups[i][other_asym_id]["res_of_self_in_other"].update({asym_id: other_in_self[other_asym_id]})
                                if utils.debug: print("results", asym_id, other_asym_id, asyms_groups[i][other_asym_id])
                    asyms_groups = utils.add_to_group(asym_id, asym_dict(self_in_other), asyms_groups, self_in_other, other_in_self)
                    if utils.debug: print("asyms_groups", asyms_groups)

                # Pick one asym from each group to add to the list of nonredundant chains (basically, the first of each group)
                for group in asyms_groups:
                    to_keep = sorted({asym: asym_d for asym, asym_d in group.items() if asym_d["not_in_other"]} or group)[0]
                    self._nonredundant_label_asym_ids.append(to_keep)
            # If there is only 1 label_asym_id of the entity, there is no redundancy so add it to the list of nonredundant chains
            else:
                self._nonredundant_label_asym_ids.append(label_asym_ids.item())
            if utils.debug: print("nonredundant_label_asym_ids", self._nonredundant_label_asym_ids)

    @utils.property_cache
    def residues(self):
        """
        Overrides utils.Structure.residues. Table with all residues of the nonredundant structure, in the same format as .residues from the parent structure

        TODO: why doesn't it override ._residues instead????
        """
        if utils.debug: print("getting nonredundant_site_pdb residues")
        return utils.pd.concat(
            (
                self._site.modulator_residues,
                self._pdb.residues.query(f"label_asym_id in {self._nonredundant_label_asym_ids}")
            )
        )

    @utils.property_cache
    def _protein_entities(self):
        """
        Points to the parent pdb's ._protein_entities private method (they have the same protein entities). Returns a list of the entity_ids that correspond to proteins of the structure
        """
        if utils.debug: print("getting nonredundant_site_pdb protein_entities")
        return self._pdb._protein_entities

    @utils.property_cache
    def nonredundant_site(self):
        """
        BaseAllosite-type object with the (nonredundant) site that the annotated modulator forms in the present structure
        """
        if utils.debug: print("getting nonredundant_site_pdb nonredundant_site")
        nonredundant_site = BaseAllosite()
        nonredundant_site.__doc__ = "BaseAllosite-type object with the (nonredundant) site that the annotated modulator forms in the present structure"
        
        nonredundant_site.modulator = self._site.modulator
        nonredundant_site.pdb = self
        nonredundant_site.site = get_site_res(nonredundant_site)
        return nonredundant_site




class EmptySite(Exception): pass

def get_site_res(site, threshold=6.1): # site.pdb CAN BE PDB OR ASSEMBLY (must have .cif and .residues)
    """
    Function to, given a site (BaseAllosite (or child class)-type object and its parent structure in attribute .pdb), return a standardized list of residues (with utils.simplify_residues) from the parent structure (from proteins and any other elements) that define the site (according to the passed threshold distance) with the Python interface of open-source PyMOL
    """
    # Define the PyMOL-style selection of the modulator residues 
    sele = " or ".join(
        f"{res['label_asym_id']}/{res['auth_asym_id']}/{res['auth_comp_id']}`{res['auth_seq_id']}{res['pdbx_PDB_ins_code'].replace('?', '')}/*"
        for i, res in site.modulator_residues.iterrows()
    )
    
    with pymol2.PyMOL() as pymol:
        pymol.cmd.feedback(
            "disable", "executive", "details"
        )  # to silence "ExecutiveLoad-Detail: Detected mmCIF"

        # Load the parent structure of the site to PyMOL (it can only read a "real" file and not from string)
        if hasattr(site.pdb.cif, "filename"):
            pymol.cmd.load(getattr(site.pdb.cif, "filename")) ## Is "getattr" necessary if it's already checked that the attr exists?
        else:
            with utils.tempfile.NamedTemporaryFile("w+", suffix=".cif") as f:
                f.write(site.pdb.cif.text)
                pymol.cmd.load(f.name)

        # Retrieve all atoms within the threshold of the modulator selection
        site_atoms = pymol.cmd.get_model(f"br. all within {threshold} of {sele}")

    # Process the atom selection to obtain residue identifiers
    site_list = set(
        tuple(
            (
                a.segi, a.chain, a.resn,
                a.resi_number, a.ins_code or '?' # pdbx_PDB_ins_code or "?" if none
            ) 
            for a in site_atoms.atom
        )
    )

    if len(site_list) == 0:
        raise EmptySite("Site selection doesn't have any residues")

    # Transform the PyMOL-derived residue identifiers into a standard table of residues that can be used to retrieve the rows/residues from the parent structure's .residues table
    site_res = site.pdb.residues.merge(
        utils.pd.DF(
            site_list,
            columns=[
                "label_asym_id", "auth_asym_id", "auth_comp_id",
                "auth_seq_id", "pdbx_PDB_ins_code"
            ],
            dtype=str
        )
    )

    # Return the minimal, simplified residues annotation
    return utils.simplify_residues(site_res, site.pdb.residues, starting_field_level=0)

