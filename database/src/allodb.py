"""
Main module of the database

Defines the Classes/Database tables that structure the database, inheriting and using methods from the utils and siteutils (and cifutils) modules.
"""

import peewee

# Using "Model" from signals to be able to use @pre_save to automatize tasks: properly process/parse the passed modulator molecule of the Site
from playhouse.signals import Model, pre_save

# Using sqlite for simplicity (.db file), with the APSW drivers for efficiency and to exploit JSONField to save dictionaries, dataframes, lists...
from playhouse.apsw_ext import APSWDatabase, DateField
from playhouse.sqlite_ext import JSONField


from . import utils, siteutils
from .viz import Viz



# Defer initialization of the database until the script is executed from the command-line
db = APSWDatabase(None)


# .cif.gz files should be saved on a "data" folder inside the same directory as this module, or specified upon execution otherwise
datapath = utils.os.path.join(utils.os.path.dirname(__file__), "data")
# wheter to save the .cif.gz files of PDBs upon requesting them if they aren't already saved
save_cifs = False
if save_cifs: utils.os.makedirs(datapath, exist_ok=True)



class PDB(Model, utils.Structure):
    """
    Class/Objects of PDB have to inherit from peewee's "Model", and they inherit some common methods from utils.Structure.
    
    Each PDB is defined by its entry_id and the hash of the cif file with which it was created.
    """
    entry_id = peewee.CharField(primary_key=True, unique=True)
    _cif_hash = peewee.FixedCharField(max_length=64)

    # Necessary according to docs
    class Meta:
        database = db

    
    def __hash__(self):
        """
        Define how each PDB object's uniqueness has to be calculated for saving info in cache
        """
        return hash((Model.__hash__(self), self.entry_id, self._cif_hash)) #, self.sites

    @utils.property_cache
    def cif(self):
        """
        Overrides utils.Structure.cif. Access the corresponding utils.PDBCif class object with the info
        """
        if utils.debug: print("getting pdb cif")
        try:
            return self._cif
        except:
            return utils.PDBCif(self, update=False, save=save_cifs)

    @utils.property_cache
    def atoms(self):
        """
        Overrides utils.Structure.atoms. Table with all atoms of structure in the "_atom_site" block of the cif
        """
        if utils.debug: print("getting pdb atoms")
        return utils.pd.DF(self.cif.data["_atom_site"], dtype=str)

    @utils.property_cache # cannot be cached_property or it might not print the message # TODO: there is not a message to print anymore
    def _assembly(self):
        """
        Private method to retrieve the structure's assembly and handle the "AssemblyIsModel" error to just return None in that case.

        TODO: probably is not needed and should be deprecated.
        """
        if utils.debug: print("getting pdb assembly")
        try:
            return utils.Assembly(self)
        except utils.Assembly._AssemblyIsModel:
            return None

    @utils.property_cache # cannot be cached_property or it might not print the message # TODO: there is not a message to print anymore
    def _assembly_asyms(self):
        """
        Private method to return a dataframe mapping the label_asym_ids that are in each assembly_id #
        """
        if utils.debug: print("getting pdb assembly_asyms")
        if self.assembly is not None:
            return utils.pd.DF(self.cif.data["_pdbx_struct_assembly_gen"], dtype=str).sort_values("assembly_id")

    @property
    def assembly(self):
        """
        utils.Structure-type object of the structure assembly, if it is different from the model

        A biological assembly can be a selection of the chains/atoms in the model, or a spatially arranged repetition, in which case the chain IDs are remapped to reflect the number of repetitions (e.g., chain A --> chain A1 A2)
        """
        if self._assembly is not None: # is None
            # print(utils.Assembly._AssemblyIsModel._message)
            # else:
            return self._assembly

    @utils.property_cache
    def _entities(self):
        """
        Private method to return a dataframe with the polymer entities of the structure
        """
        if utils.debug: print("getting pdb polymer entities")
        return (
            utils.pd.DF(self.cif.data["_entity_poly"], dtype=str)
        )

    @utils.property_cache
    def _protein_entities(self):
        """
        Private method to return a list of the entity_ids that correspond to proteins of the structure
        """
        if utils.debug: print("getting pdb protein entities")
        return (
            self._entities
            .query("type == 'polypeptide(L)'")
            .entity_id.to_list()
        )

    @utils.cached_property # @utils.property_cache # cannot be property_cache because PDB is cached w/out taking into account changes in its .sites and this depends on them
    def minimal_pdb(self):
        """
        utils.Structure-type object which only keeps the modulators of the sites of the PDB, and the PROTEIN chains that interact with them
        """
        if utils.debug: print("getting pdb minimal_pdb")
        return utils.Minimal_pdb._from_PDB(self)

    def view(self, assembly_id=''):
        """
        Visualize the PDB structure. assembly_id='' -> Model
        """
        return Viz(self, assembly_id=assembly_id)

    

class Orthosite(Model, siteutils.BaseSite):
    """
    Class/Objects of Orthosite have to inherit from peewee's "Model", and they inherit some common methods from siteutils.BaseSite.
    
    Each Allosite is defined by, mainly, the pdb it belongs to and a list of residues of the site.

    In addition, it contains an information dictionary of the data source and the last date in which it was updated.
    """
    pdb = peewee.ForeignKeyField(PDB, index=True, backref="orthosites", on_delete="CASCADE") # backref="orthosites" creates a .orthosites attribute on each PDB
    site = JSONField(null=True)
    info = JSONField(null=True)
    updated = DateField()

    # Necessary according to docs
    class Meta:
        database = db

    def __hash__(self):
        """
        Define how each object's uniqueness has to be calculated for saving info in cache
        """
        return hash((Model.__hash__(self), self.pdb, sum(map(hash, ((k, tuple(v)) for k, v in self.site.items())))))

    def get_site(self, obj, info:["residues", "atoms"]="residues"):
        """
        Retrieve the annotated orthosteric site residues (or info="atoms", or info="protein_residues") from the passed object.
        To be used with: PDB, Assembly, Site, Nonredundant_Site_PDB, Minimal_PDB...
        """
        return (
            getattr(obj, info)
            .merge(utils.pd.DF(self.site, dtype=str))
        )


@pre_save(sender=Orthosite)
def _process_orthosite(model, site, created):
    """
    Try to get the information passed as "site" for Orthosite creation from the residues of the passed "pdb", and standardize the saved "site" info
    """
    if created:
        # Check that the info passed as "site" is present in the pdb
        site_residues = site.pdb.residues.merge(utils.pd.DF(site.site, dtype=str))
        assert len(site_residues) > 0
        # Check that, e.g., all of the residue IDs or all the Uniprot IDs passed as site.site are in site_residues?

        site.site = utils.simplify_residues(
            residues_subset = site_residues, # or site.site?
            residues = site.pdb.residues,
            fields = (utils.pd.DF(site.site, dtype=str).columns.to_list(),) ##############
        )



class Site(Model, siteutils.BaseAllosite):
    """
    Class/Objects of Site (Allosite) have to inherit from peewee's "Model", and they inherit some common methods from siteutils.BaseAllosite.
    
    Each Site is defined by, mainly, the pdb it belongs to, the annotated modulator molecule, and a list of residues of the site retrieved upon creation.

    In addition, it contains an information dictionary of the data source and fixes, the last date in which it was updated, and a dictionary containing
    information about related_sites (other molecules of the annotated modulator(s) present in the structure that can be equivalent (they form the same site 
    in the same protein entity but a different chain), or nonequivalent (situated elsewhere and not annotated as modulator/allosteric site)).

    TODO: fully migrate to Allosite
    """
    pdb = peewee.ForeignKeyField(PDB, index=True, backref="sites", on_delete="CASCADE") # backref="sites" creates a .sites attribute on each PDB
    modulator = JSONField()
    site = JSONField(null=True)
    info = JSONField(null=True)
    related_sites = JSONField(null=True)
    updated = DateField()

    # Necessary according to docs
    class Meta:
        database = db

    def __hash__(self):
        """
        Define how each PDB object's uniqueness has to be calculated for saving info in cache
        """
        return hash((Model.__hash__(self), siteutils.BaseAllosite.__hash__(self)))


    def __hash__(self):
        """
        Define how each BaseAllosite (or child) object's uniqueness has to be calculated for saving info in cache
        """
        return hash((self.pdb, sum(map(hash, ((k, tuple(v)) for k, v in self.modulator.items())))))


    
    @utils.property_cache
    def _assembly_site(self):
        """
        Private method to obtain the siteutils.BaseAllosite(AssemblySite)-type object with the site that the annotated modulator forms in the assembly, if the PDB has an assembly
        """
        if utils.debug: print("getting site assembly_site")
        if self.pdb.assembly is not None: # if this is not none, pdb._assembly_asyms will not be None
            # Get the assembly_id (#) in which the modulator chain(s) is(are) present
            ass_id = int(next(
                ass["assembly_id"] for ass in self.pdb._assembly_asyms.to_dict(orient="records") 
                if any(m in ass["asym_id_list"].split(',') for m in self.modulator_residues["label_asym_id"].unique().tolist())
            ))
            try:
                if ass_id == 1:
                    return siteutils.AssemblySite(self.pdb.assembly, self.modulator)
                else:
                    # In the rare case that the annotated allosteric modulator is not in the Biological assembly 1, print a message
                    print(f"{self.pdb.entry_id}, site {self.id}: Modulator is in biological assembly {ass_id}")
                    return siteutils.AssemblySite(self.pdb.assembly(ass_id), self.modulator)                    
            except AssertionError:
                return False

    @property
    def assembly_site(self):
        """
        siteutils.BaseAllosite(AssemblySite)-type object with the site that the annotated modulator forms in the assembly, if the PDB has an assembly

        TODO: it returns False; should it return None for consistency?
        """
        if self._assembly_site is False:
            print("Modulator is not in assembly.")
            return False
        else:
            return self._assembly_site
            

    @utils.property_cache
    def nonredundant_site_pdb(self):
        """
        utils.Structure-type object with the modulator(s) of the site and the PROTEIN chain(s) that interact which, if the modulator is in
        a (symmetrical) interface of chains, only selects one of the chains of each group to reduce redundancy
        """
        if utils.debug: print("getting site nonredundant_site_pdb")
        if len(self.protein_residues) > 0:
            return siteutils.Nonredundant_Site_pdb(self)

    @utils.property_cache
    def nonredundant_site(self):
        """
        siteutils.BaseAllosite-type object with the site that the annotated modulator forms in the nonredundant_site_pdb
        """
        if utils.debug: print("getting site nonredundant_site")
        if self.nonredundant_site_pdb is not None:
            return self.nonredundant_site_pdb.nonredundant_site
    



@pre_save(sender=Site)
def _process_allosite(model, site, created):
    """
    Try to get the information passed as "modulator" for Allosite creation from the residues of the passed "pdb", standardize the saved "modulator" info, and get the site with PyMOL
    """
    if created:
        # Check that the info passed as "modulator" is present in the pdb
        modulator_residues = site.pdb.residues.merge(utils.pd.DF(site.modulator, dtype=str)) # all site.modulator inc. _seq_id should be converted to str
        assert len(modulator_residues) > 0

        # Standardize the way in which the modulator identification/information is saved
        site.modulator = utils.simplify_residues(
            residues_subset=modulator_residues,
            residues=site.pdb.residues
        )

        # Fill the "site" column getting the site with PyMOL
        site.site = siteutils.get_site_res(site)




    