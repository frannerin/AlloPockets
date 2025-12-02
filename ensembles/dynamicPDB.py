#!/usr/bin/env python
# coding: utf-8

# # dynamicPDB in allodb
# 
# Assuming that the chain is the `auth_asym_id`.

# In[2]:


import pandas as pd


# In[5]:

basedir = "/data/fnerin/dpdbcifs"


csv = pd.read_csv("data/dynamicPDB_PDB_ID.csv")
csv


# In[7]:


assert len(csv) == len(csv.pdb_name.unique())



dpdb = csv.pdb_name.unique().tolist()



from src import *


# In[10]:


db.init('database.db')


# ## Download cifs

# In[18]:


from src import utils
import os

import tempfile
from src.cifutils import MMCIF2Dict, CifFileWriter

class PDBCif:
    def __init__(self, pdb):
        self.filename = f"{basedir}/{pdb}_updated.cif.gz"
        self._name = pdb.upper()

    @property
    def data(self):
        return MMCIF2Dict().parse(self.filename)[self._name]

    def _get_or_save_text(self, filename=None, compress=False):
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

    @property
    def text(self):
        return self._get_or_save_text()

    def save(self, filename):
        return self._get_or_save_text(filename=filename)



class Pdb:
    def __init__(self, pdb):
        self.entry_id = pdb
    
    @property
    def cif(self):
        return PDBCif(self.entry_id)

    @property
    def atoms(self):
        return utils.pd.DF(self.cif.data["_atom_site"], dtype=str)

    @property
    def residues(self):
        return (
            self.atoms.drop(
                [
                    'group_PDB', 'id', 'type_symbol',
                    'auth_atom_id', 'label_atom_id', 'label_alt_id',
                    'Cartn_x', 'Cartn_y', 'Cartn_z',
                    'occupancy', 'B_iso_or_equiv', 'pdbx_formal_charge'
                ],
                axis=1
            )
            .drop_duplicates()
        )

    @property
    def _entities(self):
        return (
            utils.pd.DF(self.cif.data["_entity_poly"], dtype=str)
        )

    @property
    def _protein_entities(self):
        return (
            self._entities
            .query("type == 'polypeptide(L)'")
            .entity_id.to_list()
        )    


# # Process

# In[25]:


def get_protein_label_asym_ids(chain_res, pdb):
    "Get the id and residues of each label_asym_id of the passed anno. chain (auth_asym_id)'s residues that corresponds to a protein"
    return dict(
        (label_asym_id, label_asym_id_res)
            for label_asym_id in chain_res.label_asym_id.unique()
                for label_asym_id_res in [
                    pdb.residues.query(f"label_asym_id == '{label_asym_id}'")
                ]
                    if (
                        label_asym_id_res
                        .label_entity_id.unique().squeeze() 
                        in pdb._protein_entities
                    )
    )

def get_uniprots(res):
    "Get the Uniprot id and info (residues DataFrame, min and max) for each Uniprot in the passed residues"
    return {
        u: {
            "res": ures,
            "min": n.min(),
            "max": n.max(),
                     
        }  
            for u, ures in res.groupby("pdbx_sifts_xref_db_acc")
                if u != "?"
                    for n in [ures.pdbx_sifts_xref_db_num.astype(int),] 
    }


def get_uniprot_monomer_sites(u):
    "Get all monomer sites with info (label_entity_id, label_asym_id, and the results of get_uniprots function for the label_asym_id' residues)"
    for s in Site.raw(
f'''
SELECT Site.*
FROM Site, json_each(Site.info, '$.interacting_chains_info')
WHERE json_extract(json_each.value, '$.Uniprot') LIKE '%{u}%'
'''
    ):
        entities = [
            c
            for c in s.info["interacting_chains_info"]
                if c["polymer_type"] == "polypeptide(L)" 
        ]
        if len(entities) > 1: continue
            
        label_asym_id = s.nonredundant_site.protein_residues.label_asym_id.unique().tolist() # not homomer (only 1 polypeptidic interacting entity AND chain)
        if len(label_asym_id) > 1: continue
    
        yield (s, entities[0]["label_entity_id"], label_asym_id[0], get_uniprots(s.pdb.residues.query(f"label_asym_id == '{label_asym_id[0]}'")))
        


# In[26]:


import pymol2

def get_chain_ligands(pdb, label_asym_id, threshold=4):
    sele = f"{label_asym_id}///*"

    with pymol2.PyMOL() as pymol:
        pymol.cmd.feedback(
            "disable", "executive", "details"
        )  # to silence "ExecutiveLoad-Detail: Detected mmCIF"
        pymol.cmd.load(pdb.cif.filename)

        # Retrieve all atoms within the threshold of the modulator selection
        atoms = pymol.cmd.get_model(f"(br. all within {threshold} of {sele}) and not {sele}")
        
    # Process the atom selection to obtain residue identifiers
    residue_ids = set(
        tuple(
            (
                a.segi, a.chain, a.resn,
                a.resi_number, a.ins_code or '?' # pdbx_PDB_ins_code or "?" if none
            ) 
            for a in atoms.atom
        )
    )

    # Transform the PyMOL-derived residue identifiers into a standard table of residues that can be used to retrieve the rows/residues from the parent structure's .residues table
    residues = pdb.residues.merge(
        pd.DataFrame(
            residue_ids,
            columns=[
                "label_asym_id", "auth_asym_id", "auth_comp_id",
                "auth_seq_id", "pdbx_PDB_ins_code"
            ],
            dtype=str
        )
    ).query("auth_comp_id != 'HOH'")

    # If there are any residues
    if len(residues) > 0:
        # Get the cif field with entity data        
        edata = pd.DataFrame(pdb.cif.data["_entity"], dtype=str)

        # Get dictionaries of the info of only non-polymer items
        ligands = [
            e
            for e in (
                edata
                .query(f"id in {residues.label_entity_id.unique().tolist()} and type == 'non-polymer'")[
                    ["id", "pdbx_description", "formula_weight"]
                ].to_dict("records")
            )
        ]

        # If there are any non-polymer items, return the dictionary with info
        if len(ligands) > 0:
            return {
                ("dpdb", "ions"): [e for e in ligands if "ion" in e["pdbx_description"].lower()],
                ("dpdb", "ligands"): [e for e in ligands if "ion" not in e["pdbx_description"].lower()]
            }


# In[27]:


import numpy as np


# In[89]:


warnings = {}

def get_matches(p):
    split = p.split("_")
    pdb = Pdb(split[0])
    
    if len(split) == 2:
        c = split[1]
        # sc: chain exists and is an auth_asym_id
        assert c in pdb.residues.auth_asym_id.unique()
        
    elif len(split) == 1: #assuming there will be only 1 polypeptidic chain
        c = pdb.residues.query(f"label_entity_id in {pdb._protein_entities}").auth_asym_id.drop_duplicates().item() # will fail if there's more than 1 chain
    
    # Get the residues of the annotated chain
    cres = pdb.residues.query(f"auth_asym_id == '{c}'")
    
    # Get the residues of each label_asym_id of the anno. chain that corresponds to a protein
    prot_label_asym_ids = get_protein_label_asym_ids(cres, pdb)
    # sc: at least there has to be one protein; but it would be weird if there's more than 1
    assert len(prot_label_asym_ids) > 0
    if len(prot_label_asym_ids) > 1: 
        warnings[p] = f"{p} has more than 1 protein label_asym_ids ({prot_label_asym_ids.keys()}) in the chain {c}"
        print(warnings[p])

    # For each protein label_asym_id, if it has a Uniprot annotation
    for prot_label_asym_id, prot_label_asym_id_res in prot_label_asym_ids.items():        
        if "pdbx_sifts_xref_db_name" in prot_label_asym_id_res:

            ligands = get_chain_ligands(pdb, prot_label_asym_id)
            
            # Get the dictionaries of information of the Uniprots present in it; expected 1
            ups = get_uniprots(prot_label_asym_id_res)
            if len(ups) > 1: 
                warnings[p] = f"{p} has more than 1 Uniprots in the chain {c} (label_asym_id {prot_label_asym_id}): {list(ups.keys())}"
                print(warnings[p])

            # For each Uniprot present in the label_asym_id
            for u, uinfo in ups.items():
                # Start a dictionary with base information about the chain and Uniprot
                d = {
                    ("dpdb", "p"): p,
                    ("dpdb", "pdb"): pdb.entry_id,
                    ("dpdb", "auth_asym_id"): c,
                    ("dpdb", "label_asym_id"): prot_label_asym_id,
                    ("dpdb", "label_entity_id"): prot_label_asym_id_res.label_entity_id.unique().squeeze(),
                    ("dpdb", "uniprot"): u,
                    ("dpdb", "u_min"): uinfo["min"],
                    ("dpdb", "u_max"): uinfo["max"],
                    # ("dpdb", "u_len"): uinfo["len"], ####### ?
                }
                d.update(ligands or {})
                
                # Get the sites of allodb that contain the present Uniprot id and the dictionaries of information of them
                sites = get_uniprot_monomer_sites(u)
                urange = range(uinfo["min"], uinfo["max"]+1)

                # For each site match (monomers or homomers)
                for s, slabel_entity_id, slabel_asym_id, sites_uniprots in sites:
                    # And for each Uniprot found in the site
                    for su, suinfo in sites_uniprots.items():
                        surange = range(suinfo["min"], suinfo["max"]+1)
                        minmax = len(np.intersect1d(urange, surange))
                        
                        merge = len(
                            uinfo["res"]
                            .merge(
                                suinfo["res"],
                                on = ["pdbx_sifts_xref_db_name", "pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"],
                                how = "outer", indicator = True
                            ).query("_merge == 'both'")
                        )
                        
                        s = ({
                            ("allodb", "pdb"): s.pdb.entry_id,
                            ("allodb", "site_id"): s.id,
                            ("allodb", "label_entity_id"): slabel_entity_id,
                            ("allodb", "label_asym_id"): slabel_asym_id,
                            ("allodb", "uniprot"): u,
                            ("allodb", "u_min"): suinfo["min"],
                            ("allodb", "u_max"): suinfo["max"],
                            # ("allodb", "u_len"): uinfo["len"], ####### ?
                        
                            ("overlap_minmax", "atlas_in_allodb"): minmax/len(urange),
                            ("overlap_minmax", "allodb_in_atlas"): minmax/len(surange),
                            ("overlap_merge", "atlas_in_allodb"): merge/len(uinfo["res"]),
                            ("overlap_merge", "allodb_in_atlas"): merge/len(suinfo["res"]),
                        })
                        
                        sd = d.copy()
                        sd.update(s)
                        yield sd


# In[3]:


import pandas as pd
from tqdm.notebook import tqdm


# In[4]:


processed = []
matches = pd.DataFrame()
errors = []


# In[ ]:


for p in dpdb:#tqdm(dpdb, smoothing=0):
    if p not in processed:
        try:
            match = pd.DataFrame(get_matches(p))
        except:
            errors.append(p)
            continue
    
        if len(match) > 0:
            matches = pd.concat([
                matches,
                match
            ])
        processed.append(p)


# In[5]:


import pickle

with open("data/dpdb_processed_list.pkl", "wb") as f:
    pickle.dump(processed, f)


# In[6]:


with open("data/dpdb_errors.pkl", "wb") as f:
    pickle.dump(errors, f)

with open("data/dpdb_warnings.pkl", "wb") as f:
    pickle.dump(warnings, f)


# In[7]:


matches.to_pickle("data/dpdb_processed_dpdb.pkl")