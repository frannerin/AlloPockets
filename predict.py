import os, tempfile
import pandas as pd
from tqdm.notebook import tqdm


import sys
sys.path.append("training_data")

from utils.new_pdbs import Pdb, cached_property
from utils.structure_fixing import get_fixed_structure, CifFileWriter
from utils.utils import Cif as BaseCif

class Cif(BaseCif):
    @cached_property
    def _protein_entities(self):
        return (
            pd.DF(self.cif.data["_entity_poly"], dtype=str)
            .query("type == 'polypeptide(L)'")
            .entity_id.to_list()
        )


path = "predict" # Path to write files and results to
uniref_path = "/data/fnerin/UniRef30_2023_02/UniRef30_2023_02" # Path to the uncompressed UniRef database


def get_cif(
    pdb_id,
    path=path
):
    os.makedirs(path, exist_ok=True)
    Pdb.path = path
    Pdb.original_cifs_path = path
    
    pdb = Pdb(pdb_id.lower())

    # Save original and uncompressed cif
    with open(f"{path}/{pdb.entry_id}_updated.cif.gz", "wb") as f:
        f.write(pdb.cif._cif_content)
    with open(f"{path}/{pdb.entry_id}_updated.cif", "w") as f:
        f.write(pdb.cif.text)
        
    # Cache the contents of the file
    pdb.cif.data
    return pdb

import pymol2

def get_site(site, only_protein=True, threshold=6): # site.pdb CAN BE PDB OR ASSEMBLY (must have .cif and .residues)
    """
    Function to, given a site, return a standardized list of residues from the parent structure that define the site with the Python interface of open-source PyMOL
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
        with tempfile.NamedTemporaryFile("w+", suffix=".cif") as f:
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

    # Transform the PyMOL-derived residue identifiers into a standard table of residues that can be used to retrieve the rows/residues from the parent structure's .residues table
    site_res = site.pdb.residues.merge(
        pd.DataFrame(
            site_list,
            columns=[
                "label_asym_id", "auth_asym_id", "auth_comp_id",
                "auth_seq_id", "pdbx_PDB_ins_code"
            ],
            dtype=str
        )
    ).query("pdbx_PDB_model_num == '1'")

    if only_protein:
        site_res = site_res.query(f"label_entity_id in {site.pdb._protein_entities} and label_asym_id not in {site.modulator_residues.label_asym_id.unique().tolist()}")

    assert len(site_res) > 0, "Site selection doesn't have any residues"

    return site_res

class Site:
    def __init__(self, pdb, modulator_residues=None, residues=None, only_protein=True, distance_threshold=6):
        self.pdb = pdb
        if modulator_residues is not None:
            self.modulator_residues = modulator_residues
            self.residues = get_site(self, only_protein=only_protein, threshold=distance_threshold)
        elif residues is not None:
            self.residues = pdb.residues.merge(pd.DataFrame(residues, dtype=str)).query("pdbx_PDB_model_num == '1'")
            if only_protein:
                self.residues = self.residues.query(f"label_entity_id in {self.pdb._protein_entities}")
        else:
            raise Exception("Pass one of 'modulator_residues' or 'residues'")

def get_clean_pdb(pdb, protein_chains, path=path):
    os.makedirs(path, exist_ok=True)
    Cif.path = path
    Cif.original_cifs_path = path
    
    fixed_structure = get_fixed_structure(pdb, pdb, list(protein_chains), path, save=True)
    with open(f"{path}/{pdb.entry_id}.cif", "w+") as f:
        writer = CifFileWriter(f.name, compress=False)
        writer.write({
            pdb.entry_id.upper(): {
                "_atom_site": fixed_structure.to_dict(orient="list"),
                "_entity_poly": pdb.cif.data["_entity_poly"]
            }
        })

    cif = Cif(pdb.entry_id)
    # Cache the contents of the files
    cif.origcif.data
    cif.cif.data
    return cif

from ipymolstar import PDBeMolstar

def view_pdb(pdb, **kwargs):
    return PDBeMolstar(
        custom_data = {
                'data': pdb.cif.text,
                'format': 'cif',
                'binary': False,
            },
        sequence_panel = True,
        assembly_id='',
        **kwargs
    )

colors = {
    "orange": "#0FD55E00".lower(),
    "green": "#0F009E73".lower(),
    "blue": "#0F0072B2".lower()
}



def get_pockets(
    clean_pdb,
    path=path
):
    if not os.path.isdir(f"{path}/{clean_pdb.entry_id}/{clean_pdb.entry_id}_out"):
        os.makedirs(f"{path}/{clean_pdb.entry_id}", exist_ok=True)
        os.system(f"cp {clean_pdb.filename} {path}/{clean_pdb.entry_id}/")
        os.system(f"fpocket -m 3 -M 6 -i 35 --file {path}/{clean_pdb.entry_id}/{clean_pdb.entry_id}.cif")

    return pd.DataFrame((
        {"pocket": (
            pocketf.split("_")[0]
            for pocketf in os.listdir(f"{path}/{clean_pdb.entry_id}/{clean_pdb.entry_id}_out/pockets")
                if pocketf.endswith(".cif")
        )}
        
    ))



def get_pocket(pdb, pocket, path=path):
    pocketn = pocket.replace('pocket', '')
    pocket_atoms = (
        Cif(pdb, f"{path}/{pdb}/{pdb}_out/{pdb}_out.cif", name=f"{pdb}_out")
        .atoms
        .query(f"label_comp_id == 'STP' and label_seq_id == '{pocketn}'")
    )
    pocket_atoms["label_asym_id"] = 'ZZZ'
    pocket_atoms["label_entity_id"] = '99'

    return pocket_atoms
    

def view_pockets(
    pdb,
    pockets:dict, # {"pocketn": {"color": ""}}
    protein_chains=None,
    site_residues=None,
    modulator_residues=None,
    path=path
):
    # Establish PDB
    if type(pdb) == str:
        os.makedirs(path, exist_ok=True)
        Cif.path = path
        Cif.original_cifs_path = path
        pdb = Cif(pdb, filename=f"{path}/{pdb}_updated.cif")
        
    chains = protein_chains or pdb.residues.query(f"label_entity_id in {pdb._protein_entities}").label_asym_id.unique().tolist()
    pdb = pdb.entry_id
    cif = Cif(pdb, f"{path}/{pdb}_updated.cif") # Final cif file has to be the complete cif file regardless

    pockets = {
        pocketn: {
            "atoms": get_pocket(pdb, pocketn, path=path),
            "color": colors.get(pocket["color"], pocket["color"])
        }
        for pocketn, pocket in pockets.items()
    }

    # Fake entity data
    entities = pd.concat((
        pd.DataFrame(cif.cif.data["_entity"], dtype=str),#.query(f"id in {minimal_elements('label_entity_id')}"),
        pd.DataFrame([{"id": "99", "type": "branched", "pdbx_description": "pockets"}]) # Fake the pockets as carbohydrates to manage their representation
    )).fillna(".")

    columns = list( set.intersection( *map(set, (pocket["atoms"].columns for pocket in pockets.values())) ) )
    atoms = pd.concat((
        cif.atoms[columns], # if label_asym_id in protein_chains or modulator_residues.chains or label_entity_id not in protein_entities
        *(pocket["atoms"][columns] for pocket in pockets.values())
    ))

    with tempfile.NamedTemporaryFile("w+", suffix=".cif") as f:
        writer = CifFileWriter(f.name)
        writer.write({cif.entry_id.upper(): {
            "_entity": entities.to_dict(orient="list"),
            "_atom_site": atoms.to_dict(orient="list"),
        }})
        combined = Cif(pdb, filename=f.name)
        combined.cif.data # to cache it while 'f' exists

    data = [
        # Protein
        {"struct_asym_id": asym_id, 'representation': 'cartoon', 'representationColor': '#AEAEAE', 'focus': True}
        for asym_id in chains
    ]

    if site_residues is not None:
        data += [
            {'struct_asym_id': r["label_asym_id"], 'residue_number': int(r["label_seq_id"]), 'representationColor': "black"}
            for i, r in site_residues.iterrows()
        ]
        
    # Ligands and molecules
    if modulator_residues is not None:
        data += [
            {'struct_asym_id': r["label_asym_id"], 'color': 'white'}
            for i, r in (
                combined.residues
                # Not modulator residues and only small molecule entities
                .merge(
                    modulator_residues if modulator_residues is not None else pd.DataFrame(columns=combined.residues.columns), # if modulator_residues not passed, empty df
                    how="outer", indicator=True
                )
                .query(f"""_merge == 'left_only' and label_entity_id in {entities.query("type == 'non-polymer'").id.unique().tolist()}""")
                .drop(columns="_merge")
                .iterrows()
            
            )
        ]

    # Pockets
    data += [
        {
            "struct_asym_id": "ZZZ", 'residue_number': int(pocketn.replace('pocket', '')), 'representation': 'point', 'representationColor': pocket["color"]
        }
        for pocketn, pocket in pockets.items()
    ]

    data += [
        {
            "struct_asym_id": "ZZZ", 'residue_number': int(pocketn.replace('pocket', '')), 'representation': 'gaussian-volume', 'representationColor': pocket["color"]
        }
        for pocketn, pocket in pockets.items()
    ]

    return view_pdb(
        combined,
        
        hide_polymer = True,
        # hide_heteroatoms = True,
        # hide_non_standard = True,
        hide_carbs = True,
        hide_water = True,
        
        color_data = {
            "data": data,
            "nonSelectedColor": None,
            "keepColors": True,
            "keepRepresentations": False,
        }
    )

from utils.pocket_utils import Pocket, get_pockets_info, get_mean_pocket_features
from utils.features_classes import * # Each FClass
from utils.features_utils import calculate_features, get_pdb_features

# Path to the mkdssp executable downloaded from https://github.com/PDB-REDO/dssp/releases/tag/v4.4.0
BiopythonF.dssp_path = "training_data/utils/external/mkdssp-4.4.0-linux-x64" 
os.chmod(BiopythonF.dssp_path, 0o755)
# f"mkdssp --mmcif-dictionary {os.environ['CONDA_PREFIX']}/share/libcifpp/mmcif_pdbx.dic"#"training_data/utils/external/mkdssp-4.4.0-linux-x64"



def get_features(
    clean_pdb,
    path=path,
    uniref_path=uniref_path
):
    os.makedirs(f"{path}/features/{clean_pdb.entry_id}", exist_ok=True)
    HHBlitsF.uniref_path = uniref_path

    progressbar = tqdm(FClasses)
    for fc in progressbar:
        progressbar.set_description(f"Calculating {fc.__name__[:-1]}")
        file = f"{path}/features/{clean_pdb.entry_id}/{fc.__name__}.pkl"
        if not os.path.isfile(file):
            calculated = calculate_features(clean_pdb.entry_id, fc, file, path, path)
            assert calculated, f"Feature calculation failed: {fc}"
            
    return get_pdb_features(
        clean_pdb,
        sites = [pd.DataFrame(columns=clean_pdb.residues.columns),],
        features_path = path
    )


def get_pockets_features(
    clean_pdb,
    pockets,
    features,
    path=path
):
    pockets_features = pd.concat(
        (
            pockets,
            pockets.apply(
                lambda row: pd.Series(
                    Pocket(f"{path}/{clean_pdb.entry_id}/{clean_pdb.entry_id}_out/pockets/{row['pocket']}_atm.cif").feats
                ), axis=1
            )
        ),
        axis=1
    )

    cols = ["pdb", "pocket"] + [c for c in pockets_features.columns if c in ["nres", "site_in_pocket", "pocket_in_site", "label"]]
    
    pockets_features = pd.concat(
        (
            pockets_features[cols],
            # pockets_features["label"],
            pockets_features.drop(columns=cols) # "label", 
        ),
        axis=1,
        # keys=["Pockets", "Label", "FPocket"]
        keys=["Pockets", "FPocket"]
    )
    

    return pd.concat(
        (
            pockets_features,
            pockets_features.apply(
                lambda row: get_mean_pocket_features(
                    row[("Pockets", "pdb")],
                    row[("Pockets", "pocket")],
                    pdb_features = features,
                    pockets_path = path # # f"{pockets_path}/{pdb}/{pdb}_out/pockets/{pocket}_atm.cif"
                ), 
                axis=1 
            )
        ),
        axis=1
    )



from autogluon.tabular import TabularDataset, TabularPredictor

model = TabularPredictor.load("models/pockets_physchem_deploy")

def prepare_data(df):
    df.index = df["Pockets"][["pdb", "pocket"]].apply(lambda x: "_".join(x), axis=1)
    df = df.drop(columns=["Pockets"], level=0)
    df.columns = map(lambda x: "_".join(x), df.columns.values)
    # df.loc[:,'Label_label'] = df['Label_label'].astype("category")
    return TabularDataset(df)




def predict(
    pdb,
    protein_chains=None,
    path=path,
    uniref_path=uniref_path
):
    # Establish PDB
    if type(pdb) == str:
        os.makedirs(path, exist_ok=True)
        Cif.path = path
        Cif.original_cifs_path = path
        pdb = Cif(pdb, filename=f"{path}/{pdb}_updated.cif")

    # Clean PDB
    protein_chains = protein_chains or pdb.residues.query(f"label_entity_id in {pdb._protein_entities}").label_asym_id.unique().tolist()

    clean_pdb = get_clean_pdb(
        pdb,
        protein_chains=protein_chains,
        path=path
    )

    # Pockets
    pockets = get_pockets(
        clean_pdb,
        path=path
    )
    pockets["pdb"] = clean_pdb.entry_id

    # Features
    features = get_features(
        clean_pdb,
        path=path,
        uniref_path=uniref_path
    )

    pockets_features = get_pockets_features(
        clean_pdb,
        pockets,
        features,
        path=path
    )
    
    data = prepare_data(pockets_features)
    
    preds = model.predict_proba(data)[[1]].sort_values(1, ascending=False).rename(columns={1: "Allosteric score"})
    preds.index = preds.index.map(lambda x: x.split("_")[-1])
    return preds