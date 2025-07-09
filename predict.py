import os, tempfile, re, subprocess
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
# uniref_path = "/data/fnerin/UniRef30_2023_02/UniRef30_2023_02" # Path to the uncompressed UniRef database


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
        pdb = Cif(pdb, filename=f"{path}/{pdb}.cif")
        
    chains = protein_chains or pdb.residues.label_asym_id.unique().tolist()
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

from colabfold.batch import get_msa_and_templates
from colabfold.utils import DEFAULT_API_SERVER
from pathlib import Path as plPath

class HHBlitsF_msa(HHBlitsF):
    def _hhblits(self, seq, entity_id, *args, **kwargs):
        fn = lambda ext: f"{self._path}/{self._jobname}_{entity_id}.{ext}"
        jobname = f"AlloPockets_{self._jobname}"

        if not os.path.isfile(fn('a3m')):
            with tempfile.TemporaryDirectory() as tmpdir:
                get_msa_and_templates(
                    jobname=jobname,
                    query_sequences= seq,
                    a3m_lines=None,
                    result_dir=plPath(tmpdir),
                    msa_mode= "mmseqs2_uniref", # earch against the UniRef database only (mmseqs2_uniref) or UniRef and ColabFoldDB (mmseqs2_uniref_env, default)
                    use_templates= False, # AlphaPulldown uses True
                    custom_template_path=None,
                    pair_mode="none",
                    host_url=DEFAULT_API_SERVER,
                    user_agent=self._email #'alphapulldown'
                )
                subprocess.run(f"cp {tmpdir}/{jobname}_all/uniref.a3m {fn('a3m')}", shell=True)

        if not os.path.isfile(fn('hhm')):
            subprocess.run(f"hhmake -i {fn('a3m')} -o {fn('hhm')} -v 0", shell=True)
            
        with open(fn("hhm"), "r") as fp:
            data = []
            seq = []
            regex = re.compile("^\w\s\d+")
            starting = 0
            lines = fp.readlines()
            for i in range(len(lines)):
                if lines[i].startswith("NULL"):
                    pieces = lines[i].split()
                    seq.append([pieces[0]])
                    data.append(
                        [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[1:21]]
                        + [0] * 10
                    )
                if lines[i].startswith("HMM    A	C	D"):
                    col_desc = lines[i].split()[1:] + lines[i + 1].split()
                    starting = 1
                if starting > 0:
                    starting += 1
                if starting >= 4 and regex.match(lines[i]):
                    pieces = lines[i].split()
                    seq.append([pieces[0]])
                    d = [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[2:22]]
                    pieces = lines[i + 1].split()
                    d += [2 ** (-int(x) / 1000) if x != "*" else 0 for x in pieces[:7]]
                    d += [0.001 * int(x) for x in pieces[7:10]]
                    data.append(d)
    
        df = pd.DataFrame(
            np.hstack((np.vstack(seq), np.vstack(data))), columns=["seq"] + col_desc
        )

        return df, None

            
def get_colabfold_msa(
    clean_pdb,
    email,
    path=path
):    
    # Establish PDB
    if type(clean_pdb) == str:
        os.makedirs(path, exist_ok=True)
        Cif.path = path
        Cif.original_cifs_path = path
        clean_pdb = Cif(clean_pdb, filename=f"{path}/{clean_pdb}.cif")
    pdb = clean_pdb.entry_id

    # Establish calc. data
    HHBlitsF_msa._jobname = pdb
    HHBlitsF_msa._email = email
    HHBlitsF_msa._path = path
        
    os.makedirs(f"{path}/features/{pdb}", exist_ok=True)
    file = f"{path}/features/{pdb}/HHBlitsF.pkl"
    if not os.path.isfile(file):
        calculated = calculate_features(pdb, HHBlitsF_msa, file, path, path)
        assert calculated, f"ColabFold MSA retrieval failed"
        

def get_features(
    clean_pdb,
    uniref_path=None,
    path=path
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
    email=None,
    uniref_path=None
):
    if all(i is None for i in [email, uniref_path]):
        print("One of 'email' or 'uniref_path' must be passed appropriately")
        return
    if email == "youremail@yourinstitution.com":
        print("Please provide a valid email")
        return
    
        
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

    # ColabFold MSA if necessary:
    get_colabfold_msa(
        clean_pdb,
        email=email,
        path=path
    )
    
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





import networkx as nx
from correlationplus.calculate import calcENMnDCC

def get_correlationplus_network(
    atoms,
    nodes,
    pdb,
    path=path
):
    # Calculate correlationplus network or read
    networkf = f"{path}/{pdb}_correlationplus.dat"
    if not os.path.isfile(networkf):
        cc_matrix = calcENMnDCC(selectedAtoms=atoms, cut_off=15, out_file=networkf) # method="ANM", 
    else:
        cc_matrix = np.loadtxt(networkf, dtype=float)
    
    # Create graph, add nodes, and then correlation edges
    G = nx.Graph()
    for i, node in nodes.iterrows():
        G.add_node(i, **node)
        
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)): # Matrix is symmetrical, only use upper
            G.add_edge(i, j, value=cc_matrix[i, j], distance=-np.log(abs(cc_matrix[i, j]) + 10E-10) + 10E-10)
            # The approach in lit. is to use -log10(|corr|) as edge weights/distances in the network for analyses
            # e.g., https://www.pnas.org/doi/full/10.1073/pnas.0810961106
    
    return G

def get_prs_network(
    prodycif,
    pdb,
    nodes,
    path=path
):
    # Calculate PRS or read
    networkf = f"{path}/{pdb}_prody_prs_matrix.dat"
    if not os.path.isfile(networkf):
        prs_mat, _ = prodycif._prs()
    else:
        prs_mat = np.loadtxt(networkf, dtype=float)
    
    # Create DIRECTED graph, add nodes, and then prs edges
    G = nx.DiGraph()
    for i, node in nodes.iterrows():
        G.add_node(i, **node)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                G.add_edge(i, j, value=prs_mat[i, j], distance=-np.log(abs(prs_mat[i, j]) + 10E-10) + 10E-10) # PRS values are always positive but abs() doesn't hurt
                # The approach in lit. is to use -log10(|corr|) as edge weights/distances in the network for analyses
                # e.g., https://www.pnas.org/doi/full/10.1073/pnas.0810961106
    
    return G

def get_pathways(
    pdb,
    pathways,
    source_pocket,
    pathway_dist_threshold=20,
    top_pathways=10,
    path=path
):
    cif = Cif(pdb, f"{path}/{pdb}.cif")
    
    # Parse the structure file with ProDy
    prodycif = ProDyF(cif)
    atoms = prodycif._cas # parseMMCIF(cif.filename).select('name CA')
    nodes = prodycif._res_df

    # Calculate correlationplus network or read and obtain Graph
    if pathways == "correlationplus":
        G = get_correlationplus_network(
            atoms,
            nodes,
            pdb,
            path=path
        )
    elif pathways == "prs":
        G = get_prs_network(
            prodycif,
            pdb,
            nodes,
            path=path
        )
    
    # Determine sources and calculate shortest paths using sources (fast calculation)
    sources = (
        nodes.merge(
            Pocket(f"{path}/{pdb}/{pdb}_out/pockets/{source_pocket}_atm.cif").residues, 
            how="left", indicator=True
        )
        .query("_merge == 'both'").drop(columns="_merge")
    ) # DataFrame with the nodes/CA atoms corresponding to the pocket to use as pathway sources
    paths_lengths, paths = nx.shortest_paths.multi_source_dijkstra(
        G, sources=sources.index.to_list(), target=None, cutoff=None, weight='distance'
    )
    
    # Determine targets based on distance from sources and filter paths
    sources_selstr = "( " + " or ".join((
        selstr
        for g, res in sources.groupby("auth_asym_id")
            for selstr in (
                f"""( chain {g} and (resnum {' '.join(f"{str(resnum)}{'_' if inscode == '?' else inscode}" for resnum, inscode in res[['auth_seq_id', 'pdbx_PDB_ins_code']].values)}) )""",
            # for insertion codes http://www.bahargroup.org/prody/manual/reference/atomic/select.html#atom-data-fields
            )
    )) + " )"
    targets = nodes.merge(
        pd.DataFrame(
            *(
                {
                    "auth_asym_id": selatoms.getChids(), # label_asym_id are stored in getSegnames; newer prody versions might make them Chids
                    "auth_seq_id": selatoms.getResnums(),
                    "pdbx_PDB_ins_code": (i or "?" for i in selatoms.getIcodes())
                }
                for selatoms in (atoms.select(f'within {pathway_dist_threshold} of {sources_selstr}'),) # CAs within X Å of source, to filter them OUT)
            ),
            dtype=str
        ),
        how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")
    selected_paths = tuple(k for k in paths_lengths if k in targets.index) # List of selected paths, sorted by ascending path length (paths_lengths is sorted)

    return pd.concat(
        cif.atoms.merge(
            nodes.loc[paths[p]].assign(label_atom_id="CA")
        ).assign(
            label_entity_id='98',
            label_asym_id=f"P{p}_top{i}",
            label_seq_id=lambda df: tuple(str(i) for i in range(1, len(df)+1)),
            occupancy = paths_lengths[p]
        )
        for i, p in enumerate(selected_paths[:top_pathways], 1)
    ), G

paultol_palette = (
    ('blue', '#4477AA'), ('cyan', '#66CCEE'), ('green', '#228833'), ('yellow', '#CCBB44'), ('red', '#EE6677'), ('purple', '#AA3377'), ('grey', '#BBBBBB'),
    ('light blue', '#77AADD'), ('light cyan', '#99DDFF'), ('mint green', '#44BB99'), ('sand', '#DDCC77'), ('pink', '#CC6677'), ('plum', '#882255'), ('pale grey', '#DDDDDD'),
    ('steel blue', '#AAAAEE'), ('sky blue', '#117733'), ('orange', '#EEDD88'), ('raspberry', '#EE8866'), ('dark purple', '#9988DD'), ('olive', '#661100'), ('light brown', '#444444')
)

def view_pockets_pathways(
    pdb, 
    pathways:("correlationplus", "prs"), # , "essa"
    source_pocket,
    pathway_dist_threshold=20,
    n_top_pathways=10,
    pathways_colors=paultol_palette,
    pockets:dict={}, # {"pocketn": {"color": ""}}
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
        pdb = Cif(pdb, filename=f"{path}/{pdb}.cif")

    chains = protein_chains or pdb.residues.label_asym_id.unique().tolist()
    pdb = pdb.entry_id
    cif = Cif(pdb, f"{path}/{pdb}_updated.cif")

    if len(pockets) > 0:
        pockets = {
            pocketn: {
                "atoms": get_pocket(pdb, pocketn, path=path),
                "color": colors.get(pocket["color"], pocket["color"])
            }
            for pocketn, pocket in pockets.items()
        }

    pathways, _ = get_pathways(pdb, pathways, source_pocket, pathway_dist_threshold, n_top_pathways, path)
    

    # Fake entity data
    entities = pd.concat((
        pd.DataFrame(cif.cif.data["_entity"], dtype=str),#.query(f"id in {minimal_elements('label_entity_id')}"),
        pd.DataFrame([
            {"id": "99", "type": "branched", "pdbx_description": "pockets"}, # Fake the pockets as carbohydrates to separate their rep. from other ligands
            {"id": "98", "type": "polymer", "pdbx_description": "pathways"}
        ]) 
    )).fillna(".")

    
    columns = list( set.intersection( *map(set, (pocket["atoms"].columns for pocket in pockets.values())) ) )
    atoms = pd.concat((
        cif.atoms[columns],
        *(pocket["atoms"][columns] for pocket in pockets.values()),
        pathways[columns]
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
        {"struct_asym_id": asym_id, 'representation': 'cartoon', 'representationColor': '#DADADA', 'focus': True}
        for asym_id in chains
    ]

    if site_residues is not None:
        data += [
            {'struct_asym_id': r["label_asym_id"], 'residue_number': int(r["label_seq_id"]), 'representationColor': colors["green"]}
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
    if len(pockets) > 0:
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

    # Pathways
    for i, (p, res) in enumerate(pathways.groupby("label_asym_id", sort=False)):
        colorn, color = pathways_colors[i % len(pathways_colors)]
        data += [
            {
                "struct_asym_id": p, 'representation': 'backbone', 'representationColor': color.lower()
            }
        ]
        print(
            f"Pathway #{p.split('_top')[-1]} ({colorn}): " + ', '.join((
                ':'.join((
                    r['auth_asym_id'],
                    r['auth_comp_id'],
                    r['auth_seq_id'] + r['pdbx_PDB_ins_code'].replace("?", "")
                ))
                for ri, r in res.iterrows()
            ))
        )

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