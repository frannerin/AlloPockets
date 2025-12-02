import os, sys, subprocess, pickle
import pandas as pd
from tqdm import tqdm

sys.path.append("..")

from predict import \
get_cif, get_clean_pdb, Cif, Site, get_pockets, Pocket, \
get_pdb_features, get_pockets_features, \
prepare_data, model, view_pockets
# get_features

extra_apos_path = "pdb_ensemble"
os.makedirs(f"{extra_apos_path}/predictions", exist_ok=True)

extra_apos_original_cifs_path = f"{extra_apos_path}/origcifs"


apos = pd.read_pickle(f"{extra_apos_path}/apos.pkl")

holos_ups = {
    '7gqu': 'Q14191',
    '7yg5': 'Q15878',
    '8aq6': 'Q9GV45',
    '8f4s': 'P0DTD1',
    '8qni': 'Q13191',
    # '8uk6': 'A0A1D8PQM9',
    '8v81': 'P13569',
    '9dnm': 'P29066',
    '8jp0': 'P32418'
} # already with -2 correction

extra_holos_sitesf = "../training_data/7.Extra_set/news_sites.pkl"
with open(extra_holos_sitesf, "rb") as f:
    extra_holos_sites = {k: v for k, v in pickle.load(f).items() if k in holos_ups}

overlapsf = f"{extra_apos_path}/predictions.pkl"

if os.path.isfile(overlapsf):
    overlaps = pd.read_pickle(overlapsf)
else:
    overlaps = {}

for holo, aposd in tqdm(apos.items(), desc="### HOLOS ###"):
    if holo not in overlaps:
        overlaps[holo] = {}
    
    holocif = Cif(holo, f"../training_data/7.Extra_set/cifs/{holo}.cif")
    u = holos_ups[holo]
    umin, umax = next(((min(us), max(us)) for us in (holocif.residues.query(f"pdbx_sifts_xref_db_acc == '{u}'")["pdbx_sifts_xref_db_num"].astype(int),)))
    realsite = extra_holos_sites[holo][0]["site"][["pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"]].query(f"pdbx_sifts_xref_db_acc == '{u}'")

    for apo, chain in tqdm(aposd.items(), desc=holo):
        if apo in overlaps[holo] or apo in os.listdir("../training_data/8.Apos/Extra_set/features") or apo in ["7snx", "6wks", "8fzq"]:
            continue
            
        origciff = f"{extra_apos_original_cifs_path}/{apo}_updated.cif.gz"
        
        path = f"{extra_apos_path}/predictions/{apo}"
        os.makedirs(path, exist_ok=True)
        os.system(f"ln -s ../../origcifs/{apo}_updated.cif.gz {path}/")

        apocif = Cif(apo, origciff)
        seqmin, seqmax = (
            apocif.atoms
            .loc[lambda d: (
                d["pdbx_sifts_xref_db_acc"].eq(u) 
                & pd.to_numeric(d["pdbx_sifts_xref_db_num"], errors="coerce").between(umin, umax)
            )]
            .iloc[[0,-1]]["label_seq_id"].astype(int).to_list()
        )
        apocif.atoms = apocif.atoms.loc[lambda d: (
            pd.to_numeric(d["label_seq_id"], errors="coerce").between(seqmin, seqmax)
        )]

        if len(set(apocif.residues.pdbx_sifts_xref_db_acc.unique()) - {"?",} ) != 1:
            print(f"#### APO WITH MULTIPLE UNIPROTS: {apo}")
        clean_pdb = get_clean_pdb(apocif, chain, path)

        predsf = f"{path}/preds.pkl"
        if not os.path.isfile(predsf):
            pockets = get_pockets(
                clean_pdb,
                out=sys.stdout,
                path=path
            )
            pockets["pdb"] = clean_pdb.entry_id
        
            for feat in ['features', 'transferentropy', 'hhblits']:
                subprocess.run(f"python ../gradio/{feat}.py {clean_pdb.entry_id} {path}", shell=True, check=True)
        
            features = get_pdb_features(
                clean_pdb,
                sites = [pd.DataFrame(columns=clean_pdb.residues.columns),],
                features_path = path
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
            
            preds.to_pickle(predsf)
        else:
            preds = pd.read_pickle(predsf)

        topp = preds.iloc[0].name
        
        site_in_apo = clean_pdb.residues.merge(realsite) # hopefully will preserve all columns of clean_pdb.residues

        merge = site_in_apo.merge(
            Pocket(f"{path}/{clean_pdb.entry_id}/{clean_pdb.entry_id}_out/pockets/{topp}_atm.cif").residues
        )
        
        overlaps[holo][apo] = {
            "pocket": topp,
            "overlap": len(merge) / len(site_in_apo),
            "site": site_in_apo,
            "merge": merge
        }

        pd.to_pickle(overlaps, overlapsf)