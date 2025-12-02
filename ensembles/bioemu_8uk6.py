import os, sys, subprocess, mdtraj
import pandas as pd
from tqdm import tqdm

sys.path.append("..")

from predict import \
get_cif, get_pockets, Pocket, \
get_pdb_features, get_pockets_features, \
prepare_data, model



realsite = pd.DataFrame([{"label_asym_id": "A", "label_seq_id": str(seqnum)} for seqnum in (49, 50, 51, 52, 55, 280, 281, 282, 283, 284, 285, 288, 300, 304, 305, 306, 307, 486, 487, 488, 493)])

outdir = "bioemu_8uk6/msmclust"

for feat_name in ["dihedrals", "sasa", "distances"]:
    outpath = f"{outdir}/{feat_name}"
    os.makedirs(f"{outpath}/predictions", exist_ok=True)
    
    overlapsf = f"{outpath}/predictions.pkl"
    if os.path.isfile(overlapsf):
        overlaps = pd.read_pickle(overlapsf)
    else:
        overlaps = {}
    
    
    for i, frame in tqdm(tuple(enumerate(
        mdtraj.load(f"{outpath}/samples_sidechain_rec.xtc", top=f"{outpath}/samples_sidechain_rec.pdb")
    ))):
        name = f"frame_{i}"    
        path = f"{outpath}/predictions/{name}"
        os.makedirs(path, exist_ok=True)
        
        pdbf = f"{path}/{name}.pdb"
        frame.save_pdb(pdbf) # better to save pdb so that the cif name is handled
        
        clean_pdb = get_cif(file=pdbf, path=path)
        os.system(f"ln -s {clean_pdb.entry_id}_updated.cif {path}/{clean_pdb.entry_id}.cif")
        clean_pdb.filename = f"{path}/{clean_pdb.entry_id}.cif"
    
        predsf = f"{path}/preds.pkl"
        if not os.path.isfile(predsf):
            pockets = get_pockets(
                clean_pdb,
                out=sys.stdout,
                path=path
            )
            pockets["pdb"] = clean_pdb.entry_id
    
            if i != 0:
                os.makedirs(f"{path}/features/{clean_pdb.entry_id}", exist_ok=True)
                os.system(f"ln -s ../../../frame_0/features/frame_0/HHBlitsF.pkl {path}/features/{clean_pdb.entry_id}/HHBlitsF.pkl")

            try:
                for feat in ['features', 'transferentropy', 'hhblits']:
                    subprocess.run(f"python ../gradio/{feat}.py {clean_pdb.entry_id} {path}", shell=True, check=True)
            except Exception as e:
                print(f"{clean_pdb.entry_id} {feat_name} failed: {e}")
                continue
        
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
        
        overlaps[name] = {
            "pocket": topp,
            "overlap": len(merge) / len(site_in_apo),
            "merge": merge
        }
    
        pd.to_pickle(overlaps, overlapsf)