#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import sys, os, tempfile, shutil, subprocess, json, requests
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import molviewspec as mvs


# In[2]:


sys.path.append("..")


# In[3]:


from predict import \
Cif, Pocket, view_pockets


# In[4]:


outdir = "cofolding"
os.makedirs(outdir, exist_ok=True)


# ## Data

# In[5]:


extra_holos_featuresd = pd.read_pickle("../training_data/7.Extra_set/features.pkl")
extra_holos_featuresd


# In[6]:


extra_holos_sites = {k: v for k, v in pd.read_pickle("../training_data/7.Extra_set/news_sites.pkl").items() if k in extra_holos_featuresd}
extra_holos_sites


# In[7]:


apos = pd.read_pickle(f"pdb_ensemble/apos.pkl")
overlaps = pd.read_pickle(f"pdb_ensemble/predictions.pkl")
origapos = pd.read_pickle("../models/other_tools_apos/models_lenient_labelling.pkl")["model5"]


# #### 8jp0 - 8sgi

# In[8]:


overlaps["8jp0"].pop("8sgi")


# # Sequences

# In[23]:


seqsf = f"{outdir}/seqs.pkl"

if os.path.isfile(seqsf):
    seqs = pd.read_pickle(seqsf)
    
seqsdf = pd.DataFrame({k: {k1: v1 for k1, v1 in v.items() if k1 != "site"} for k, v in seqs.items()}).T
seqsdf


# ## A3Ms

# In[11]:


msadir = f"{outdir}/a3ms"



guide_res = {}

top=1
topcounts=3


# In[13]:


for holo in tqdm(seqsdf.index):
    if holo == "8uk6":
        continue
    print(holo)
    refapo = next(apo for apo in apos[holo] if apo in os.listdir("../training_data/8.Apos/Extra_set/features"))
    refaporesults = pd.DataFrame( origapos["results"][refapo] ).T.sort_values("prob", ascending=False)
    refapocif = Cif(refapo, f"../training_data/8.Apos/Extra_set/cifs/{refapo}.cif")

    apooverlaps = []
    aposres = []
    for i in range(top):
        apooverlaps.append(refaporesults.iloc[i]["pocket_in_site"])
        aposres.append(
            refapocif.residues.merge(
                Pocket(f"../training_data/8.Apos/Extra_set/pockets/{refapo}/{refapo}_out/pockets/{refaporesults.iloc[i].name}_atm.cif").residues
            )[
                ["pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"]
            ]
        )
        
    for apo, apod in overlaps[holo].items():
        preds = pd.read_pickle(f"pdb_ensemble/predictions/{apo}/preds.pkl")
        apores = Cif(apo, f"pdb_ensemble/predictions/{apo}/{apo}.cif").residues
        apooverlaps.append(apod["overlap"]) # only top1
        for i in range(top):
            aposres.append(
                apores.merge(
                    Pocket(f"pdb_ensemble/predictions/{apo}/{apo}/{apo}_out/pockets/{preds.iloc[i].name}_atm.cif").residues
                )[
                    ["pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"]
                ]
            )
    
    counts = refapocif.residues.merge(
        pd.concat(aposres).value_counts().to_frame().reset_index(),
        how="right"
    )

    topres = pd.DataFrame()
    for _, g in counts.groupby("count", sort=False):
        topres = pd.concat((
            topres, 
            g.sample(n=min(3-len(topres), len(g)), random_state=0)
        ))
        if len(topres) == 3:
            break

    all_topres = counts.loc[lambda x: x["count"].isin(counts.iloc[:topcounts]["count"])]
    # topres = counts.loc[lambda x: x["count"].isin(counts.iloc[:topcounts]["count"])].sample(n=3, random_state=0)

    print("\t", "".join(
        topres
        .sort_values("pdbx_sifts_xref_db_num", key=lambda x: x.astype(int))
        ["pdbx_sifts_xref_db_res"]
    ))

    holo_topres = (
        Cif(holo, f"../training_data/7.Extra_set/cifs/{holo}.cif")
        .residues.replace({"pdbx_sifts_xref_db_acc": {"P32418-2": "P32418"}})
        .merge(topres[["pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"]])
        .sort_values("label_seq_id", key=lambda x: x.astype(int))
    )
    ids = sorted(holo_topres.label_seq_id.astype(int) - seqs[holo]["seqmin"])

    all_ids = list(
        Cif(holo, f"../training_data/7.Extra_set/cifs/{holo}.cif")
        .residues.replace({"pdbx_sifts_xref_db_acc": {"P32418-2": "P32418"}})
        .merge(all_topres[["pdbx_sifts_xref_db_acc", "pdbx_sifts_xref_db_num"]])
        .sort_values("label_seq_id", key=lambda x: x.astype(int))
        .label_seq_id.astype(int)
        - seqs[holo]["seqmin"] + 1 # to make them 1-indexed as label_seq_id
    )
    
    print("\t", "".join(
        pd.DataFrame(list(seqs[holo]["seq"]))
        .iloc[ids]
        [0]
    ))

    top_overlap = (
        extra_holos_sites[holo][0]["site"]
        .replace({"pdbx_sifts_xref_db_acc": {"P32418-2": "P32418"}})
        .merge(holo_topres)
    )

    guide_res[holo] = {
        "seqids": tuple(pd.Series(ids) + 1), # to make them 1-indexed as label_seq_id
        "all_seqids": all_ids,
        "mean_overlap": pd.Series(apooverlaps).mean(),
        "top_3_overlap": len(top_overlap),
        "top_3_overlaps": top_overlap,
        "overlaps": apooverlaps,
    }




bioemuensdir = "bioemu_8uk6/msmclust/dihedrals"


# In[15]:


bioemuensresults = pd.read_pickle(f"{bioemuensdir}/predictions.pkl")
# In[16]:


holo = "8uk6"
refapo = 'frame_0'
# refaporesults = pd.DataFrame( origapos["results"][refapo] ).T.sort_values("prob", ascending=False)
refapocif = Cif(refapo, f"{bioemuensdir}/predictions/{refapo}/{refapo}.cif")

apooverlaps = []
aposres = []
    
for apo, apod in bioemuensresults.items():
    preds = pd.read_pickle(f"{bioemuensdir}/predictions/{apo}/preds.pkl")
    apores = Cif(apo, f"{bioemuensdir}/predictions/{apo}/{apo}.cif").residues
    apooverlaps.append(apod["overlap"]) # only top1
    for i in range(top):
        aposres.append(
            apores.merge(
                Pocket(f"{bioemuensdir}/predictions/{apo}/{apo}/{apo}_out/pockets/{preds.iloc[i].name}_atm.cif").residues
            )["label_seq_id"]
        )

counts = refapocif.residues.merge(
    pd.concat(aposres).value_counts().to_frame().reset_index(),
    how="right"
)

topres = pd.DataFrame()
for _, g in counts.groupby("count", sort=False):
    topres = pd.concat((
        topres, 
        g.sample(n=min(3-len(topres), len(g)), random_state=0)
    ))
    if len(topres) == 3:
        break
all_topres = counts.loc[lambda x: x["count"].isin(counts.iloc[:topcounts]["count"])]
# topres = counts.loc[lambda x: x["count"].isin(counts.iloc[:topcounts]["count"])].sample(n=3, random_state=0)

print("\t", " ".join(
    topres
    .sort_values("label_seq_id", key=lambda x: x.astype(int))
    ["label_comp_id"]
))

ids = topres["label_seq_id"].astype(int).sort_values()
holo_seqids = ( ids + seqs[holo]["seqmin"] - 1 ).astype(str)

holo_topres = (
    Cif(holo, f"../training_data/7.Extra_set/cifs/{holo}.cif").residues
    .merge(holo_seqids)
    .sort_values("label_seq_id", key=lambda x: x.astype(int))
)
# ids = sorted(holo_topres.label_seq_id.astype(int) - seqs[holo]["seqmin"])

print("\t", "".join(
    pd.DataFrame(list(seqs[holo]["seq"]))
    .iloc[ids - 1]
    [0]
))

top_overlap = (
    extra_holos_sites[holo][0]["site"]
    .merge(holo_topres)
)

guide_res[holo] = {
    "seqids": tuple(ids),# tuple(pd.Series(ids) + 1), # to make them 1-indexed as label_seq_id
    "all_seqids": all_topres["label_seq_id"].astype(int).sort_values().to_list(),
    "mean_overlap": pd.Series(apooverlaps).mean(),
    "top_3_overlap": len(top_overlap),
    "top_3_overlaps": top_overlap,
    "overlaps": apooverlaps,
}


# ## Results

# In[17]:


guide_res_df = pd.DataFrame(guide_res).T


outdir = Path(outdir)



proxdir = outdir / "protenix"
proxdir.mkdir(exist_ok=True)


# In[86]:


for dist in (10.0, 20.0):
    for restraints in (True, False):
        if restraints is False and dist == 10: # only run one norestraints bc dist doesn't affect it
            continue
        for holo, holod in tqdm(seqs.items(), desc='restr ' + str(restraints) + ' dist: ' + str(dist)):
            if holo in ["7yg5", "8v81"]: # not enough CUDA memory
                continue

            name = holo + ("_norestraints" if not restraints else "") + ("_10" if dist == 10 else "")
            path = proxdir / name
            if path.exists():
                continue
            path.mkdir()

            open(path / "non_pairing.a3m", "wb").write(open(f"{outdir}/a3ms/{holo}.a3m", "rb").read().replace(b"\x00", b""))
            
            jsonf = proxdir / f"{name}.json"
            with open(jsonf, "w") as f:
                json.dump([dict(
                    **{
                        "name": name,
                        "sequences": [
                            {"proteinChain": {
                                "sequence": holod["seq"], 
                                "count": 1,
                                "msa": {
                                    "precomputed_msa_dir": str(path),
                                    "pairing_db": "uniref100"
                                }
                            }},
                            {"ligand": {
                                "ligand": holod["smiles"], # can be ccd if i need to download the db anyway
                                "count": 1
                            }}
                        ],
                        # "covalent_bonds": [...]
                    },
                    **({
                        "constraint": {"pocket": {
                            "binder_chain": ["2", 1], # [entity_number, copy_index]
                            "contact_residues": [["1", 1, seqid] for seqid in guide_res[holo]['seqids']], # [entity_number, copy_index, position],
                            "max_distance": int(dist)
                        }}
                    } if restraints else {})
                
                )], f)

            with open(proxdir / f"{name}.log", "w") as f:
                subprocess.run( # PROTENIX_DATA_ROOT_DIR={proxdir}/protenix_data
                    f"TRITON_PTXAS_PATH=/home/fnerin/miniconda3/envs/protenix/bin/ptxas /home/fnerin/miniconda3/envs/protenix/bin/protenix predict --input {jsonf} --out_dir {path} --seeds 0 --sample 5 --model_name protenix_base_default_v0.5.0 --use_msa True --use_default_params True --trimul_kernel cuequivariance --triatt_kernel cuequivariance", 
                    shell = True, check = True, stdout=f, stderr=f
                )