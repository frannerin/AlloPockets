#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import os, re, tempfile
from pathlib import Path
import subprocess as sp
import pandas as pd
# import argparse
# import gzip
# import os
# import shutil

# import sys
# import tempfile

from typing import List, Tuple, Optional
# 

from schrodinger import structure as sstruct

SCHRO = "/opt/schrodinger2025-2"


# In[2]:


outdir = Path("dldocking")
outdir.mkdir(exist_ok=True)


# ## Data

# In[3]:


extra_holos_featuresd = pd.read_pickle("../training_data/7.Extra_set/features.pkl")
extra_holos_featuresd


# In[4]:


extra_holos_sites = {k: v for k, v in pd.read_pickle("../training_data/7.Extra_set/news_sites.pkl").items() if k in extra_holos_featuresd}
extra_holos_sites


# In[5]:


apos = pd.read_pickle(f"pdb_ensemble/apos.pkl")
overlaps = pd.read_pickle(f"pdb_ensemble/predictions.pkl")
origapos = pd.read_pickle("../models/other_tools_apos/models_lenient_labelling.pkl")["model5"]

origholos = pd.read_pickle("../models/other_tools/models.pkl")["model5"]


# #### 8jp0 - 8sgi

# In[6]:


overlaps["8jp0"].pop("8sgi")


# In[7]:


seqsf = f"cofolding/seqs.pkl"

seqs = pd.read_pickle(seqsf)


# In[8]:


seqsdf = pd.DataFrame({k: {k1: v1 for k1, v1 in v.items() if k1 != "site"} for k, v in seqs.items()}).T
seqsdf


# In[9]:


bioemuensdir = Path("bioemu_8uk6/msmclust/dihedrals")


# In[10]:


bioemuensresults = pd.read_pickle(bioemuensdir / "predictions.pkl")
bioemuensresults


# # Consensus residues

# In[11]:


guide_resf = outdir / "guide_res.pkl"

if guide_resf.exists():
    guide_res = pd.read_pickle(guide_resf)
else:
    guide_res = {}

guide_res#### 8uk6bioemuensdir = "bioemu_8uk6/msmclust/dihedrals"bioemuensresults = pd.read_pickle(f"{bioemuensdir}/predictions.pkl")

guide_res_df = pd.DataFrame(guide_res).T
guide_res_df


# # Functions

# ## General

# In[13]:


def run(cmd: List[str], cwd: Optional[str] = None, **kwargs) -> None:
    print(f"[CMD] {cmd if 'shell' in kwargs else ' '.join(cmd)}")
    try:
        p = sp.run(cmd, check=True, cwd=cwd, text=True, capture_output=True, **kwargs)
        print(p.stdout, p.stderr)
    except sp.CalledProcessError as e:
        print("[CMD FAILED]", e.returncode, e.cmd)
        print("[STDOUT]\n", e.stdout)
        print("[STDERR]\n", e.stderr)
        raise
    return


def which(tool: str) -> str:
    """Return full path to a Schrödinger CLI tool."""
    p = Path(SCHRO) / tool
    if p.exists():
        return str(p)
    # Some tools live at the root (e.g., $SCHRODINGER/prepwizard), others under subdirs.
    for sub in ["", "utilities", "shape_screen", "glide", "sitemap", "ligprep", "epik", "vsw"]:
        pp = Path(SCHRO) / sub / tool
        if pp.exists():
            return str(pp)
    return tool  # fallback on PATH


def ensure_outdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {d}")


# # Guided

dockings = pd.read_pickle(Path("glide") / "glide-guided_dockings.pkl")
dockings



glidedir = Path("glide_ifd")
glidedir.mkdir(exist_ok=True)
(glidedir / "ligpreps").mkdir(exist_ok=True)




def prepare_ligands(smiles: Path, holo: str, outmae: Path, cwd: Path): # ligprep_inp: Path, out_mae: Path,
    # -> Path: #  pH: float = 7.4, dpH: float = 1.0
    """
    LigPrep with Epik states; outputs a single MAEGZ of prepared ligands.
    Accepts SDF/MAE/SMILES (SMILES must be in a .smi or via -ismi).
    """
    
    ligprep_inp = cwd / f'{holo}.inp'
    ligprep_inp.write_text(f"""
INPUT_FILE_NAME   {smiles.absolute()}
MAX_ATOMS   500
FORCE_FIELD   16
EPIK   yes
EPIKX   no
EPIK_METAL_BINDING   no
INCLUDE_ORIGINAL_STATE   no
DETERMINE_CHIRALITIES   no
IGNORE_CHIRALITIES   yes
NUM_STEREOISOMERS   32
OUT_MAE   {outmae}
    """)

    # Detect SMILES input by extension
    # is_smiles = ligand_in.suffix.lower() in [".smi", ".smiles", ".can"] # has to be .smi
    ligprep = which("ligprep")
    cmd = [ligprep, "-WAIT", "-inp", str(ligprep_inp.absolute())]
    # Epik-like states around target pH
    # cmd += ["-ph", f"{pH}", "-pht", f"{dpH}", "-s", "1"]  # 1 conformer per state (adjust as needed)
    run(cmd, cwd=cwd)
    return





inp_str = lambda strucmae, ligprep: f"""
#  Multiple input structures can be specified by adding additional
#  INPUT_FILE lines or including multiple structures in a single
#  file.
#
#  If beginning with an existing Pose Viewer file, simply specify
#  it as the INPUT_FILE (making sure the name ends in "_pv.mae"
#  or "_pv.maegz") and ensure that the first GLIDE_DOCKING stage
#  is commented out.  The ligand used in producing the Pose Viewer
#  file must also be provided to the second GLIDE_DOCKING stage,
#  using the LIGAND_FILE keyword.

INPUT_FILE	{strucmae}

# Prime Loop Prediction
#  Perform a loop prediction on the specified loop, including
#  side chains within the given distance.  Only return
#  structures within the specified energy range from the
#  lowest energy prediction, up to the maximum number of
#  conformations given.
#
#  Note: This stage is disabled by default.  Uncomment the
#   lines below and edit the fields appropriately to enable it.
#STAGE PRIME_LOOP
#  START_RESIDUE A:11
#  END_RESIDUE A:16
#  RES_SPHERE 7.5
#  MAX_ENERGY_GAP 30.0
#  MAX_STRUCTURES 5
#  USE_MEMBRANE no

STAGE GLIDE_DOCKING2
  BINDING_SITE ligand Z:899
  INNERBOX 15.0
  OUTERBOX 30.0
  LIGAND_FILE  {ligprep}
  LIGANDS_TO_DOCK all
  GRIDGEN_RECEP_CCUT  0.25
  GRIDGEN_RECEP_VSCALE 0.50
  GRIDGEN_FORCEFIELD OPLS_2005
  DOCKING_PRECISION SP
  DOCKING_LIG_CCUT  0.15
  DOCKING_CV_CUTOFF  100.0
  DOCKING_LIG_VSCALE 0.50
  DOCKING_POSES_PER_LIG 20
  DOCKING_FORCEFIELD OPLS_2005
  DOCKING_RINGCONFCUT 2.5
  DOCKING_AMIDE_MODE penal

STAGE COMPILE_RESIDUE_LIST
  DISTANCE_CUTOFF	5.0

STAGE PRIME_REFINEMENT
  NUMBER_OF_PASSES	1
  USE_MEMBRANE no
  OPLS_VERSION OPLS_2005

STAGE SORT_AND_FILTER
  POSE_FILTER	r_psp_Prime_Energy
  POSE_KEEP	30.0

STAGE SORT_AND_FILTER
  POSE_FILTER	r_psp_Prime_Energy
  POSE_KEEP	20#

STAGE GLIDE_DOCKING2
  BINDING_SITE ligand Z:999
  INNERBOX 10.0
  OUTERBOX auto
  LIGAND_FILE  {ligprep}
  LIGANDS_TO_DOCK self
  GRIDGEN_RECEP_CCUT  0.25
  GRIDGEN_RECEP_VSCALE 1.00
  GRIDGEN_FORCEFIELD OPLS_2005
  DOCKING_PRECISION SP
  DOCKING_LIG_CCUT  0.15
  DOCKING_CV_CUTOFF  0.0
  DOCKING_LIG_VSCALE 0.80
  DOCKING_POSES_PER_LIG 1
  DOCKING_FORCEFIELD OPLS_2005
  DOCKING_RINGCONFCUT 2.5
  DOCKING_AMIDE_MODE penal

STAGE SCORING
  SCORE_NAME  r_psp_IFDScore
  TERM 1.0,r_i_glide_gscore,0
  TERM 0.05,r_psp_Prime_Energy,1
  REPORT_FILE report.csv
"""






from schrodinger.structure import StructureReader, StructureWriter


for holo, holod in dockings.items():        
    outmae = (glidedir / "ligpreps" / f'{holo}.mae').absolute()
    if not outmae.exists():
        prepare_ligands(
            smiles = (glidedir.parent / "glide" / holo / f"{holo}.smi").absolute(),
            holo = holo,
            outmae = outmae,
            cwd = glidedir / "ligpreps"
        )
    
    for guide in ("pocket", "guide"):
        path = glidedir / f"{holo}_{guide}"
        path.mkdir(exist_ok=True)

        for apo, apod in holod.items():
            apopath = path / apo
            apopath.mkdir(exist_ok=True)
            if (apopath / f"{apo}-out.maegz").exists():
                continue

            cog_coords = apod[f"{guide}_cog"]
            strucmae = apopath / f"{apo}_center.mae"
            if holo == "8uk6" and apo != "8uk6":
                st = next(StructureReader((glidedir.parent / "glide" / holo / "preps" / f"{apo}_updated_prep.mae")))
            else:
                st = next(StructureReader((glidedir.parent / "glide" / holo / "preps" / f"{apo}_prep.mae")))
            a = st.addAtom("Du", *cog_coords)
            a.pdbres = "DUM"
            a.resnum = 899
            a.chain = "Z"
            StructureWriter(strucmae).append(st)
            StructureWriter(str(strucmae).replace(".mae", ".pdb")).append(st)
    
            ifd_inp = apopath / f"{apo}.inp"
            ifd_inp.write_text(
                inp_str(strucmae.name, outmae)
            )
        
            run(
                " ".join((
                    SCHRO + "/ifd",
                    ifd_inp.name,
                    "-NGLIDECPU 24 -NPRIMECPU 24 -NOLOCAL -HOST localhost -SUBHOST localhost -TMPLAUNCHDIR -WAIT"
                )),
                cwd=apopath, 
                shell=True
            )
            
            for rank, st in enumerate(
                sorted(
                    StructureReader(apopath / f"{apo}-out.maegz"), 
                    key=lambda st: st.property.get("r_psp_IFDScore")
                )[:5],
                1
            ):
                fname = apopath / f"{apo}_rank{rank}.pdb"
                with sstruct.StructureWriter(str(fname)) as w: w.append(st)