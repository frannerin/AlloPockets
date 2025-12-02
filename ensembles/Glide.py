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


# ## ProtPrep

# In[14]:


# ## Grid
#     grid_in.write_text(contents)

def build_grid(
    grid_inp: Path,
    grid_zip: Path,
    receptor_mae: Path,
    center: Tuple[float, float, float],
): # -> Path:
    inner = 20#float,
    outer = 30#float
    x, y, z = center
    contents = f"""FORCEFIELD   OPLS_2005
GRID_CENTER   {x:.8f}, {y:.8f}, {z:.8f}
GRIDFILE   {grid_zip.absolute()}
INNERBOX   {inner}, {inner}, {inner}
OUTERBOX   {outer}, {outer}, {outer}
RECEP_FILE   {receptor_mae.absolute()}
"""
    grid_inp.write_text(contents)
    glide = which("glide")
    run([glide, "-WAIT", str(grid_inp.absolute())], cwd=grid_zip.parent)
    return
    # grid_zip = grid_inp.with_suffix(".zip")
    # if not grid_zip.exists():
    #     raise RuntimeError(f"Grid not created: {grid_zip}")
    # return grid_zip





# # Guided

dockings = pd.read_pickle(Path("glide") / "glide-guided_dockings.pkl")
dockings


# In[80]:


glidedir = Path("glide_bigger")
glidedir.mkdir(exist_ok=True)


for holo, holod in dockings.items():
    for guide in ("pocket", "guide"):
        path = glidedir / f"{holo}_{guide}"
        path.mkdir(exist_ok=True)
    
        if (path / f"xglide/{holo}_topcomplexes.maegz").exists():
            continue
        
        grids = path / "grids"
        grids.mkdir(exist_ok=True)
        xout = path / "xglide"
        xout.mkdir(exist_ok=True)

        gridsl = ""
        for apo, apod in holod.items():
            apopath = path / apo
            apopath.mkdir(exist_ok=True)
                
            grid_zip = grids / f"{apo}_grid.zip"
            if not grid_zip.exists():
                build_grid(
                    grid_inp = grids / f"{apo}_grid.inp",
                    grid_zip = grid_zip,
                    receptor_mae = glidedir.parent / "glide" / holo / "preps" / (f"{apo}_prep.mae" if (holo != "8uk6" or holo == apo) else f"{apo}_updated_prep.mae"),
                    center = apod[f"{guide}_cog"],
                )
            gridsl += f"GRID	{grid_zip.absolute()}\n"

        xglide_inp = xout / f"{holo}.inp"    
        xglide_inp.write_text(f"""
{gridsl}
ALIGN	FALSE
LIGAND	{(glidedir.parent / "glide" / holo / f"{holo}.smi").absolute()}
GRIDGEN_GRID_CENTER	AUTO
GRIDGEN_INNERBOX	10
GRIDGEN_OUTERBOX	26.0
PPREP FALSE
LIGPREP	TRUE
LIGPREP_EPIK	TRUE
DOCK_PRECISION	SP
DOCK_WRITE_XP_DESC	FALSE
DOCK_POSE_OUTTYPE	poseviewer
DOCK_POSES_PER_LIG   5
NATIVEONLY	FALSE
DOCK_LIG_VSCALE	0.80
SKIP_DOCKING	FALSE
GOOD_RMSD	2.0
GENERATE_TOP_COMPLEXES	5
        """)
    
        run(
            " ".join((
                SCHRO + "/run xglide.py",
                xglide_inp.name,
                "-WAIT -HOST localhost:32 -TMPLAUNCHDIR"
            )),
            cwd=xout, 
            shell=True
        )

        counts = {}
        with sstruct.StructureReader(xout / f"{holo}_topcomplexes.maegz") as r:
            for i, st in enumerate(r, 1):
                apo = (
                    st.property.get("s_i_glide_gridfile", "grid")
                    .rsplit("_", 1)[0]
                    .replace("_updated", "").replace("updated", "")
                )# '8pfp_grid' or frame_0_grid
                rank = counts.get(apo, 0) + 1
                if rank == 6: continue
                fname = path / apo / f"{apo}_rank{rank}.pdb"
                with sstruct.StructureWriter(str(fname)) as w: w.append(st)
                counts[apo] = rank