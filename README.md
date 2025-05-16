# AlloPockets

Pocket-based machine learning model for allosteric site prediction.

## Setup

Clone and create conda environment (recommended [Miniforge](https://conda-forge.org/download/)):

```bash
git clone https://github.com/zoecournia/AlloPockets
CONDA_CHANNEL_PRIORITY=disabled PIP_NO_DEPS=1 conda env create -n allopockets --file conda_env.yaml
```

### Extra requirements

Download and save in `training_data/utils/external`:
- [predict_ddG.py script from PyRosetta](https://github.com/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/additional_scripts/predict_ddG.py)
- [DSPP software executable](https://github.com/PDB-REDO/dssp/releases/download/v4.4.0/mkdssp-4.4.0-linux-x64)

Additionally, download and uncompress the [UniRef30 database](https://wwwuser.gwdguser.de/~compbiol/uniclust/2023_02) (`tar -xzf UniRef30_2023_02_hhsuite.tar.gz`).

## Predict

Activate the environment and run the `predict.ipynb` notebook:

```bash
conda activate allopockets
jupyter lab predict.ipynb
```

![4OR2](https://github.com/user-attachments/assets/f88e24bd-cbf1-4c08-a47f-91394057491f)
