# AlloPockets

Pocket-based machine learning model for allosteric site prediction.

## Setup

Clone and create conda environment (recommended [Miniforge](https://conda-forge.org/download/)):

```bash
git clone https://github.com/zoecournia/AlloPockets
CONDA_CHANNEL_PRIORITY=disabled PIP_NO_DEPS=1 conda env create -n allopockets --file conda_env.yaml
```

### Additional requirements

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

## HHBlits for multiple predictions

`predict.ipynb` uses the [ColabFold](https://github.com/sokrypton/ColabFold) server to obtain a Multiple Sequence Alignment to build an HHM file, replacing HHBlits resource-intensive calculations for users. Please respect its usage, uphold [ColabFold's MSA server usage limits](https://github.com/sokrypton/ColabFold#:~:text=Is%20it%20okay%20to%20use%20the%20MMseqs2%20MSA%20server%20(cf.run_mmseqs2)%20on%20a%20local%20computer%3F) and [acknowledge the tool](https://github.com/sokrypton/ColabFold#how-do-i-reference-this-work) appropriately. 

To perform multiple AlloPockets computations, please switch to the local setup of HHBlits:

- Download and uncompress the [UniRef30 database](https://wwwuser.gwdguser.de/~compbiol/uniclust/2023_02) (`tar -xzf UniRef30_2023_02_hhsuite.tar.gz`).
- Skip the MSA retrieval, and instead provide the path to the uncompressed database to the `get_features` function with the argument `uniref_path=`

![6T4K](https://github.com/user-attachments/assets/f392b49f-500a-4e38-acd3-fd431f0aaa9d)