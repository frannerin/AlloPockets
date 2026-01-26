# AlloPockets

AlloPockets is a machine learning allosteric site prediction tool. Given a protein structure, it detects pockets, computes descriptors, and outputs a ranked table of pockets and three-dimensional visualization (`predict.ipynb`). It was trained using a curated and updated dataset of >3,000 structures of proteins with bound small-molecule allosteric modulators (`database.ipynb`).


## Quickstart

### Setup

Clone the repository and create the conda environment (recommended: [Miniforge](https://conda-forge.org/download/)):

```bash
git clone https://github.com/zoecournia/AlloPockets
cd AlloPockets
CONDA_CHANNEL_PRIORITY=disabled PIP_NO_DEPS=1 conda env create -n allopockets --file conda_env.yaml
conda activate allopockets
```

### Quickstart

```python
from predict import get_cif, predict

pdb = get_cif(pdb_id="6t4k")
# pdb = get_cif(file="your_structure.cif")    # also supports .pdb / .cif / .cif.gz

clean_pdb, predictions = predict(
    pdb,
    protein_chains=["A"],          
    email="you@institution.edu"               # to retrieve a MSA from the ColabFold server
)

predictions                                   # a table ranked by "Allosteric score"
```

**Run this code interactively using the `predict.ipynb` notebook:**

```bash
jupyter lab predict.ipynb
```


## HHBlits for multiple predictions

`predict.ipynb` uses the [ColabFold](https://github.com/sokrypton/ColabFold) server to obtain a Multiple Sequence Alignment to build an HHM file, replacing HHBlits resource-intensive calculations for users. Please respect its usage limits, uphold [ColabFold's MSA server usage limits](https://github.com/sokrypton/ColabFold#:~:text=Is%20it%20okay%20to%20use%20the%20MMseqs2%20MSA%20server%20(cf.run_mmseqs2)%20on%20a%20local%20computer%3F) and [acknowledge the tool](https://github.com/sokrypton/ColabFold#how-do-i-reference-this-work) appropriately. 

To perform **multiple AlloPockets computations**, please switch to the local setup of HHBlits:

- Download and uncompress the [UniRef30 database](https://wwwuser.gwdguser.de/~compbiol/uniclust/2023_02) (e.g., `tar -xzf UniRef30_2023_02_hhsuite.tar.gz`).
- Skip the MSA retrieval, and instead provide the path to the uncompressed database to the `get_features` function with the argument `uniref_path=`


## Repository layout

Entry points:
- `predict.ipynb`: main user-facing notebook for running predictions and visualizing results.
- `predict_advanced.ipynb`: advanced/extended prediction notebook.
- `predict.py`: core Python functions used by the notebooks (prediction pipeline, feature preparation, helpers).
- `database.ipynb`: top-level notebook related to the database (see also `database/` folder below).

Database construction/curation:
- `database/`: notebooks and code for assembling/curating the allosteric site database:
  - `database/data/README.md`: links/instructions for obtaining source datasets.
  - `database/src/`, `database/data/`: supporting code/data folders for database generation.
- `database.db`: SQLite database file tracked in the repo.

Training data generation:
- `training_data/`: notebooks for dataset preparation and feature generation (plus `training_data/utils/`).

Model development/comparisons:
- `models/`: notebooks related to model variants and experiments, plus the deployed predictor artifacts:
  - `models/pockets_physchem_deploy/`: exported model used for prediction (loaded by the code at runtime).
  - `models/other_tools/` and `models/other_tools_apos/`: notebooks and notes for benchmarking other tools.


## Cite

Paper/preprint reference: TBA.

For now, if you use AlloPockets, please acknowledge this repository:

```bibtex
@software{AlloPockets,
  title  = {AlloPockets},
  author = {AlloPockets authors},
  url    = {https://github.com/frannerin/AlloPockets},
  year   = {2026}
}
```

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

**Note:** PyRosetta dependency requires separate licensing for commercial use [www.pyrosetta.org](https://www.pyrosetta.org).

![6T4K](https://github.com/user-attachments/assets/f392b49f-500a-4e38-acd3-fd431f0aaa9d)
