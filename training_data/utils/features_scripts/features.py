# To copy to and run from the step folder where features are being calculated, e.g. 4.Features
# Adapt to define the FClassses custom paths, 'structures_path', 'original_cifs_path' and 'representatives' suitably
# Example: nohup python features.py &> features.log &


import sys, os, pickle
sys.path.append("..")

from utils.features_utils import calculate_features, get_pdb_features # featsd, res_cols
from utils.features_classes import * # Each FClass
BiopythonF.dssp_path = "../utils/external/mkdssp-4.4.0-linux-x64"
HHBlitsF.uniref_path = "/data/fnerin/UniRef30_2023_02/UniRef30_2023_02"


structures_path = "../1.Minimal_structures/Minimal_structures"
original_cifs_path = "/home/fnerin/Desktop/allodb_new/src/data"

with open("../3.Pockets/pockets.pkl", "rb") as f:
    representatives = pickle.load(f).pdb.unique().tolist()
    


if __name__ == '__main__':
    
    FClasses = [
        GrapheinF,
        FreeSASAF,
        DSSPF,
        MelodiaF,
        BiopythonF,
        PyRosettaF,
    ]
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = {}
        
        for pdb in representatives:
            print(pdb, flush=True)
            pdbpath = f"features/{pdb}"
            
            for fc in FClasses:
                file = f"{pdbpath}/{fc.__name__}.pkl"
                if not os.path.isfile(file):
                    os.makedirs(pdbpath, exist_ok=True)
                    future = executor.submit(calculate_features, pdb, fc, file, structures_path, original_cifs_path)
                    futures[future] = [pdb, fc.__name__]
                
        for future in as_completed(futures):
            pdb, fc = futures[future]
            try:
                calculated = future.result()
                if calculated is False:
                     with open(f"features_calculation_errors.txt", "a") as f:
                         f.write(f"{pdb}\t{fc}\n")
            except:
                 with open(f"features_errors.txt", "a") as f:
                     f.write(f"{pdb}\t{fc}\n")