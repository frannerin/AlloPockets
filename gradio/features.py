import sys, os, pickle

sys.path.append("training_data")

from utils.features_utils import calculate_features
from features_classes import *
# BiopythonF.dssp_path = "../training_data/utils/external/mkdssp-4.4.0-linux-x64"
BiopythonF.dssp_path = "training_data/utils/external/mkdssp-4.4.0-linux-x64" 
os.chmod(BiopythonF.dssp_path, 0o755)


_, pdb, path = sys.argv


if __name__ == '__main__':    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=len(FClasses)) as executor:
        futures = {}
        pdbpath = f"{path}/features/{pdb}"
        os.makedirs(pdbpath, exist_ok=True)
        for fc in FClasses:
            file = f"{pdbpath}/{fc.__name__}.pkl"
            if not os.path.isfile(file):
                future = executor.submit(calculate_features, pdb, fc, file, path, path)
                futures[future] = [pdb, fc.__name__]
                
        for future in as_completed(futures):
            pdb_id, fc = futures[future]
            calculated = future.result()
            assert calculated, f"Feature calculation failed: {fc}"