# To copy to and run from the step folder where features are being calculated, e.g. 4.Features
# Adapt to define the FClassses custom paths, 'structures_path', 'original_cifs_path' and 'representatives' suitably
# Establish the RAM of the computer, this script was tested on 64GB RAM
# Example: nohup python features.py &> features.log &

memory = 64

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
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import numpy as np
    from utils.utils import Cif

    predict_mem = lambda length: (0.2564017651296701 * np.exp(0.0038526113282513364 * length))*64 / memory # returns % of memory

    queue = sorted(
        (
            (pdb, len(Cif(pdb, f"{structures_path}/{pdb}.cif").residues)) 
            for pdb in representatives
            if not os.path.isfile(f"features/{pdb}/TransferEntropyF.pkl")
        ),
        key=lambda x: x[1]
    )
    

    while len(queue):
        used_mem = 0
        pdbs = []
        
        while used_mem < 80 and len(queue):
            pdb, seqlen = queue.pop(0)
            mem = predict_mem(seqlen)
            if mem > 80:
                print(pdb, "very big", flush=True)
                if len(pdbs) == 0:
                    pdbs.append(pdb)
                    break
            if used_mem + mem > 90 or len(pdbs) >= 30:
                queue.insert(0, (pdb, seqlen))
                break
            else:
                pdbs.append(pdb)
                used_mem += mem

        if len(pdbs) == 0:
            break
    
        with ProcessPoolExecutor(max_workers=len(pdbs)) as executor:
            futures = {}
            
            for pdb in pdbs:
                pdbpath = f"features/{pdb}"

                fc = TransferEntropyF
                file = f"{pdbpath}/{fc.__name__}.pkl"
                if not os.path.isfile(file):
                    print(pdb, flush=True)
                    os.makedirs(pdbpath, exist_ok=True)
                    future = executor.submit(calculate_features, pdb, fc, file, structures_path, original_cifs_path)
                    futures[future] = [pdb, fc.__name__]
                    
            for future in as_completed(futures):
                pdb, fc = futures[future]
                try:
                    calculated = future.result()
                    if calculated is False:
                         with open(f"transferentropy_calculation_errors.txt", "a") as f:
                             f.write(f"{pdb}\t{fc}\n")
                except:
                     with open(f"transferentropy_errors.txt", "a") as f:
                         f.write(f"{pdb}\t{fc}\n")
