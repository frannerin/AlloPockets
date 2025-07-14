import pandas as pd
import numpy as np


import biotite.structure.io.pdbx
from biotite import structure as biotite_structure
# biotite 1.0.1; other versions may change the shape of the atom_array?


class Biotite_struc:
    def __init__(self, cif):
        self._cif = cif

    @property
    def struc(self):
        return biotite_structure.io.pdbx.get_structure(biotite_structure.io.pdbx.CIFFile.read(self._cif.filename), model=1)

    @property
    def _atom_df(self):
        atom_array = self.struc
        return pd.DataFrame({
            "auth_asym_id": atom_array.chain_id,
            "auth_seq_id": atom_array.res_id,
            "auth_atom_id": atom_array.atom_name,
            "type_symbol": atom_array.element,
            "Cartn_x": atom_array.coord[:, 0],
            "Cartn_y": atom_array.coord[:, 1],
            "Cartn_z": atom_array.coord[:, 2],
            "pdbx_PDB_ins_code": (ic or '?' for ic in atom_array.ins_code)
        }, dtype=str)

    @property
    def _res_df(self):
        return self._atom_df[["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]].drop_duplicates()
        




# from AllosES: https://github.com/ChunhuaLab/AllosES/blob/main/AllosES/utils.py
import math
from threadpoolctl import threadpool_limits
# numpy is old (1.26) due to ProDy requirements; newer might be compatible with numba...

class TransferEntropyF:
    def __init__(self, cif):
        self._cif = cif
        self._biotite = Biotite_struc(self._cif)

    features = [
        "transfer_entropy"
    ]
    
    @staticmethod
    def _distance(coordinate_matrix):
        size, _ = np.shape(coordinate_matrix)
        dis = np.zeros([size, size])
        for i in range(size):
            for j in range(size):
                if j == i:
                    continue
                else:
                    dis[i, j] = math.sqrt((coordinate_matrix[i, 0] - coordinate_matrix[j, 0]) ** 2
                                          + (coordinate_matrix[i, 1] - coordinate_matrix[j, 1]) ** 2
                                          + (coordinate_matrix[i, 2] - coordinate_matrix[j, 2]) ** 2)
        return dis

    
    def _kirchhoff(self, coordinate_matrix, cutoff):
        size, _ = np.shape(coordinate_matrix)
        Kirchhoff_matrix = np.zeros([size, size])
        dis = TransferEntropyF._distance(coordinate_matrix)
        for i in range(size):
            for j in range(size):
                if j == i:
                    continue
                elif j != i:  # Non-diagonal
                    if dis[i, j] <= cutoff:
                        Kirchhoff_matrix[i, j] = -1
                    else:
                        Kirchhoff_matrix[i, j] = 0
            Kirchhoff_matrix[i, i] = -1 * sum(Kirchhoff_matrix[i, :])
        return Kirchhoff_matrix
    
    def _GNM(self, coordinate, N, cutoff):
        Kirchhoff = self._kirchhoff(coordinate, cutoff)
        [Vectors, Values, VectorsT1] = np.linalg.svd(Kirchhoff)
        sorted_indices = np.argsort(Values)
        Values = Values[sorted_indices[:: 1]]
        Vectors = Vectors[:, sorted_indices[:: 1]]
        InvKirchhoff = (Vectors) * (np.linalg.pinv(np.diag(Values))) * (Vectors.T)
        CellAij = {}
        for k in range(0, N):
            if (1 / Values[k]) < 1000:
                CellAij[k] = (Vectors[:, k] * (np.array([Vectors[:, k]]).T) / Values[k])
            else:
                CellAij[k] = np.zeros([N, N])
        return InvKirchhoff, CellAij, N, Values

    def _Transfer_entropy(self, coordinate, N, cutoff, tau):
        with threadpool_limits(limits=2):
            InvKirchhoff, CellAij, N, eig_Value = self._GNM(coordinate, N, cutoff)
            TE = np.ones((N, N), dtype=np.complex_)
            for i in range(N):
                for j in range(N):
                    aEk = [CellAij[k][j][j] for k in range(0, N)]
                    bEk = [CellAij[k][i][j] for k in range(0, N)]
                    cEk = [CellAij[k][j][j] for k in range(0, N)]
                    dEk = [CellAij[k][i][j] for k in range(0, N)]
                    eEk = [CellAij[k][i][i] for k in range(0, N)]
                    aEk = aEk * np.exp(-eig_Value * tau)
                    bEk = bEk * np.exp(-eig_Value * tau)
                    a = np.sum(cEk) ** 2 - np.sum(aEk) ** 2
                    b = (np.sum(eEk) * np.sum(cEk) ** 2)
                    c = 2 * (np.sum(dEk)) * np.sum(aEk) * np.sum(bEk)
                    d = -(((np.sum(bEk) ** 2) + (np.sum(dEk) ** 2)) * (np.sum(cEk))) - ((np.sum(aEk) ** 2) * np.sum(eEk))
                    f = np.sum(cEk)
                    g = (np.sum(eEk) * np.sum(cEk)) - (np.sum(dEk) ** 2)
                    if i == j:
                        TE[i][j] = 0
                    else:
                        TE[i][j] = 0.5 * np.log(a) - 0.5 * np.log(b + c + d) - 0.5 * np.log(f) + 0.5 * np.log(g)
            TE[TE < 0] = 0
            netTE = TE - TE.T
            Difference = np.real((netTE).sum(axis=1))
            norm_difference = Difference / np.max(np.abs(Difference))
            return norm_difference

    def _transfer_entropy(self, cutoff, tau):
        atom_array = self._biotite.struc
        cas = atom_array[ atom_array.atom_name == "CA" ]
        N = len(cas)
        coordinate = cas.coord

        return self._Transfer_entropy(coordinate, N, cutoff, tau)

    def transfer_entropy(self, cutoff = 7, tau = 5):
        te = self._transfer_entropy(cutoff, tau)
        df = self._biotite._res_df
        assert len(te) == len(df)
        df["TE"] = te
        return df





import sys, os

_, pdb, path = sys.argv

sys.path.append("training_data")
from utils.features_utils import calculate_features

if __name__ == '__main__':
    fc = TransferEntropyF
    pdbpath = f"{path}/features/{pdb}"
    os.makedirs(pdbpath, exist_ok=True)
    file = f"{pdbpath}/{fc.__name__}.pkl"
    if not os.path.isfile(file):
        calculated = calculate_features(pdb, fc, file, path, path)
        assert calculated, f"Feature calculation failed: {fc.__name__}"