import pandas as pd
from functools import cached_property, cache



from prody_class import ProDyF


# ## Biopython

# In[11]:


from Bio import PDB # conda Biopython 1.84
# mkdssp executable downloaded from https://github.com/PDB-REDO/dssp/releases/tag/v4.4.0
# dssp can be installed in env avoiding compatibility issues with pyroseta with:
## wget https://conda.anaconda.org/conda-forge/linux-64/dssp-4.4.8-h629725b_0.conda
## conda install dssp-4.4.8-h629725b_0.conda
# but gives error
import melodia_py as melodia # from my own fork to keep insertion codes # from pip (melodia-py)
from Bio.SeqUtils import seq1 # for melodia propensity


# In[12]:


class BiopythonF:
    dssp_path = "mkdssp-4.4.0-linux-x64"


    
    def __init__(self, cif):
        self._cif = cif

    features = [
        "exposureCB",
        "exposureCN",
        "residue_depth"
    ]

    
    @property
    def struc(self):
        return PDB.MMCIFParser().get_structure(self._cif.entry_id.upper(), self._cif.filename)

    @property
    def model(self):
        return self.struc[0]


    
    def _get_res_d(self, r): 
        return {
            'auth_asym_id': r.full_id[2], 
            'auth_seq_id': str(r.full_id[3][1]),
            'pdbx_PDB_ins_code': r.full_id[3][2].replace(" ", "") or "?",
        }

    def _get_res_df(self, r): 
        return pd.DataFrame([self._get_res_d(r)], dtype=str)
    


    def _process_property_list(self, property_list, ps):
        return pd.concat([
            pd.concat(
                [
                    self._get_res_df(r),
                    pd.DataFrame([{p: r.xtra[p] for p in ps}])
                ],
                axis=1
            )
            for i in property_list
                for r in [i[0],]
        ])

    def _exposureCB(self):
        return PDB.HSExposure.HSExposureCB(self.model)

    def exposureCB(self):
        return self._process_property_list(
            self._exposureCB().property_list, 
            ['EXP_HSE_B_U', 'EXP_HSE_B_D']
        )

    def _exposureCN(self):
        return PDB.HSExposure.ExposureCN(self.model)

    def exposureCN(self):
        return self._process_property_list(
            self._exposureCN().property_list, 
            ['EXP_CN']
        )



    def _residue_depth(self):
        return PDB.ResidueDepth(self.model)

    def residue_depth(self):
        return self._process_property_list(
            self._residue_depth().property_list, 
            ['EXP_RD', 'EXP_RD_CA']
        )





    # Might fail because some residues are renamed during structure fixing
    def _dssp(self):
        extra = pd.DataFrame(self._cif.origcif.data["_pdbx_poly_seq_scheme"], dtype="str")
        extra = (
            extra.merge(
                self._cif.residues,
                left_on=["asym_id", "entity_id", "seq_id"],
                right_on=["label_asym_id", "label_entity_id", "label_seq_id"]
            )[extra.columns]
            .drop_duplicates()
            .to_dict(orient="list")
        )

        with self._cif._extended_temp_ciff({"_pdbx_poly_seq_scheme": extra}) as f:
            struc = PDB.MMCIFParser().get_structure(self._cif.entry_id.upper(), f.name)
            model = struc[0]
            dssp = PDB.DSSP(model, f.name, dssp=self.dssp_path)
        return dssp

    def dssp(self):
        return pd.DataFrame([
            {
                'auth_asym_id': k[0], 
                'auth_seq_id': str(k[1][1]), 
                'pdbx_PDB_ins_code': k[1][2].replace(" ", "") or "?", 
                **dict(zip(
                    "dssp index, amino acid, secondary structure, relative ASA, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy".split(", "), 
                    v
                ))
            }
            for k, v in self._dssp().property_dict.items()
        ]).drop(["dssp index", "amino acid"], axis=1)



    
    def _melodia(self):
        return pd.DataFrame([
            {
                'auth_asym_id': full_id[2], 
                'auth_seq_id': str(full_id[3][1]),
                'auth_comp_id': r.name,
                'pdbx_PDB_ins_code': full_id[3][2].replace(" ", "") or "?",
                "curvature": r.curvature,
                "torsion": r.torsion,
                "arc_len": r.arc_len,
                "writhing": r.writhing,
                "phi": r.phi,
                "psi": r.psi
            }
            for c in melodia.geometry_dict_from_structure(self.struc).values()
                for r in c.residues.values()
                    for full_id in [r.res_ann["full_id"],]
        ])

    def melodia(self):
        df = self._melodia()
        
        # Add melodia_propensity
        ptable = melodia.PropensityTable()
        df["propensity"] = [
            ptable.get_score(
                target = 'A' if r["auth_comp_id"] != "ALA" else "G", 
                residue = seq1(r["auth_comp_id"]),
                phi = r["phi"], psi = r["psi"]
            )
            for i, r in df.iterrows()
        ]
        
        return df.drop(columns="auth_comp_id")


# ### DSSP

# In[13]:


class DSSPF:
    def __init__(self, cif):
        self._cif = cif
        self.dssp = BiopythonF(self._cif).dssp
        

    features = [
        "dssp"
    ]



import tempfile
import numpy as np
from Bio.Data.PDBData import residue_sasa_scales

class DSSPF:
    def __init__(self, cif):
        self._cif = cif

    def _get_chain_df(self, mmcif):
        # From ChatGPT
        summ = pd.DataFrame(mmcif["_dssp_struct_summary"])
        hb = pd.DataFrame(mmcif["_dssp_struct_bridge_pairs"])
    
        for df in (summ, hb):
            df["label_asym_id"] = df["label_asym_id"].astype(str)
            df["label_seq_id"] = df["label_seq_id"].astype(str)
    
        base = summ[["label_asym_id", "label_seq_id"]].copy()
        base["secondary structure"] = (
            summ["secondary_structure"].astype(str).replace({".": "-", "?": "-", " ": "-"})
        )

        def _num(s: pd.Series, *, dtype: str, default):
            s = s.replace({".": np.nan, "?": np.nan})
            return pd.to_numeric(s, errors="coerce").fillna(default).astype(dtype)
    
        acc_abs = _num(summ["accessibility"], dtype="float64", default=np.nan)
        comp = summ["label_comp_id"].astype(str)
        max_acc = residue_sasa_scales["Sander"]
        rel_asa = (acc_abs / comp.map(max_acc).astype("float64")).clip(upper=1.0)
        base["relative ASA"] = rel_asa.where(comp.map(max_acc).notna(), np.nan).astype("float64")
    
        base["phi"] = _num(summ["phi"], dtype="float64", default=360)
        base["psi"] = _num(summ["psi"], dtype="float64", default=360)
    
        hb["id"] = _num(hb["id"], dtype="int64", default=0)
        id_map = dict(
            zip(
                zip(hb["label_asym_id"].to_numpy(), hb["label_seq_id"].to_numpy()),
                hb["id"].to_numpy(),
            )
        )
    
        def _relidx(prefix: str) -> pd.Series:
            a = hb[f"{prefix}_label_asym_id"].astype(str).to_numpy()
            s = hb[f"{prefix}_label_seq_id"].astype(str).to_numpy()
            partner = np.fromiter(
                (id_map.get((aa, ss), 0) if aa not in {".", "?"} and ss not in {".", "?"} else 0 for aa, ss in zip(a, s)),
                dtype=np.int64,
                count=len(hb),
            )
            return (partner - hb["id"].to_numpy()).astype("int64")
    
        hb_keep = hb[["label_asym_id", "label_seq_id"]].copy()
        hb_keep["NH_O_1_relidx"] = _relidx("acceptor_1")
        hb_keep["NH_O_1_energy"] = _num(hb["acceptor_1_energy"], dtype="float64", default=0.0)
        hb_keep["O_NH_1_relidx"] = _relidx("donor_1")
        hb_keep["O_NH_1_energy"] = _num(hb["donor_1_energy"], dtype="float64", default=0.0)
        hb_keep["NH_O_2_relidx"] = _relidx("acceptor_2")
        hb_keep["NH_O_2_energy"] = _num(hb["acceptor_2_energy"], dtype="float64", default=0.0)
        hb_keep["O_NH_2_relidx"] = _relidx("donor_2")
        hb_keep["O_NH_2_energy"] = _num(hb["donor_2_energy"], dtype="float64", default=0.0)
    
        out = base.merge(hb_keep, on=["label_asym_id", "label_seq_id"], how="left", validate="one_to_one")
    
        out["label_asym_id"] = out["label_asym_id"].astype("object")
        out["label_seq_id"] = out["label_seq_id"].astype("object")
        out["secondary structure"] = out["secondary structure"].astype("object")
    
        for c in ("NH_O_1_relidx", "O_NH_1_relidx", "NH_O_2_relidx", "O_NH_2_relidx"):
            out[c] = _num(out[c], dtype="int64", default=0)
    
        for c in ("relative ASA", "phi", "psi", "NH_O_1_energy", "O_NH_1_energy", "NH_O_2_energy", "O_NH_2_energy"):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
            
        return out    

    def dssp(self):
        chains_dfs = []
        for chain in self._cif.residues.label_asym_id.unique():
            with (
                tempfile.TemporaryDirectory() as tmpdir,
                self._cif._extended_temp_ciff({
                    "_atom_site": self._cif.atoms.query(f"label_asym_id == '{chain}'").to_dict(orient="list")
                }) as f
            ):
                subprocess.run([f"mkdssp", "--calculate-accessibility", f.name, f"{tmpdir}/out.cif"], capture_output=True)
                chains_dfs.append(
                    self._get_chain_df(  self._cif.__class__(self._cif._name, f"{tmpdir}/out.cif").cif.data )
                )
            
        return pd.concat((
            chains_dfs
        ))
        
    features = [
        "dssp"
    ]


# ### Melodia_py

# In[14]:


class MelodiaF:
    def __init__(self, cif):
        self._cif = cif
        self.melodia = BiopythonF(self._cif).melodia
        

    features = [
        "melodia"
    ]


# ## Graphein

# In[15]:


from graphein import protein as graphein # from pip


# In[16]:


class GrapheinF:
    def __init__(self, cif):
        self._cif = cif

    features = [
        "graphein"
    ]

    @cached_property
    def _graph(self):
        extra_config = {
            "verbose": False,
            'node_metadata_functions': [
                graphein.features.nodes.amino_acid.meiler_embedding,  # https://doi.org/10.1007/s008940100038
                graphein.features.nodes.amino_acid.expasy_protein_scale, # AAIndex https://web.expasy.org/protscale/
            ],
        }
        config = graphein.config.ProteinGraphConfig(**extra_config)

        # put biopandas pandaspdb column names and fix types
        df = self._cif.atoms.reset_index().rename(columns={ 
            "index": "line_idx",
            "group_PDB": "record_name",
            "id": "atom_number",
            "label_atom_id": "atom_name",
            "label_alt_id": "alt_loc",
            "auth_comp_id": "residue_name",
            "auth_asym_id": "chain_id",
            "auth_seq_id": "residue_number",
            "pdbx_PDB_ins_code": "insertion",
            'Cartn_x': "x_coord", 
            'Cartn_y': "y_coord",
            'Cartn_z': "z_coord",
            'B_iso_or_equiv': "b_factor",
            "label_asym_id": "segment_id",
            "type_symbol": 'element_symbol',
            'pdbx_formal_charge': 'charge'
        })
        df[['atom_number', 'residue_number', 'line_idx']] = df[['atom_number', 'residue_number', 'line_idx']].astype(int)
        df[['x_coord', 'y_coord', 'z_coord', 'occupancy', 'b_factor']] = df[['x_coord', 'y_coord', 'z_coord', 'occupancy', 'b_factor']].astype(float)
        
        graph = graphein.graphs.construct_graph(
            config=config, path=self._cif.filename,
            df = df
        )
        return graph

    def graphein(self):
        return pd.DataFrame((
            {
                "auth_asym_id": rd["chain_id"],
                "auth_seq_id": str(rd["residue_number"]),
                "pdbx_PDB_ins_code": r.split(":")[-2],
                **rd["meiler"],
                **rd["expasy"]
            }
        for r, rd in dict(self._graph.nodes(data=True)).items()
        ))


# ## FreeSASA

# In[17]:


import subprocess, json


# In[18]:


class FreeSASAF:
    def __init__(self, cif):
        self._cif = cif

    features = [
        "freesasa"
    ]

    def _freesasa(self):
        result = subprocess.run([f"freesasa", "--depth=residue", "--cif", "--format=json", self._cif.filename], capture_output=True)
        return json.loads(result.stdout.decode().strip())

    def freesasa(self):
        resdf = pd.DataFrame(self._cif.residues[["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]])
        resdf["freesasaid"] = [f"{num}{ic.replace('?','')}" for num, ic in resdf[["auth_seq_id", "pdbx_PDB_ins_code"]].values]
        df = pd.DataFrame([
            {
                "auth_asym_id": chain["label"],
                "freesasaid": str(res["number"]),
                **{
                    f"{a}_{t}": tv
                    for a in ["area", "relative-area"]
                        for t, tv in res[a].items()
                }
            }
            for chain in self._freesasa()['results'][0]["structure"][0]["chains"]
                for res in chain["residues"]
        ])
        return resdf.merge(df).drop(columns="freesasaid") 



# ## PyRosetta


from pyrosetta import init, pose_from_pdb, get_score_function; init("-mute core.pack basic core.scoring -ignore_zero_occupancy false")
from utils.external import predict_ddG # script in-folder from pyrosetta tutorial

class PyRosettaF:
    def __init__(self, cif):
        self._cif = cif

    features = [
        "ddG"
    ]

    @cached_property
    def _pose(self):
        return pose_from_pdb(self._cif.filename)
    
    
    def _ddG(self):
        # Create a energy function
        sfxn = get_score_function(True)

        pose = self._pose
        pdbinfo = pose.pdb_info()

        for i, res in enumerate(pose.residues, 1):
            aa1 = res.name1()
            resnum = res.seqpos()
            # Repack and score the native conformation
            mutated_pose = predict_ddG.mutate_residue(
                pose,
                mutant_position=resnum,
                mutant_aa="A" if aa1 != "A" else "G",
                pack_radius=8.0,
                pack_scorefxn=sfxn
            )
            # Score the alanine mutated pose
            score_A1 = sfxn.score(mutated_pose)
            # Repack and score the original conformation
            pose_1 = predict_ddG.mutate_residue(
                pose,
                mutant_position=resnum,
                mutant_aa=aa1,
                pack_radius=8.0,
                pack_scorefxn=sfxn
            )
            score_1 = sfxn.score(pose_1)
            # Compute the ddG of mutation as mutant_score - native_score (final-initial
            ddG = score_A1 - score_1
            
            yield {
                "auth_asym_id": pdbinfo.chain(i),
                "auth_seq_id": str(pdbinfo.number(i)),
                "pdbx_PDB_ins_code": pdbinfo.icode(i).replace(" ", "") or "?",
                "ddG": ddG
            }

    def ddG(self):
        return pd.DataFrame(self._ddG())



# ## HHBlits

# In[23]:


# from moleculekit.tools.hhblitsprofile import getSequenceProfile


# # In[24]:


# class HHBlitsF:

#     uniref_path = "."

    
#     def __init__(self, cif):
#         self._cif = cif

#     features = [
#         "hhblits"
#     ]

#     def _hhblits(self, seq, entity_id, ncpu):
#         return getSequenceProfile(
#             seq,
#             hhblits="hhblits", 
#             hhblitsdb=self.uniref_path, 
#             ncpu=ncpu, niter=4
#         )

#     def hhblits(self, ncpu=10):
#         dfs = []
#         res = self._cif.residues
#         ents = pd.DataFrame(self._cif.cif.data["_entity_poly"], dtype=str)
#         for entity_id, entity_res in res.groupby("label_entity_id"):
#             seq = ents.query(f"entity_id == '{entity_id}'")["pdbx_seq_one_letter_code_can"].item().replace("\n", "")
#             df, _ = self._hhblits(seq, ncpu=ncpu, entity_id=entity_id)
#             for asym_id in entity_res.label_asym_id.unique():
#                 dfs.append(
#                     pd.concat(
#                         [
#                             pd.DataFrame({
#                                 "label_asym_id": [asym_id]*len(seq),
#                                 "label_seq_id": range(1, len(seq)+1)
#                             }, dtype=str),
#                             df.iloc[1:].drop(columns="seq").reset_index(drop=True)
#                         ], 
#                         axis=1
#                     )
#                 )
#         hhblits = pd.concat(dfs)
#         return res.merge(
#             hhblits,
#             on=["label_asym_id", "label_seq_id"]
#         )[hhblits.columns]




## Transfer entropy

# In[25]:


# import biotite.structure.io.pdbx
# from biotite import structure as biotite_structure
# # biotite 1.0.1; other versions may change the shape of the atom_array?


# # In[26]:


# class Biotite_struc:
#     def __init__(self, cif):
#         self._cif = cif

#     @property
#     def struc(self):
#         return biotite_structure.io.pdbx.get_structure(biotite_structure.io.pdbx.CIFFile.read(self._cif.filename), model=1)

#     @property
#     def _atom_df(self):
#         atom_array = self.struc
#         return pd.DataFrame({
#             "auth_asym_id": atom_array.chain_id,
#             "auth_seq_id": atom_array.res_id,
#             "auth_atom_id": atom_array.atom_name,
#             "type_symbol": atom_array.element,
#             "Cartn_x": atom_array.coord[:, 0],
#             "Cartn_y": atom_array.coord[:, 1],
#             "Cartn_z": atom_array.coord[:, 2],
#             "pdbx_PDB_ins_code": (ic or '?' for ic in atom_array.ins_code)
#         }, dtype=str)

#     @property
#     def _res_df(self):
#         return self._atom_df[["auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code"]].drop_duplicates()


# In[27]:


# # from AllosES: https://github.com/ChunhuaLab/AllosES/blob/main/AllosES/utils.py
# import math
# from threadpoolctl import threadpool_limits
# # numpy is old (1.26) due to ProDy requirements; newer might be compatible with numba...


# # In[28]:


# class TransferEntropyF:
#     def __init__(self, cif):
#         self._cif = cif
#         self._biotite = Biotite_struc(self._cif)

#     features = [
#         "transfer_entropy"
#     ]
    
#     @staticmethod
#     def _distance(coordinate_matrix):
#         size, _ = np.shape(coordinate_matrix)
#         dis = np.zeros([size, size])
#         for i in range(size):
#             for j in range(size):
#                 if j == i:
#                     continue
#                 else:
#                     dis[i, j] = math.sqrt((coordinate_matrix[i, 0] - coordinate_matrix[j, 0]) ** 2
#                                           + (coordinate_matrix[i, 1] - coordinate_matrix[j, 1]) ** 2
#                                           + (coordinate_matrix[i, 2] - coordinate_matrix[j, 2]) ** 2)
#         return dis

    
#     def _kirchhoff(self, coordinate_matrix, cutoff):
#         size, _ = np.shape(coordinate_matrix)
#         Kirchhoff_matrix = np.zeros([size, size])
#         dis = TransferEntropyF._distance(coordinate_matrix)
#         for i in range(size):
#             for j in range(size):
#                 if j == i:
#                     continue
#                 elif j != i:  # Non-diagonal
#                     if dis[i, j] <= cutoff:
#                         Kirchhoff_matrix[i, j] = -1
#                     else:
#                         Kirchhoff_matrix[i, j] = 0
#             Kirchhoff_matrix[i, i] = -1 * sum(Kirchhoff_matrix[i, :])
#         return Kirchhoff_matrix
    
#     def _GNM(self, coordinate, N, cutoff):
#         Kirchhoff = self._kirchhoff(coordinate, cutoff)
#         [Vectors, Values, VectorsT1] = np.linalg.svd(Kirchhoff)
#         sorted_indices = np.argsort(Values)
#         Values = Values[sorted_indices[:: 1]]
#         Vectors = Vectors[:, sorted_indices[:: 1]]
#         InvKirchhoff = (Vectors) * (np.linalg.pinv(np.diag(Values))) * (Vectors.T)
#         CellAij = {}
#         for k in range(0, N):
#             if (1 / Values[k]) < 1000:
#                 CellAij[k] = (Vectors[:, k] * (np.array([Vectors[:, k]]).T) / Values[k])
#             else:
#                 CellAij[k] = np.zeros([N, N])
#         return InvKirchhoff, CellAij, N, Values

#     def _Transfer_entropy(self, coordinate, N, cutoff, tau):
#         with threadpool_limits(limits=2):
#             InvKirchhoff, CellAij, N, eig_Value = self._GNM(coordinate, N, cutoff)
#             TE = np.ones((N, N), dtype=np.complex_)
#             for i in range(N):
#                 for j in range(N):
#                     aEk = [CellAij[k][j][j] for k in range(0, N)]
#                     bEk = [CellAij[k][i][j] for k in range(0, N)]
#                     cEk = [CellAij[k][j][j] for k in range(0, N)]
#                     dEk = [CellAij[k][i][j] for k in range(0, N)]
#                     eEk = [CellAij[k][i][i] for k in range(0, N)]
#                     aEk = aEk * np.exp(-eig_Value * tau)
#                     bEk = bEk * np.exp(-eig_Value * tau)
#                     a = np.sum(cEk) ** 2 - np.sum(aEk) ** 2
#                     b = (np.sum(eEk) * np.sum(cEk) ** 2)
#                     c = 2 * (np.sum(dEk)) * np.sum(aEk) * np.sum(bEk)
#                     d = -(((np.sum(bEk) ** 2) + (np.sum(dEk) ** 2)) * (np.sum(cEk))) - ((np.sum(aEk) ** 2) * np.sum(eEk))
#                     f = np.sum(cEk)
#                     g = (np.sum(eEk) * np.sum(cEk)) - (np.sum(dEk) ** 2)
#                     if i == j:
#                         TE[i][j] = 0
#                     else:
#                         TE[i][j] = 0.5 * np.log(a) - 0.5 * np.log(b + c + d) - 0.5 * np.log(f) + 0.5 * np.log(g)
#             TE[TE < 0] = 0
#             netTE = TE - TE.T
#             Difference = np.real((netTE).sum(axis=1))
#             norm_difference = Difference / np.max(np.abs(Difference))
#             return norm_difference

#     def _transfer_entropy(self, cutoff, tau):
#         atom_array = self._biotite.struc
#         cas = atom_array[ atom_array.atom_name == "CA" ]
#         N = len(cas)
#         coordinate = cas.coord

#         return self._Transfer_entropy(coordinate, N, cutoff, tau)

#     def transfer_entropy(self, cutoff = 7, tau = 5):
#         te = self._transfer_entropy(cutoff, tau)
#         df = self._biotite._res_df
#         assert len(te) == len(df)
#         df["TE"] = te
#         return df



FClasses = [
    GrapheinF,
    FreeSASAF,
    DSSPF,
    MelodiaF,
    BiopythonF,    
    PyRosettaF,
    ProDyF,
    # TransferEntropyF,
    # HHBlitsF,
]