from .utils import Cif

import pickle, time, os
from Bio.SeqUtils import IUPACData
import pandas as pd
import numpy as np


def calculate_features(pdb, fc, file, structures_path, original_cifs_path):
    """
    Given a PDB ID and a feature class from the features_classes module, calculate the features from the fc and save them in the passed file path ('.pkl' file).
    """
    name = fc.__name__
    if not os.path.isfile(file):
        # Establish a Cif of the PDB ID and with the passed paths
        Cif.path = structures_path
        Cif.original_cifs_path = original_cifs_path
        cif = Cif(pdb)

        # Establish the feature class for feature calculation
        setattr(cif, name, fc(cif))
        feats = fc.features

        # Calculate each feature and save if successful; if there's an error there will be missing feature and it will be handled later
        results = []
        for feat in feats:
            try:
                start = time.time()
                result = getattr(getattr(cif, name), feat)()
                results.append({
                    "pdb": pdb,
                    "feature": feat,
                    "result": result,
                    "time": time.time() - start
                })
            except:
                break

        # If not all features, return False to be handled by the function calling calculate_features
        if all(f in (d["feature"] for d in results) for f in feats):
            with open(file, "wb") as f:
                pickle.dump(results, f)
            return True
        else:
            return False



# Dictionary with the names of the feature classes, the names of their features, and the expected columns for each feature
featsd = {
    'GrapheinF': {'graphein': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'dim_1', 'dim_2', 'dim_3', 'dim_4', 'dim_5', 'dim_6', 'dim_7', 'pka_cooh_alpha', 'pka_nh3', 'pka_rgroup', 'isoelectric_points', 'molecularweight', 'numbercodons', 'bulkiness', 'polarityzimmerman', 'polaritygrantham', 'refractivity', 'recognitionfactors', 'hphob_eisenberg', 'hphob_sweet', 'hphob_woods', 'hphob_doolittle', 'hphob_manavalan', 'hphob_leo', 'hphob_black', 'hphob_breese', 'hphob_fauchere', 'hphob_guy', 'hphob_janin', 'hphob_miyazawa', 'hphob_argos', 'hphob_roseman', 'hphob_tanford', 'hphob_wolfenden', 'hphob_welling', 'hphob_wilson', 'hphob_parker', 'hphob_ph3_4', 'hphob_ph7_5', 'hphob_mobility', 'hplchfba', 'hplctfa', 'transmembranetendency', 'hplc2_1', 'hplc7_4', 'buriedresidues', 'accessibleresidues', 'hphob_chothia', 'hphob_rose', 'ratioside', 'averageburied', 'averageflexibility', 'alpha_helixfasman', 'beta_sheetfasman', 'beta_turnfasman', 'alpha_helixroux', 'beta_sheetroux', 'beta_turnroux', 'coilroux', 'alpha_helixlevitt', 'beta_sheetlevitt', 'beta_turnlevitt', 'totalbeta_strand', 'antiparallelbeta_strand', 'parallelbeta_strand', 'a_a_composition', 'a_a_swiss_prot', 'relativemutability']}, 
    'FreeSASAF': {'freesasa': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'area_total', 'area_polar', 'area_apolar', 'area_main-chain', 'area_side-chain', 'relative-area_total', 'relative-area_polar', 'relative-area_apolar', 'relative-area_main-chain']}, # 'relative-area_side-chain' is not used as it's always NaN for glycines
    'DSSPF': {'dssp': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'secondary structure', 'relative ASA', 'phi', 'psi', 'NH_O_1_relidx', 'NH_O_1_energy', 'O_NH_1_relidx', 'O_NH_1_energy', 'NH_O_2_relidx', 'NH_O_2_energy', 'O_NH_2_relidx', 'O_NH_2_energy']}, 
    'MelodiaF': {'melodia': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'curvature', 'torsion', 'arc_len', 'writhing', 'phi', 'psi', 'propensity']}, 
    'BiopythonF': {'exposureCB': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'EXP_HSE_B_U', 'EXP_HSE_B_D'], 
                   'exposureCN': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'EXP_CN'], 
                   'residue_depth': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'EXP_RD', 'EXP_RD_CA']}, 
    'PyRosettaF': {'ddG': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'ddG']}, 
    'ProDyF': {'prs': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'prs_effectiveness', 'prs_sensitivity'], 
               'mechstiff': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'mechstiff'], 
               'rmsf': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'rmsf'],
               'essa': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'essa']},
    'TransferEntropyF': {'transfer_entropy': ['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code', 'TE']}, 
    'HHBlitsF': {'hhblits': ['label_asym_id', 'label_seq_id', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'M->M', 'M->I', 'M->D', 'I->M', 'I->I', 'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D']}
}

# Minimal columns used to identify individual residues of proteins
res_cols = ['label_entity_id', 'label_asym_id', 'label_seq_id', 'auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code']





def get_labels(residues, sites):
    """
    Given a pd.DataFrame of residues and a list of pd.DataFrames of residues of sites, return the labelled residues of allosteric sites
    """
    sites_res = pd.concat(sites).drop_duplicates()
    labels = pd.Series([0]*len(residues), index=residues.index.tolist(), name="label")
    labels.iloc[
        residues.merge(sites_res, how="left", indicator=True)
        .query("_merge == 'both'")
        .index
    ] = 1
    
    return labels




def encode_column(col, values):
    """
    Given a column (pd.Series) with categorical values, one-hot encode them and make sure that all possible 'values' columns exist
    """
    names = {v: f"{col.name}_{v}" for v in values}
    return (
        pd.get_dummies(col, dtype=int)
        .rename(columns=names)
        .reindex(columns=names.values(), fill_value=0)
    )

def get_pdb_features(cif, sites, features_path):
    """
    Process the features (.pkl files) calculated for a PDB/Cif structure into a pd.DataFrame containing all of them, as well as labels for the residues
    """
    pdb = cif.entry_id

    # Check that all 3-letter residue codes are among the 20 standard
    assert all(res.lower().capitalize() in IUPACData.protein_letters_3to1 for res in cif.residues["label_comp_id"].unique()), (pdb, "Unrecognized amino acid 3-letter name:", set( res.lower().capitalize() for res in cif.residues["label_comp_id"].unique() ) - set(IUPACData.protein_letters_3to1))

    # DataFrame of residues with only the minimal residue ID columns
    residues = cif.residues[res_cols]

    # Labels of residues given a list of sites
    labels = get_labels(residues, sites)

    # Starting DataFrame of features with the residue ID columns, labels, and one-hot encoded 1-letter code amino acids ('label_comp_id')
    # The DataFrame will have MultiIndex columns, with an upper level for each of the 3 elements mentioned: Residues, Labels, Amino acids
    features = pd.concat(
        (
            residues,
            labels,
            encode_column(
                cif.residues["label_comp_id"].apply(lambda x: IUPACData.protein_letters_3to1[x.lower().capitalize()]),
                ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'V', 'W', 'Y', "S", 'X']
            ),
        ),
        axis=1,
        keys = ["Residues", "Label", "Amino acids"]
    )

    # Process each featureclass and its features and add them to the features DataFrame
    for fcname in featsd.keys():        
        merged = residues
        for feat in pd.read_pickle(f"{features_path}/features/{pdb}/{fcname}.pkl"):
            fname = feat["feature"]
            result = feat["result"]

            # Drop FreeSASA's relative-area_side-chain because glycines are always going to be NaNs
            if fname == "freesasa":
                result = result.drop(columns='relative-area_side-chain')

            # Check that all feature columns exist and that the feature result has all residues of the structure
            if len(result.columns) != len(featsd[fcname][fname]):
                print((pdb, fcname, fname), "Not all feature columns", set(featsd[fcname][fname]) - set(result.columns))
            if len(result) != len(residues):
                print((pdb, fcname, fname), "Different # of residues:", len(result), f"(original {len(residues)})")

            # For DSSP, fix 'relative ASA' NaNs and type it as numeric, and one-hot encode 'secondary structure'
            if fname == "dssp":
                result = pd.concat(
                    (
                        result.drop(columns=["relative ASA", "secondary structure"]),
                        result["relative ASA"].replace("NA", np.nan).astype(float),
                        encode_column(result["secondary structure"], ["H", "B", "E", "G", "I", "T", "S", "-"])
                    ),
                    axis=1
                )
            # For HHBlits, keep only the minimal residue ID columns and transform feature columns to float
            elif fname == "hhblits":
                result = pd.concat(
                    (
                        result[[c for c in result.columns if c in residues]],
                        result.drop(columns=[c for c in result.columns if c in residues]).astype(float)
                    ),
                    axis=1
                )            

            # Check that all corrected and encoded feature columns are numeric, if there are any NaNs and/or all-NaN rows
            if not result[[c for c in result.columns if c not in residues.columns]].dtypes.apply(lambda c: c.kind in 'if').all():
                print((pdb, fcname, fname), "Not all columns are numeric:", result[[c for c in merged.columns if c not in residues.columns]].dtypes)
            if result[[c for c in result.columns if c not in residues.columns]].isna().all(axis=1).any():
                print((pdb, fcname, fname), "Some all-NaN rows")
            if result.isna().any().any():
                print((pdb, fcname, fname), "NaNs:", result.columns[result.isna().any()].tolist())

            
            # Merge the final features DataFrame with the residues DataFrame, and check that all residues are in it
            merged = merged.merge(
                result,
                on=[c for c in merged.columns if c in residues.columns and c in result],
                how="left"
            )
        
            if len(merged) != len(residues):
                print((pdb, fcname, fname), "Different # of residues after merging feature:", len(merged), f"(original {len(residues)})")

        # Include the processed DataFrame into the features DataFrame, assigning a top-level MultiIndex to designate the featureclass where the feature columns come from
        features = features.merge(
            (
                merged
                .set_axis(
                    pd.MultiIndex.from_tuples([
                        ("Residues" if c in residues.columns else fcname[:-1], c) for c in merged.columns
                        
                    ]), 
                    axis=1
                )
            ),
            on=features[["Residues"]].columns.tolist(),
            how="left"
        )
        
        # Check that all residues remain after merging
        if len(result) != len(residues):
            print((pdb, fcname, fname), "Different # of residues after merging feature class:", len(features), f"(original {len(residues)})")

    # Final check that all expected columns are present (173)
    if len(features.columns) != 173:
        print(pdb, "Different # of feature columns:", len(features.columns), "(original 173)")
    
    return features
