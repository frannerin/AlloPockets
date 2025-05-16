from .utils import CifFileWriter

import tempfile
import pandas as pd

from pdbfixer import PDBFixer
from simtk import unit
from Bio.SeqUtils import seq1, seq3, IUPACData
from MDAnalysis.lib.util import inverse_aa_codes

def get_fixed_atoms(f, pdb, orig_atoms, no_alt=True, no_hetatm=True, standard_resnames=True):
    """
    Given a cif and atoms dataframe (and pdb to retrieve some data from the cif), use OpenMM's PDBFixer to add missing atoms (not whole residues) and in general standardize the structure file.
    """
    
    # Setup PDBFixer and and find missing atoms
    fixer = PDBFixer(filename=f)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    # Replace missing residues and terminals with empty dicts to fill only incomplete residues
    fixer.missingResidues = {}
    fixer.missingTerminals = {}
    # Add missing atoms
    fixer.addMissingAtoms()

    # Create a new cif-like dataframe with the new fixed atoms and their positions
    newdf = pd.DataFrame([
        {
            "auth_seq_id": str(a.residue.id),
            "auth_comp_id": a.residue.name,
            "auth_asym_id": a.residue.chain.id,
            "pdbx_PDB_ins_code": a.residue.insertionCode or "?",
            
            "type_symbol": a.element.symbol,
            "label_atom_id": a.name,
            "auth_atom_id": a.name,
            
            "Cartn_x": f"{pos[0]:.4f}",
            "Cartn_y": f"{pos[1]:.4f}",
            "Cartn_z": f"{pos[2]:.4f}",
    
            "group_PDB": "ATOM",
            "label_alt_id": ".",
            "occupancy": f"{1:.2f}",
            "B_iso_or_equiv": f"{0:.2f}",
            "pdbx_formal_charge": "?",
        }
        for a, pos in zip(fixer.topology.atoms(), fixer.positions.value_in_unit(unit.angstrom))
    ], dtype=str)

    # Merge the old and new atom DFs prioritizing existing data
    mergecols = ['type_symbol', 'auth_atom_id',  'auth_seq_id', 'auth_comp_id', 'auth_asym_id', 'pdbx_PDB_ins_code']
    merged = orig_atoms.merge(
        newdf, 
        on=mergecols,
        how="right", suffixes=(None, "_new")
    )

    # Renumber atom ids
    merged.loc[:,"id"] = range(1, len(merged)+1)

    # Transfer the new data for the newly added atoms 
    news = merged[orig_atoms.columns].isna().any(axis=1)
    merged.loc[news, [c for c in newdf.columns if c not in mergecols]] = merged.loc[news, [f"{c}_new" for c in newdf.columns if c not in mergecols]].values

    # Update all positions with the filled-minimized coordinates
    merged.loc[:, ["Cartn_x", "Cartn_y", "Cartn_z"]] = merged.loc[:, ["Cartn_x_new", "Cartn_y_new", "Cartn_z_new"]].values

    # Change all atomtypes to ATOM if flag
    if no_hetatm:
        merged.loc[:,"group_PDB"] = ["ATOM"]*len(merged)

    # Fill the NaN columns of new atoms by duplicating the information of the previously-existing atoms of the same residue
    # For each residue and its atoms
    for g, gres in merged.groupby(['auth_asym_id', 'auth_seq_id', 'pdbx_PDB_ins_code']):
        # If it has any newly-added atom (any NaN)
        if gres.isna().any().any():
            nas = gres.isna().any()
            # Fill the columns with values of the rest of atoms, make sure that it's a unique value and thus residue-dependent and not atom-dependent
            gres.loc[:, nas] = gres.loc[:, nas].ffill()
            try:
                assert len(gres.loc[:, nas].drop_duplicates()) == 1
            except:
                print(f, g, gres.loc[:, nas])
                raise
            merged.loc[gres.index, :] = gres

    # Change residue names to standard ones if flag, using both dictionaries from MDAnalysis and Biopython and from the own PDB modified residues info
    if standard_resnames:
        d = {
            **inverse_aa_codes,
            **IUPACData.protein_letters_3to1_extended
        }
        try:
            orig_map = pdb.cif.data["_pdbx_struct_mod_residue"]
        except:
            orig_map = False
        
        for c in ["label_comp_id", "auth_comp_id"]:
            special = {
                k: seq1(v)
                for k, v in dict(
                    pd.DataFrame(orig_map)
                    .loc[:, [c, "parent_comp_id"]]
                    .values.tolist()
                ).items()
            } if orig_map else {}
            merged.loc[:,c] = [
                seq3(seq1(
                    comp, 
                    custom_map = {
                        **d,
                        **special
                    }
                )).upper()
                for comp in merged[c] 
            ]

    # Make sure there aren't any NaNs
    assert not merged.isna().any().any()

    # Drop merged columns
    final = merged[[c for c in merged.columns if not c.endswith("_new")]]

    # If flag, delete alternative atom positions and only keep the principal/first(A)
    if no_alt:
        final = final.query("label_alt_id in ['.', 'A']")

    return final



def get_fixed_structure(pdb, p, minimal_chains, path, save=False):
    """
    Given a pdb, a minimal structure (p) and a set of minimal_chains, save a fixed .cif of the minimal structure.
    """
    # Get the atoms from the relevant structure, relevant chains, and first model
    orig_atoms = p.atoms[pdb.atoms.columns].query(
        f"label_asym_id in {minimal_chains} and pdbx_PDB_model_num == '1'"
    )

    # Delete any calculated atoms (e.g., new PDBs with hydrogens (7aia))
    if 'calc_flag' in orig_atoms.columns:
        orig_atoms = orig_atoms.query("calc_flag not in ['c', 'calc' 'dum']")

    # Make sure that none of the chains are only CA traces
    assert all(
        len(asym_atoms.label_atom_id.unique()) > 1 
        for asym_id, asym_atoms in orig_atoms.groupby("label_asym_id")
    ), f"{pdb.entry_id}: Some chains are only CA"

    # Write the atoms to a temporary cif for fixing and finally write the fixed .cif
    with tempfile.NamedTemporaryFile("w+", suffix=".cif") as tempf:
        
        tempwriter = CifFileWriter(tempf.name, compress=False)
        tempwriter.write({
            pdb.entry_id.upper(): {"_atom_site": orig_atoms.to_dict("list")}
        })

        fixed_structure = get_fixed_atoms(tempf.name, pdb, orig_atoms)

        if save:
            with open(f"{path}/{pdb.entry_id}.cif", "w+") as f:
                writer = CifFileWriter(f.name, compress=False)
                writer.write({
                    pdb.entry_id.upper(): {
                        "_atom_site": fixed_structure.to_dict(orient="list"),
                        "_entity_poly": pdb.cif.data["_entity_poly"]
                    }
                })
    return fixed_structure