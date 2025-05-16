import os, sys
# Include the path where the "src" of allodb is; w.r.t. the location of this file, it is 2 folders behind
# '__file__' is used as "../../" doesn't work because it uses the location of the script/notebook that imports this module, and not the location of this module
sys.path.append(
    os.path.dirname( os.path.abspath(__file__) )
    .rsplit("/", 2)[0]
)

from src.viz import Viz


from .utils import Cif
from .features_utils import res_cols as orig_res_cols
res_cols = list(orig_res_cols)
res_cols.pop(res_cols.index("label_entity_id"))

import pandas as pd



class Pocket(Cif):
    """
    Class to manage the .cif file of an FPocket-identified pocket. Inherits from utils.Cif and includes extra methods to extract the pocket features saved in the .cif and related files.
    """

    
    def __init__(self, filename):
        self._name = filename.split("/")[-1].replace(".cif", "")
        self.filename = filename


    @property
    def cif_feats(self):
        """
        Extract pocket features stored in the cif in '_struct.pdbx_descriptor'. The plain-text formatting of the information requires very specific processing.
        """
        feats = {
            " ".join(l[:-3]): float(l[-2])
            for feat in (
                " ".join(
                    self.cif.data["_struct"]["pdbx_descriptor"] + ['15']
                )
                .replace(" 10 -Pocket volume (convex hull)", " 10 - Pocket volume (convex hull)")
                .split(" - ")[1:]
            )
            for l in [feat.split(" "),]
        }
        # Assert that all .cif features are extracted correctly
        assert set(feats.keys()) == {'Pocket Score', 'Drug Score', 'Number of alpha spheres', 'Mean alpha-sphere radius', 'Mean alpha-sphere Solvent Acc.', 'Mean B-factor of pocket residues', 'Hydrophobicity Score', 'Polarity Score', 'Amino Acid based volume Score', 'Pocket volume (Monte Carlo)', 'Pocket volume (convex hull)', 'Charge Score', 'Local hydrophobic density Score', 'Number of apolar alpha sphere', 'Proportion of apolar alpha sphere'}
        # Assert that all features are properly typed as numeric
        assert all(type(v) in (float, int) for v in feats.values())
        return feats

    @property
    def info_feats(self):
        """
        Extract pocket features stored in the FPocket output file of the whole PDB, by pocket number.
        """
        basepath, path = self.filename.rsplit("/", 3)[:2]
        with open(f"{basepath}/{path}/{path.replace('_out', '_info.txt')}", "r") as f:
            feats = next(
                {
                    line.split(':')[0].strip(): (
                        float(line.split(':')[1].strip())
                    )
                    for line in pocket.strip().split("\n")[1:]
                }
                for pocket in f.read().split("Pocket")
                if pocket.strip().startswith(f'{self._name.replace("pocket", "").replace("_atm", "")} :')
            )
        # Assert that all .cif features are extracted correctly
        assert set(feats.keys()) == {'Score', 'Druggability Score', 'Number of Alpha Spheres', 'Total SASA', 'Polar SASA', 'Apolar SASA', 'Volume', 'Mean local hydrophobic density', 'Mean alpha sphere radius', 'Mean alp. sph. solvent access', 'Apolar alpha sphere proportion', 'Hydrophobicity score', 'Volume score', 'Polarity score', 'Charge score', 'Proportion of polar atoms', 'Alpha sphere density', 'Cent. of mass - Alpha Sphere max dist', 'Flexibility'}
        # Assert that all features are properly typed as numeric
        assert all(type(v) in (float, int) for v in feats.values())
        return feats

    @property
    def feats(self):
        """
        Combine features extracted from the pocket .cif and the whole-PDB FPocket results file. Some features are common between the two sources (although they have different names, mapped with the 'common' dictionary) and there are cif-exclusive and results-file-exclusive features.
        """
        # name in cif: name in info
        common = {'Pocket Score': 'Score', 'Drug Score': 'Druggability Score', 'Number of alpha spheres': 'Number of Alpha Spheres', 'Mean alpha-sphere radius': 'Mean alpha sphere radius', 'Mean alpha-sphere Solvent Acc.': 'Mean alp. sph. solvent access', 'Hydrophobicity Score': 'Hydrophobicity score', 'Polarity Score': 'Polarity score', 'Amino Acid based volume Score': 'Volume score', 'Pocket volume (Monte Carlo)': 'Volume', 'Charge Score': 'Charge score', 'Local hydrophobic density Score': 'Mean local hydrophobic density', 'Proportion of apolar alpha sphere': 'Apolar alpha sphere proportion'}
        cif_exclusive = ['Mean B-factor of pocket residues', 'Pocket volume (convex hull)', 'Number of apolar alpha sphere']
        info_exclusive = ['Total SASA', 'Polar SASA', 'Apolar SASA', 'Proportion of polar atoms', 'Alpha sphere density', 'Cent. of mass - Alpha Sphere max dist', 'Flexibility']

        return {
            **{
                k: v
                for k, v in self.cif_feats.items()
                if (k in common or k in cif_exclusive)
            },
            **{
                k: self.info_feats[k]
                for k in info_exclusive
            }
        }





def get_pocket_view(
    pdb, 
    cif,
    pocket,
    pockets_path,
    sites, # list of dicts with "site" (and "mod")
    elements_colors=(("site", "black"), ("mod", "purple")),
    pocket_color="purple"
):
    """
    Returns the viewer, and the colors to apply to it, to visualize a pocket and the sites of the structure.
    """
    pocket = Pocket(
        f"{pockets_path}/{pdb}/{pdb}_out/pockets/{pocket}_atm.cif"
    )

    # Colors of the pocket residues to pass to Viz's .color method from ipymolstar
    colors = [
        {
            'struct_asym_id': res["label_asym_id"], 
            "residue_number": int(res["label_seq_id"]),
            'representation': 'spacefill',
            'representationColor': pocket_color,
            "color": "white",
            'focus': True
        }
        for i, res in pocket.residues.iterrows()
    ]

    # Colors of the sites (and modulators, if available) to pass to Viz's color_residues method that processes it for ipymolstar
    sites_colors = [
        {
            "res": s[elem], 
            "color": color
        }
        for s in sites
            for elem, color in elements_colors
                if elem in s
    ]
    
    return Viz(cif), colors, sites_colors

## Cell 1:
# v, colors, sites_colors = view_pocket("11bg", "pocket1")
# v
## Cell 2:
# v.color(colors, keep_colors=True, keep_representations=True)
# v.color_residues(sites_colors)




    
def get_pockets_info(cif, sites, pockets_path): # f"{pockets_path}/{pdb}/{pdb}_out/pockets"
    """
    For all the pockets of the PDB/Cif passed, return a dictionary for each pocket with its number of residues and overlaps:
    - % of residues of a site (the site that results in the max %) in the pocket
    - % of residues of the pocket in the site (the site that results in the max %)
    - The maximum %, and the mean %, of the previous two
    """
    info = []
    pdb = cif.entry_id
    
    pdbpath = f"{pockets_path}/{pdb}/{pdb}_out/pockets"
    for pocketf in os.listdir(pdbpath):
        if pocketf.endswith(".cif"):
            pocket = Pocket(f"{pdbpath}/{pocketf}")
            name = pocketf.split("_")[0]
            pocket_res = pocket.residues

            # Percentage of residues of "one" in "other"
            get_overlap = lambda one, other: (
                len(
                    one
                    .merge(
                        other,
                        on=["label_asym_id", "label_seq_id"],
                        )
                ) / len(one)
            )

            perc_of_site_in_pocket = max(get_overlap(s["site"], pocket_res) for s in sites)
            perc_of_pocket_in_site = max(get_overlap(pocket_res, s["site"]) for s in sites)

            info.append({
                "pdb": pdb,
                "pocket": name,
                "nres": len(pocket_res),
                "site_in_pocket": perc_of_site_in_pocket,
                "pocket_in_site": perc_of_pocket_in_site,
                # "max_overlap": max((perc_of_site_in_pocket, perc_of_pocket_in_site)),
                # "mean_overlap": pd.Series((perc_of_site_in_pocket, perc_of_pocket_in_site)).mean(),
            })
            
    return tuple(info)




def get_mean_pocket_features(pdb, pocket, pdb_features, pockets_path): # f"{pockets_path}/{pdb}/{pdb}_out/pockets/{pocket}_atm.cif"
    """
    Given a PDB ID, a pocket and a pd.DataFrame of features of that PDB, return the average of the features of the residues of the pocket.
    """
    return pdb_features.merge((
        Pocket(f"{pockets_path}/{pdb}/{pdb}_out/pockets/{pocket}_atm.cif").residues[res_cols]
        .set_axis(
            pd.MultiIndex.from_tuples([ ("Residues", c) for c in res_cols ]), 
            axis=1
        )
    )).drop(columns=["Residues", "Label"], level=0).mean()
    