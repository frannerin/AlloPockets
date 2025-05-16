"""
Module with accessory functions for the creation of new entries of the database

The main function is 'create_entry', which also uses the functions 'group_site' and 'get_info'. 'combine_sites' is used in the notebooks when new sites are being created

TODO: 'combine_sites' should be used inside create_entry because it shouldn't be the user's responsibility to appropriately combine_sites???
"""

import pandas as pd
pd.DF = pd.DataFrame

from . import *
from . import utils
from .siteutils import EmptySite

from datetime import date


debug = False


def group_site(site, groups, site_dict):
    """
    Function to group a site into an existing dictionary of site groups, according to the calculated similarities to other group members

    site : allodb.Site
    groups : list of dicts
        List with each group's dict with their group members as items (with site_id as keys)
    self_dict : dict
        Dictionary with information/to store information about the site being grouped (e.g., similarity to other sites in the same or other groups)

    TODO: why when site.nonredundant_site is None the site is added directly to a new group?
    Was it because if the site is not in a protein site (in contact with protein chains) it returns None? I think so
    """
    # If there are no groups yet or ???????, start a new group with the site
    if (
        len(groups) == 0
        or site.nonredundant_site is None ####### ???????
    ):
        groups.append({site.id: site_dict({})})
    # Else, calculate the similarities to other groups' members (utils.calculate_common_residues) and use them to group the site (utils.add_to_group)
    else:
        # Establish the residues to measure similarities against and dictionaries to store the values 
        protein_residues = site.nonredundant_site.protein_residues
        self_in_other, other_in_self = {}, {}
        # For each other group and each member inside each group
        for i, group in enumerate(groups):
            for other_id, other_dict in group.items():
                # Measure the residue similarities of self_in_other and other_in_self with utils.calculate_common_residues and add the information to the dictionaries
                other_nonredundant = Site.get(Site.id == other_id).nonredundant_site
                if other_nonredundant is not None:
                    res_of_self_in_other, res_of_other_in_self = utils.calculate_common_residues(protein_residues, other_nonredundant.protein_residues)
                    self_in_other.update({other_id: res_of_self_in_other})
                    other_in_self.update({other_id: res_of_other_in_self})
                    groups[i][other_id]["res_of_self_in_other"].update({site.id: other_in_self[other_id]})
        # Pass everything to utils.add_to_group
        groups = utils.add_to_group(site.id, site_dict(self_in_other), groups, self_in_other, other_in_self)
        
    return groups



def get_info(site):
    """
    Get information from a Site in a standardized dictionary format to save it into the database for quick querying
    """
    # Save the annotated modulators of the site as a list of lists
    # each list can be used to create a single DataFrame (with each modulator molecule) to use for querying, merging...
    modulator_info = [[mod] for mod in pd.DF(site.modulator).to_dict(orient="records")]
    
    # For each modulator molecule, create a dictionary to store all the information
    for i, mod in enumerate(modulator_info):
        # Modulator identifier as it was just retrieved (a list to make a DF)
        modd = {"modulator": mod}
        mod_residues = site.modulator_residues.merge(pd.DF(mod)).dropna(axis=1, how="all")

        # Modulator entity_id and entity information: type (polymer, non-polymer...) and pdbx_description (name in PDB)
        modd["label_entity_id"] = mod_residues.label_entity_id.unique().item()
        modd.update(
            pd.DF(site.pdb.cif.data["_entity"], dtype=str)
            .query(f"id == '{modd['label_entity_id']}'")
            .loc[:, ["type", "pdbx_description"]]
            .squeeze().to_dict()
        )

        # If the modulator (entity) is a polymer, save the type of polymer
        if modd["type"] == "polymer":
            modd["polymer_type"] = (
                pd.DF(site.pdb.cif.data["_entity_poly"], dtype=str)
                .query(f"entity_id == '{modd['label_entity_id']}'")
                .loc[:, "type"]
                .item()
            )
            # If the polymer modulator is a polypeptide (protein) and it has Uniprot annotations, save the Uniprot IDs
            if modd["polymer_type"] == "polypeptide(L)" and (
                "pdbx_sifts_xref_db_name" in mod_residues.columns 
                and "pdbx_sifts_xref_db_acc" in mod_residues.columns # if these are not in the columns it won't have any accessory database annotations
                and (
                    len( 
                        set(mod_residues.pdbx_sifts_xref_db_name.unique()) - set(("?",))
                    ) > 0 
                    and mod_residues.pdbx_sifts_xref_db_name.replace('?', utils.np.nan).dropna().unique().item() == "UNP"
                )
            ):
                modd["Uniprot"] = mod_residues.pdbx_sifts_xref_db_acc.unique().tolist()

            ## TODO: maybe this print is not necessary anymore
            elif modd["polymer_type"] != "polypeptide(L)": print(modd["polymer_type"])

            # Save length of polymer modulator
            modd["length"] = len(mod_residues)

        ## TODO: save any info from non-polymer modulators?
        elif modd["type"] == "non-polymer":
            pass # fields from PDBe chemcomp's cif?
    
        modulator_info[i] = modd

    # Save info about the Site around the annotated modulator(s)
    site_info = []
    # The entity ids of the polymers forming the site
    polymer_ents = pd.DF(site.pdb.cif.data["_entity"], dtype=str).query(f"type == 'polymer'").id.to_list()
    site_polymer_residues = (
        site.residues
        .merge(site.modulator_residues.label_entity_id, how="outer", indicator=True)
        .query(f"_merge == 'left_only'").drop("_merge", axis=1)
        .query(f"label_entity_id in {polymer_ents}")
    )

    # For each entity of the site, create a dictionary to store all the information
    for ent in site_polymer_residues.label_entity_id.unique():
        # Save the entity id
        entd = {"label_entity_id": ent}
        entity_site_residues = site_polymer_residues.query(f"label_entity_id == '{ent}'")
        
        # Save the label_asym_ids of the chains of the entity that participate in the site
        entd["interacting_chains"] = {"label_asym_id": entity_site_residues.label_asym_id.unique().tolist()}
        
        # Save the polymer type of the entity
        entd["polymer_type"] = (
            pd.DF(site.pdb.cif.data["_entity_poly"], dtype=str)
            .query(f"entity_id == '{ent}'")
            .type.item()
        )
        # If the polymer is a polypeptide (protein) and it has Uniprot annotations, save the Uniprot IDs
        if entd["polymer_type"] == "polypeptide(L)" and (
            "pdbx_sifts_xref_db_name" in entity_site_residues.columns
            and "pdbx_sifts_xref_db_acc" in entity_site_residues.columns
            and (
                    len( 
                        set(entity_site_residues.pdbx_sifts_xref_db_name.unique()) - set(("?",))
                    ) > 0 
                    and entity_site_residues.pdbx_sifts_xref_db_name.replace('?', utils.np.nan).dropna().unique().item() == "UNP"
                )
        ):
            entd["Uniprot"] = entity_site_residues.pdbx_sifts_xref_db_acc.unique().tolist()
            
        site_info.append(entd)

    # Return a dictionary to save in the Site.info column
    return {"modulator_info": modulator_info, "interacting_chains_info": site_info}




def combine_sites(db, pdb, old_sites, new_sites, auto_site_grouping, stringent_site_grouping):
    """
    Given a PDB object/database entry and lists of its sites before and after adding a new Site, respectively, check if the newly added site can be combined with an old one, i.e. if there is redundancy, (and do it)

    db : Database-type object from peewee (APSWDatabase)
        Database in which things are being written
    pdb : allodb.PDB
        PDB to which the new Site was added
    old_sites : list
        Frozen list of PDB.sites before adding the new site
        (frozen: transformed from query that is executed every time (PDB.sites) to a static list (list(PDB.sites)))
    new_sites : list
        List of newly created Sites to check for redundancy
    auto_site_grouping : bool
        ###########################################
    stringent_site_grouping : bool
        ###########################################
    """
    # Make a list of all unique modulator IDs of all old and new sites passed
    all_mods = []
    for l in [old_sites, new_sites]:
        all_mods.extend([site.modulator for site in l if site.modulator not in all_mods])

    # In an atomic context (uses the database without modifying anything permanently unless the code inside succeeds), create all of the Sites that would be created when passing all of the unique modulator IDs as annotated modulator residues to create_entry
    # and compare the information with the passed old and new sites: if the combined site creation returns the same number as old+new sites, or a smaller number (means there's redundancy)
    with db.atomic() as combination:
        # Get all sites that would be created when passing all of the unique modulator IDs
        all_sites = create_entry(db, pdb, all_mods, auto_site_grouping, stringent_site_grouping)
        # If they are the same number as the combination of old and newly created, revert the changes (combined site creation) and exit
        if len(all_sites) == (len(old_sites) + len(new_sites)):
            combination.rollback()
            return
        # Else, check for redundancy and process it
        elif len(all_sites) < (len(old_sites) + len(new_sites)):
            # Create a dictionary to store the correspondences between each newly allsite/combined site ID and the old sites it corresponds to
            mapping = {allsite.id: [] for allsite in all_sites}
            # Process each of the old and new sites passed to find its corresponding allsite/combined site ID
            for l in [old_sites, new_sites]:
                for site in l:
                    # For each site, store the ID of all the allsites/combined sites that contain (or any of their related_sites contain) the saved modulator_residues of the site being processed
                    allsites = [
                        allsite.id 
                        for allsite in all_sites 
                            if any(
                                len(site.modulator_residues.merge(pd.DF(mod))) == len(site.modulator_residues)
                                for mod in (
                                    [allsite.modulator]
                                    + [other_site["other_site"] for other_site in allsite.related_sites["equivalent"]]
                                )
                            )
                    ]
                    # sanity check: each site of old and/or new should only map to a single new allsite/combined site
                    assert len(allsites) <= 1, "Modulator of previous site is found in many of the new all_sites"
                    # Store the old/new site in the list of the corresponding allsite/combined site
                    mapping[allsites[0]].append(site)
            # sanity check: all previous sites have been mapped to a new site
            assert all(
                sum([site in allsite for allsite in mapping.values()]) == 1 
                for l in [old_sites, new_sites] for site in l
            )
            # For each of the new allsites/combined sites, copy the site information (i.e., data source) of the old/new sites it corresponds to
            for allsite in all_sites:
                if "source" not in allsite.info:
                    allsite.info["source"] = {}
                for previous_site in mapping[allsite.id]:
                    for db in previous_site.info["source"]:
                        if db not in allsite.info["source"]:
                            allsite.info["source"][db] = []
                        allsite.info["source"][db].extend(previous_site.info["source"][db])
                allsite.save()
            # Delete the previous old/new sites
            Site.delete().where(
                Site.pdb == pdb, 
                Site.id.in_(
                    tuple(site.id for l in [old_sites, new_sites] for site in l)
                )
            ).execute()

            # And return the new list of combined sites
            return all_sites

        # The new allsites/combined sites shouldn't be more than the combination of old and new passed sites, so revert the database changes and raise an Error
        else:
            combination.rollback()
            raise Exception("len(all_sites) > (len(old_sites) + len(new_sites))") 

        



class NoModulator(Exception):
    """Exception to raise if the passed modulator is not found among the PDB residues"""

def create_entry(db, entry_id, dataset_annotated_modulators, auto_site_grouping=True, stringent_site_grouping=True):
    """
    Given a PDB id and the modulators for which Site(s) need to be created, process the information and return a list with the created Site(s)

    1. Create the PDB object corresponding to the entry_id if needed
    2. Process passed modulator(s) and find in the PDB
    3.a If auto_site_grouping, create Sites for ALL the molecules of the same entity(ies) of the passed modulator(s)
        3.a.1 and combine the Sites of molecules that bind close together ############## if they bind on an equivalent site, or raise an Error otherwise?
    3.b If not auto_site_grouping, create Site(s) for the passed modulator molecule(s)
    4. Group sites together according to their site similarity, and pick only a representative from each group to be saved into the PDB

    db : Database-type object from peewee (APSWDatabase)
        Database in which things are being written
    entry_id : str
        PDB id to create and/or create Site(s) into (ideally lowercase)
    dataset_annotated_modulators : list
        List of modulator(s) to create Site(s) for; each element of the list must be a list of dicts or a dict that can be directly used to create a DF with standardized residue ID fields with which to directly query the residues of the PDB object to which the Site(s) is added
    auto_site_grouping : bool
        Whether to automatically create all possible sites for all molecules of the same entity(ies) as the passed modulator to then choose among those the final Site(s) to be saved,
        and also whether to attempt to combine said sites if they have molecules of the passed entity(ies) that bind together ############## on an equivalent site, or raise an Error otherwise?
    stringent_site_grouping : bool
        ###########################################
    """
    ### 1. CREATE PDB
    # Get the cif information, whether it's from an already downloaded file or from online
    cif_text, cif_hash, cif_file = utils.PDBCif._get_cif(
            entry_id, update=False, save=True, db_hash=None
        ).values()

    # Get or create the corresponding PDB, and get its residues' table to work with
    pdb, created = PDB.get_or_create(
        entry_id=entry_id, defaults = {"_cif_hash": cif_hash}
    )
    residues = pdb.residues

    ### 2. PROCESS PASSED MODULATORS

    # For each passed annotation, try to use it to find it among the PDB's residues and store the standardized ids (utils.simplify_residues) on a list, or raise NoModulator if not found
    modulators = []
    for anno in dataset_annotated_modulators:
        mod = residues.merge(pd.DF(anno, dtype=str))
        if len(mod) > 0:
            mod = utils.simplify_residues(mod, residues)
            if mod not in modulators:
                modulators.append(mod)
        elif len(mod) == 0:
            raise NoModulator(f"{entry_id}, {anno}: couldn't retrieve modulator in pdb with passed information")
    if debug: print("modulators", modulators)

    # Make a list of the entities to which the passed modulator(s) belong to
    modulator_entities = list(set(
        entity_id for mod in modulators for entity_id in residues.merge(pd.DF(mod)).label_entity_id.unique()
    ))

    # Create all the Sites, depending on auto_site_grouping
    all_sites = []
    # If auto_site_grouping, generate a Site for each molecule of the same entity(ies) of the passed modulator(s)
    if auto_site_grouping: ##### needed?
        for mod in modulators:
            for _, group in residues.query(
                f"label_entity_id in {list(residues.merge(pd.DF(mod)).label_entity_id.unique())}"
            ).groupby(list(mod.keys())):
                for d in [utils.simplify_residues(group, residues)]:
                    if not any(d == s.modulator for s in all_sites):
                        all_sites.append(
                            Site.create(
                                pdb = entry_id,
                                modulator = d,
                                updated = date.today()
                            )
                        )
    ### 3.b
    # If not, just generate a Site for each of the passed annotation
    else:
        for d in modulators:
            all_sites.append(
                Site.create(
                    pdb = entry_id,
                    modulator = d,
                    updated = date.today()
                )
            )
    if debug: print("all_sites", all_sites)

    ### 3.a.1 COMBINE the Sites of molecules that bind close together according to the passed options
    # If auto_site_grouping, check if any of the molecules for which Sites were generated bind together AND in equivalent sites 
    # (equivalent defined in the way of ~redundancy, utils.calculate_common_residues (of the site's protein residues)...)
    if auto_site_grouping:
        # Function used if stringent_site_grouping to check that two Sites formed by two modulators are equivalent (and therefore can be combined) or not
        group_modulator_sites = lambda site, other_site, all_sites: (
            any(
                (
                    set(individual_site.modulator_residues.label_asym_id.unique())
                    == set(other_site.modulator_residues.label_asym_id.unique())
                ) or (
                    site.nonredundant_site is not None 
                    and other_site.nonredundant_site is not None
                    and any(
                        i > utils.threshold
                        for i in utils.calculate_common_residues( ## take into account what might happen to antibodies here
                            (
                                individual_site.nonredundant_site.protein_residues
                                .query(f"label_entity_id not in {modulator_entities}")
                            ), 
                            (
                                # other_site is always going to be individual site
                                other_site.nonredundant_site.protein_residues
                                .query(f"label_entity_id not in {modulator_entities}")
                            )
                        )
                    )
                )
                for individual_site in all_sites
                if len(site.modulator_residues.merge(individual_site.modulator_residues)) > 0
            )
        )

        # for isite in all_sites
        #     if isite is inside of site?
        #         if isite and osite have the same label_asym_id?
        #         OR isite_nonred and osite_nonred are inside each other?
        # THE SITE HAS TO BE RETRIEVED ITERATING THROUGH ALL_SITES BECAUSE IT MAY HAVE BEEN MERGED ALREADY AND TEHREFORE THE .MODULATOR_RESIDUES IS NEVER GOING TO MATCH EXACTLY
        # i don't understand the labely_asym_id being equal being a condition?

        # Iterate through all the created sites to decide which to keep and which to delete
        processed_sites, to_delete = [], []
        itersites = list(all_sites)
        while len(itersites) > 0:
            site = itersites.pop(0)
            if debug: print("site from itersites", site, site.modulator)
            # Find residues in Site (different than the modulator) that are of the same entity of a passed modulator(s)
            others_residues = (
                site.residues
                .merge(site.modulator_residues, how="outer", indicator=True)
                .query(f"_merge == 'left_only'").drop("_merge", axis=1)
                .query(f"label_entity_id in {modulator_entities}")
            )
            if debug: print("others_residues", others_residues[["label_asym_id", "auth_seq_id"]])
            
            # If there are any, determine to which of the other sites it/they belong(s) to and if they should be combined, depending also on stringent_site_grouping
            if len(others_residues) > 0:
                #sc
                assert all(len(pd.DF(s.modulator)) == 1 for s in itersites), f"{entry_id}: problem with itersites"
                # Make a list of the other sites the found residue(s) belong(s) to, depending also on stringent_site_grouping
                other_sites = [
                    other_site 
                    for other_site in itersites
                        if len(others_residues.merge(other_site.modulator_residues)) > 0
                        and (
                            # include if not stringent_site_grouping
                            (not stringent_site_grouping)
                            # or, if stringent_site_grouping, if the condition group_modulator_sites is True
                            or (stringent_site_grouping and group_modulator_sites(site, other_site, all_sites))
                        )
                ]
                
                # If stringent_site_grouping, check that if there were any others_residues, all of their sites were retrieved in other_sites
                # if not, it means that some molecules of the entities of the modulators bind together but were not grouped together because the group_modulator_sites condition was not met (the different molecules do not bind in "equivalent" sites) and with stringent_site_grouping that should raise an Error
                if stringent_site_grouping:
                    if debug: print("other_sites", other_sites, [(other_site.modulator, other_site.modulator_residues.label_asym_id.unique()) for other_site in other_sites])
                    assert len(
                        pd.concat(
                            [other_site.modulator_residues for other_site in other_sites] + [pd.DF(columns=others_residues.columns)]
                        ).merge(others_residues)
                    ) == len(others_residues), f"Molecules of the annotated modulator(s) bind close together but were not grouped"

                # If after the identification and checks there are other_sites identified, merge them
                if len(other_sites) > 0:
                    # Remove the sites that are going to be merged from the itersites, and add them to the list of obsolete sites to be deleted
                    for s in [site] + other_sites:
                        itersites.remove(s) if s != site else None
                        to_delete.append(s)

                    # Get the standardized residue ids of all the modulators of the sites to be merged to use them for the combined Site creation                        
                    merged_mods = utils.simplify_residues(
                        pd.concat(
                            [site.modulator_residues] 
                            + [other_site.modulator_residues for other_site in other_sites]
                        ),
                        residues
                    )
                    # Create the site and add it to itersites to include it in the comparison against other sites for future possible merges of the newly created site with others
                    site = Site.create(
                        pdb = entry_id,
                        modulator = merged_mods,
                        updated = date.today()
                    )
                    itersites.insert(0, site)
                    
                    if debug: print("new site, merged_mods", site, merged_mods)
                    
            processed_sites.append(site) if site not in processed_sites else None
            if debug: print("itersites, to_delete", itersites, to_delete)

        # Get a final list of (combined) processed sites to proceed with, and delete those that were identified to do so (if any)
        processed_sites = [s for s in processed_sites if s not in to_delete]
        if len(to_delete) > 0:
            Site.delete().where(
                Site.pdb == pdb.entry_id, 
                Site.id.in_(
                    [s.id for s in to_delete]
                )
            ).execute()
    
    ### 4. GROUP SITES
    sites_groups = []
    if not auto_site_grouping:
        processed_sites = all_sites
    if debug: print("processed_sites", processed_sites, [s.modulator for s in processed_sites])
    # For each Site that proceeded, create a dictionary with/to store information about it (if they were in the original passed annotations, similarities to other Sites...)
    for site in processed_sites:
        if debug: print("modulators", modulators, "site.modulator", site.modulator, "merge", [len(site.modulator_residues.merge(pd.DF(modulator))) > 0 for modulator in modulators])
        # To store if any of the modulator residue(s) of the Site were passed in the original dataset_annotated_modulators information
        in_anno = True if any(len(site.modulator_residues.merge(pd.DF(modulator))) > 0 for modulator in modulators) else False

        # helper function to get site dictionary to store information, e.g. similarities
        site_dict = lambda common: {
            "mods": site.modulator, "in_anno": in_anno, #"in_ass": in_ass,
            "not_in_other": True, "res_of_self_in_other": common
        }
        if debug: print("site_dict", site_dict({}))
        # Function group_site: to group a site into an existing dictionary of site groups, according to the calculated similarities to other group members
        sites_groups = group_site(site, sites_groups, site_dict)                        
        if debug: print("sites_groups after grouping", sites_groups)

    # Choose representative site for each group (ideally, in_anno, not contained in other, and with the 'lowest' _asym_ids (and the 'lowest' assembly, if any))
    for group in sites_groups:
        # Variable to store the final Site.id to keep as representative of the group
        site_to_keep = None
        # Dictionary with the group filtered to keep only Sites that are "in_anno", if any; if there isn't any nothing is returned and all of the Sites are deleted
        in_anno = {s: s_d for s, s_d in group.items() if s_d["in_anno"]}
        if len(in_anno) > 0:
            # Further filter the group to keep only Sites that are not already 'contained' in other groups (see utils.add_to_group)
            to_keep = {s: s_d for s, s_d in group.items() if s_d["not_in_other"]} or group

            ### Last filter: sort to get the Site of the group with the 'lowest' _asym_ids (tuple of (label_asym_id, auth_asym_id)) 
            ### AND, if the parent PDB has an assembly, also the 'lowest' assembly #/id in which the modulator is present: (assembly_id, label_asym_id, auth_asym_id)
            # Define the asyms function to retrieve the (label_asym_id, auth_asym_id) tuple and set it as sortkey to use to sort sites, or proceed with ass_asyms
            asyms = lambda s: sorted(pdb.residues.merge(pd.DF(s[-1]["mods"]))[["label_asym_id", "auth_asym_id"]].drop_duplicates().values.tolist())
            sortkey = asyms
            if pdb.assembly is not None:
                ass_list = pdb._assembly_asyms
                if ass_list is not None:
                    # If the parent PDB has (an) assembly(ies), add to the output of the sortkey function 'asyms' (creating ass_asyms) the lowest assembly_id in which a modulator of the Site is present: (assembly_id, label_asym_id, auth_asym_id)
                    ass_asyms = lambda s: [
                        next(
                            ass["assembly_id"] 
                            for ass in ass_list.to_dict(orient="records") 
                            if any(
                                m[0] in ass["asym_id_list"].split(',') for m in asyms(s)
                            )
                        )
                    ] + asyms(s)
                else:
                    ass_asyms = asyms
                sortkey = ass_asyms
            if debug: print("sorted to_keep", to_keep, sorted(to_keep.items(), key=sortkey))
            # Finally, select the representative Site.id of the group
            site_id_to_keep = sorted(
                to_keep.items(),
                key=sortkey
            )[0][0]

            # Get the Site to keep using the selected Site.id and complete its attributes/information: in_anno, related_sites (if Site is proteic (nonredundant_site is not None)), and info
            site_to_keep = Site.get(Site.pdb == pdb.entry_id, Site.id == site_id_to_keep)
            site_to_keep.in_anno = group[site_to_keep.id]["in_anno"]
            if site_to_keep.nonredundant_site is not None:
                # Save the .related_sites attribute of the Site with the information of other molecules of the modulator that are present in the structure and that form equivalent or non-equivalent sites, and their calculated similarities
                site_to_keep.related_sites = {
                    "equivalent": [
                        {
                            "other_site": s["mods"], 
                            "res_of_other_in_site": s["res_of_self_in_other"][site_to_keep.id] if site_to_keep.id in s["res_of_self_in_other"] else None, 
                            "res_of_site_in_other": group[site_to_keep.id]["res_of_self_in_other"][k] if k in group[site_to_keep.id]["res_of_self_in_other"] else None
                        }
                        for k, s in group.items() if k != site_to_keep.id
                    ],
                    "nonequivalent": [
                        {
                            "other_site": s["mods"], 
                            "res_of_other_in_site": s["res_of_self_in_other"][site_to_keep.id] if site_to_keep.id in s["res_of_self_in_other"] else None, 
                            "res_of_site_in_other": group[site_to_keep.id]["res_of_self_in_other"][k] if k in group[site_to_keep.id]["res_of_self_in_other"] else None
                        }
                        for g in sites_groups if g != group
                            for k, s in g.items() if k != site_to_keep.id # this last if here should be redundant
                    ],
                }
            site_to_keep.info = get_info(site_to_keep)
            site_to_keep.save()
        # Delete the rest of sites of the groups that were not selected, or all of them if no site_to_keep was found
        Site.delete().where(
            Site.pdb == pdb.entry_id, 
            Site.id.in_(
                tuple(k for k in group.keys() if (site_to_keep is None or k != site_to_keep.id))
            )
        ).execute()

    # Return the created Sites, if any
    return Site.select().where(
            Site.pdb == pdb.entry_id, 
            Site.id.in_(
                tuple(k for group in sites_groups for k in group.keys())
            )
        )