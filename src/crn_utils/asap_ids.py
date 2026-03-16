import os
import sys
import pandas as pd
import json
import logging
import shutil
from pathlib import Path

from .util import load_tables

from .constants import *

repo_root = Path(__file__).resolve().parents[1]
wf_common_path = repo_root.parent / "wf-common" / "util"
sys.path.insert(0, str(wf_common_path))

# from wf-common
from common import strip_team_prefix

__all__ = [
    # Generic ID mapping functions used by all sources
    "normalize_source_for_ids",
    "export_all_id_mappers",
    "load_all_id_mappers",
    "update_meta_tables_with_asap_ids",
    "update_all_id_mappers",
    "load_id_mapper",
    "write_id_mapper",
    "generate_asap_dataset_id",
    "generate_asap_team_id",
    "get_sampr",
    "get_id",
    
    # PMDBS specific functions
    "export_pmdbs_id_mappers",
    "update_pmdbs_id_mappers",
    "generate_human_subject_ids",
    "generate_human_sample_ids",
    
    # MOUSE specific functions
    "export_mouse_id_mappers",
    "update_mouse_id_mappers",
    "generate_mouse_subject_ids",
    "generate_mouse_sample_ids",
    
    # INVITRO specific functions
    "export_cell_id_mappers",
    "update_cell_id_mappers",
    "generate_cell_ids",
    "generate_cell_sample_ids",
    
    # Used by crn-utils/release_util.py only
    "load_pmdbs_id_mappers",
    "update_pmdbs_meta_tables_with_asap_ids",
    "update_mouse_meta_tables_with_asap_ids",
    "load_mouse_id_mappers",
    "load_cell_id_mappers",
    "update_cell_meta_tables_with_asap_ids",
]


# The following functions are for generic ID utilities for the Dec2025 refactor
# to support a unified prep_release_metadata() worfkflow.
# ID generation and mapping functionality currently wraps existing source-specific
# functions and leaves these intact for backwards compatibility.
# !!!NOTE!!:
# FEB2026 release used CDE v4.1, which has SUBJECT instead of CELL and MOUSE, 
# meaning subject_id must be uniform. Further, this release included 
# schapira-fecal-metagenome-human-baseline and liddle-human-colon-spatial-cosmx*,
# the first non-PMDBS human datasets.
# For the FEB2026 release release we are using the PMDBS ID mappers (exceptions_handle_as_pmdbs)
# but future PRs will:
# 1) Replace the single-source calls with species/source/assay from a universal look up
# 2) Implement an ID system that best captures non-PMDBS human samples
# ----

#TODO: Here functions normalize_source_for_ids and export_all_id_mappers
#      work together to normalize source_for_ids.
#      It's STILL one of ["pmdbs", "mouse", "invitro"].
#      but now it's based on CDE ValidCategories (organism and sample_source).
#      This is a temporary hack for the Feb2026/March2026 releases which use PMDBS/MOUSE/CELL ASAP IDs
#      A fututre implementation will fully transition to general SUBJECT ASAP IDs.
def normalize_source_for_ids(
        organism: str, 
        source: str) -> str:
        """
        Normalize source for the purposes of ID mapping, which is currently based on PMDBS/MOUSE/CELL but will be updated in the future.

        Required fields:
        - organism: organism type of the dataset (e.g., "Human", "Mouse")
        - source: source type of the dataset (e.g., "Brain", "Fecal", "Cell lines", "iPSC")
        Note: organism and source must be values valid in the CDE ValidCategories tab
        """

        if source in ["Cell lines", "EPSC", "iPSC"]:
            source_for_ids = "invitro"
        elif organism == "Human":
            source_for_ids = "pmdbs"
        elif organism == "Mouse":
            source_for_ids = "mouse"
        else:
            print(f"organism: {organism}")
            print(f"source: {source}")
            raise ValueError(f"ERROR!!! normalize_source_for_ids: Couldn't determine which ID mappers to use.")
        return source_for_ids


def export_all_id_mappers(
    map_path: Path,
    source_for_ids: str,
    id_mappers: dict[str, dict],
    suffix: str = "ids"  # Default to "ids" suffix for the master file names, like ASAP_PMDBS_subj_ids.json
    ) -> None:
    """
    This function wraps source-specific export calls to give a unified interface.

    Required fields:
        - map_path: path to the ID mappers
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future
        - id_mappers: dict of ID mappers

        Note: organism and source must be values valid in the CDE ValidCategories tab
    
    Returns:
        - None

    """
    map_path = Path(map_path)
    
    if source_for_ids == "pmdbs":
        export_pmdbs_id_mappers(
            map_path=map_path,
            suffix=suffix,
            datasetid_mapper=id_mappers["dataset"],
            subjectid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"],
            gp2id_mapper=id_mappers["gp2"],
            sourceid_mapper=id_mappers["source_subject"]
        )
    elif source_for_ids == "mouse":
        export_mouse_id_mappers(
            map_path=map_path,
            suffix=suffix,
            datasetid_mapper=id_mappers["dataset"],
            mouseid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"]
        )
    elif source_for_ids == "invitro":
        export_cell_id_mappers(
            map_path=map_path,
            suffix=suffix,
            datasetid_mapper=id_mappers["dataset"],
            cellid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"]
        )
    else:
        raise ValueError(f"Unknown source: {source_for_ids}")


def load_all_id_mappers(map_path: Path, 
                        source_for_ids: str
                        ) -> dict[str, dict]:
    """
    Load the ID mappers for the given source type.

    Requires:
        - map_path: path to the directory containing the ID mapper files
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future

    Returns a dict with standardized keys:
        - "dataset": dataset ID mapper (always present)
        - "subject": subject/mouse/cell ID mapper (always present)
        - "sample": sample ID mapper (always present)
        - "gp2": GP2 ID mapper (PMDBS only, optional)
        - "source_subject": source subject ID mapper (PMDBS only, optional)

    """
    map_path = Path(map_path)
    id_mappers = {}

    # Dataset is common to all sources
    dataset_mapper_path = map_path / "ASAP_dataset_ids.json"
    id_mappers["dataset"] = load_id_mapper(dataset_mapper_path)
    
    if source_for_ids == "pmdbs":
        id_mappers["subject"] = load_id_mapper(map_path / "ASAP_PMDBS_subj_ids.json")
        id_mappers["sample"] = load_id_mapper(map_path / "ASAP_PMDBS_samp_ids.json")
        id_mappers["gp2"] = load_id_mapper(map_path / "ASAP_PMDBS_gp2_ids.json")
        id_mappers["source_subject"] = load_id_mapper(map_path / "ASAP_PMDBS_sourcesubj_ids.json")
    
    elif source_for_ids == "mouse":
        id_mappers["subject"] = load_id_mapper(map_path / "ASAP_MOUSE_ids.json")
        id_mappers["sample"] = load_id_mapper(map_path / "ASAP_MOUSE_samp_ids.json")
    
    elif source_for_ids == "invitro":
        id_mappers["subject"] = load_id_mapper(map_path / "ASAP_INVITRO_ids.json")
        id_mappers["sample"] = load_id_mapper(map_path / "ASAP_INVITRO_samp_ids.json")
    
    else:
        raise ValueError(f"Unknown source: {source_for_ids}.")

    return id_mappers


def update_meta_tables_with_asap_ids(
    meta_tables: dict[str, pd.DataFrame],
    dataset_id: str,
    id_mappers: dict[str, dict],
    asap_ids_schema: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Inject ASAP IDs into the metadata tables based on the provided ID mappers.
    Assumes that the ID mappers have already been populated for the given dataset.

    Required fields:
        - meta_tables: dict of metadata tables to update, keyed by table name
        - dataset_id (e.g., "team-smith-pmdbs-sn-rnaseq")
        - id_mappers: dict of ID mappers
        - asap_ids_schema: schema defining which tables need which ASAP IDs

    Returns:
        - a dict of the updated metadata tables with ASAP IDs injected.

    """

    if dataset_id.startswith("team-"):
        team = dataset_id.split("-")[1]
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    # Getting the individual mappers
    asap_dataset_id = id_mappers["dataset"].get(dataset_name)
    subject_mapper = id_mappers["subject"]
    sample_mapper = id_mappers["sample"]
    
    # Determine which tables need which IDs
    sample_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_sample_id"
    ]["Table"].to_list()
    
    subject_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_subject_id"
    ]["Table"].to_list()
    
    # Inject IDs into tables
    for table_name in meta_tables.keys():
        meta_table = meta_tables[table_name]
        
        # Inject subject IDs
        if table_name in subject_id_tables:
            if "subject_id" not in meta_table.columns:
                raise ValueError(
                    f"Table '{table_name}' is missing required 'subject_id' column. "
                    f"Please ensure SUBJECT table uses 'subject_id' (not 'mouse_id' or 'cell_id')."
                )
            
            # Map subject_id -> ASAP_subject_id
            asap_subject_ids = meta_table["subject_id"].map(subject_mapper)
            meta_table.insert(0, "ASAP_subject_id", asap_subject_ids)
        
        # Inject sample IDs
        if table_name in sample_id_tables and "sample_id" in meta_table.columns:
            asap_sample_ids = meta_table["sample_id"].map(sample_mapper)
            meta_table.insert(0, "ASAP_sample_id", asap_sample_ids)
        
        # Insert dataset and team IDs at the front of each table
        meta_table.insert(0, "ASAP_dataset_id", asap_dataset_id)
        meta_table.insert(0, "ASAP_team_id", f"TEAM_{team.upper()}")

    return meta_tables
  

def update_all_id_mappers(
    dataset_id: str,
    organism: str,
    source: str,
    metadata_dir: Path,
    map_path: Path,
    dry_run: bool = False
) -> dict[str, dict]:
    """
    This is a source-agnostic orchestrator that wraps source-specific calls to
    update a dataset's ID mappers and export them.
    
    On a dry run no files are written, otherwise the updated mappers are saved 
    but a backup of existing files is made first. The updated mappers are always
    returned as a dict.

    Required fields:
        - dataset_id: (e.g., "team-smith-pmdbs-sn-rnaseq")
        - organism: organism type of the dataset (e.g., "Human", "Mouse")
        - source: source type of the dataset (e.g., "Brain", "Fecal", "Cell lines", "iPSC")
        - metadata_dir: directory containing the metadata tables
        - map_path: path to the ID mappers
        - dry_run: if True, no files are written, only a dry run is performed
        Note: organism and source must be values valid in the CDE ValidCategories tab

    Returns:
        - updated ID mappers as a dict of dicts

    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    metadata_dir = Path(metadata_dir)
    map_path = Path(map_path)
    
    source_for_ids = normalize_source_for_ids(organism, source)

    logging.info(f"Updating ID mappers for dataset: {dataset_name} of source_for_ids: {source_for_ids}")
    
    # Load existing ID mappers which will be updated
    id_mappers = load_all_id_mappers(map_path=map_path, source_for_ids=source_for_ids)
    
    # TODO: Implement getting expected_tables based on CDE OSA rules
    # Each source updates SAMPLE and SUBJECT, with PMDBS having further IDs
    expected_tables = ["SAMPLE", "SUBJECT"]
    if source_for_ids == "pmdbs":
        expected_tables.extend(["CLINPATH"])

    meta_tables = load_tables(metadata_dir, expected_tables)
    
    for table_name in expected_tables:
        if table_name not in meta_tables:
            raise ValueError(
                f"Required metadata table '{table_name}' not found in {metadata_dir}"
            )
    
    # Call source-specific update functions, which return tuples of updated mappers
    if source_for_ids == "pmdbs":
        (
            id_mappers["dataset"],
            id_mappers["subject"],
            id_mappers["sample"],
            id_mappers["gp2"],
            id_mappers["source_subject"],
        ) = update_pmdbs_id_mappers(
            clinpath_df=meta_tables["CLINPATH"],
            sample_df=meta_tables["SAMPLE"],
            dataset_id=dataset_id,
            source_for_ids=source_for_ids,
            datasetid_mapper=id_mappers["dataset"],
            subjectid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"],
            gp2id_mapper=id_mappers["gp2"],
            sourceid_mapper=id_mappers["source_subject"],
        )
    elif source_for_ids == "mouse":
        (
            id_mappers["dataset"],
            id_mappers["subject"],
            id_mappers["sample"],
        ) = update_mouse_id_mappers(
            subject_df=meta_tables["SUBJECT"],
            sample_df=meta_tables["SAMPLE"],
            dataset_id=dataset_id,
            source_for_ids=source_for_ids,
            datasetid_mapper=id_mappers["dataset"],
            mouseid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"],
        )
    elif source_for_ids == "invitro":
        (
            id_mappers["dataset"],
            id_mappers["subject"],
            id_mappers["sample"],
        ) = update_cell_id_mappers(
            cell_df=meta_tables["SUBJECT"],
            sample_df=meta_tables["SAMPLE"],
            dataset_id=dataset_id,
            source_for_ids=source_for_ids,
            datasetid_mapper=id_mappers["dataset"],
            cellid_mapper=id_mappers["subject"],
            sampleid_mapper=id_mappers["sample"],
        )
    else:
        raise ValueError(f"Unknown source_for_ids: {source_for_ids}")
    
    # Export (note that even if dry_run is True, we still return the updated
    # mappers; if dry_run is False, we write to disk but a backup is made first)
    if not dry_run:
        logging.info(f"Writing updated ID mappers to: {map_path}")
        export_all_id_mappers(map_path=map_path, source_for_ids=source_for_ids, id_mappers=id_mappers)
    else:
        logging.info(f"Dry run: would write updated ID mappers to: {map_path}")
        
    return id_mappers


def load_id_mapper(
        id_mapper_path: Path
        ) -> dict:
    """
    Load the id mapper from the json file

    Required fields:
        - id_mapper_path: path to the id mapper json file

    Returns:
        - id_mapper: dict of existing ID mappings, or empty dict if file not found
    """

    id_mapper_path = Path(id_mapper_path)
    if os.path.exists(id_mapper_path):
        with open(id_mapper_path, "r") as f:
            id_mapper = json.load(f)
        print(f"id_mapper loaded from {id_mapper_path}")
    else:
        id_mapper = {}
        print(f"id_mapper not found at {id_mapper_path}")
    return id_mapper


def write_id_mapper(
        id_mapper: dict, 
        id_mapper_path: str | Path):
    """
    Write the id mapper to the json file

    Required fields:
        - id_mapper: dict of ID mappings to write
        - id_mapper_path: path to the id mapper json file

    Returns:
        - 0 if successful, 1 if error
    """
    id_mapper_path = Path(id_mapper_path)

    if os.path.exists(id_mapper_path):
        # copy the old file to a backup using datetime to make it unique
        backup_dir = id_mapper_path.parent / "backup"
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = backup_dir / f"{pd.Timestamp.now().strftime('%Y%m%d')}_{id_mapper_path.name}"
        shutil.copy2(id_mapper_path, backup_path)
        print(f"backed up old id_mapper to {backup_path}")

    if not os.path.exists(id_mapper_path.parent):
        os.makedirs(id_mapper_path.parent, exist_ok=True)
        print(f"created directory for id_mapper export at {id_mapper_path.parent}")

    mode = "w"
    with open(id_mapper_path, mode) as f:
        try:
            json.dump(id_mapper, f, indent=4)
        except TypeError:
            print(f"error writing id_mapper to {id_mapper_path}")
            print(f"check that the id_mapper is a dictionary")
            return 1
        print(f"saved id_mapper to {id_mapper_path}")
    return 0


#TODO: Here passing one of the three normalized sourcce values ["pmdbs", "mouse", "invitro"].
#      but now it's based on CDE ValidCategories (organism and sample_source).
#      This is a temporary hack for the Feb2026/March2026 releases which use PMDBS/MOUSE/CELL ASAP IDs
#      A fututre implementation will fully transition to general SUBJECT ASAP IDs.
def generate_asap_dataset_id(
        dataset_id_mapper: dict,
        dataset_id: str,
        source_for_ids: str

) -> tuple[str, dict]:
    """
    Generate new ASAP dataset ids from the dataset_id.  Format will be "DS_xxxx"

    Required fields:
        - dataset_id_mapper: dict of existing dataset_id mappings
        - dataset_id: (e.g., "team-smith-pmdbs-sn-rnaseq")
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future

    Returns:
        - asap_dataset_name: the generated ASAP dataset name (e.g., "DS_BRAIN_0001")
        - updated dataset_id_mapper dict with the new mapping included
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")
    
    if dataset_name in dataset_id_mapper.keys():
        print(
            f"{dataset_name} already has a dataset_name: {dataset_id_mapper[dataset_name]}"
        )
        print(f"beware of potential dataset_name collision")
        asap_dataset_name = dataset_id_mapper[dataset_name]
        return asap_dataset_name, dataset_id_mapper
    else:
        n = len(dataset_id_mapper) + 1
        asap_dataset_name = f"DS_{source_for_ids.upper()}_{n:04}"
        dataset_id_mapper[dataset_name] = asap_dataset_name
        return asap_dataset_name, dataset_id_mapper


def generate_asap_team_id(team: str) -> str:
    """
    Input: team name and output: team_id

    Required fields:
    - team: team name (e.g., "smith")

    """
    return f"TEAM_{team.upper()}"


def get_sampr(v):
    return int(v.split("_")[3].replace("s", ""))

def get_id(v):
    return v[:17]


#####################
# PMDBS specific functions
#####################

def export_pmdbs_id_mappers(
    map_path: str | Path,
    suffix: str,
    datasetid_mapper: dict,
    subjectid_mapper: dict,
    sampleid_mapper: dict,
    gp2id_mapper: dict,
    sourceid_mapper: dict,
):
    """
    Export the id mappers to json files

    Required fields:
    - map_path: path to the directory where the id mappers will be saved
    - suffix: suffix to add to the file names (e.g., "ids")
    - datasetid_mapper: dict of dataset_id mappings
    - subjectid_mapper: dict of subject_id mappings
    - sampleid_mapper: dict of sample_id mappings
    - gp2id_mapper: dict of gp2_id mappings
    - sourceid_mapper: dict of source_subject_id mappings
    """
    map_path = Path(map_path)

    source = "PMDBS"

    subject_mapper_path = os.path.join(map_path, f"ASAP_{source}_subj_{suffix}.json")
    sample_mapper_path = os.path.join(map_path, f"ASAP_{source}_samp_{suffix}.json")
    gp2_mapper_path = os.path.join(map_path, f"ASAP_{source}_gp2_{suffix}.json")
    source_mapper_path = os.path.join(
        map_path, f"ASAP_{source}_sourcesubj_{suffix}.json"
    )
    dataset_mapper_path = os.path.join(map_path, f"ASAP_dataset_{suffix}.json")
    # update the dataset_id_mapper
    write_id_mapper(datasetid_mapper, dataset_mapper_path)
    write_id_mapper(subjectid_mapper, subject_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)
    write_id_mapper(gp2id_mapper, gp2_mapper_path)
    write_id_mapper(sourceid_mapper, source_mapper_path)


def update_pmdbs_id_mappers(
    clinpath_df,
    sample_df,
    dataset_id,
    source_for_ids,
    datasetid_mapper,
    subjectid_mapper,
    sampleid_mapper,
    gp2id_mapper,
    sourceid_mapper,
):
    """
    Read in the CLINPATH and SAMPLE data tables, generate new ids, update the id_mappers

    Required fields:
        - clinpath_df: CLINPATH metadata table as a dataframe
        - sample_df: SAMPLE metadata table as a dataframe
        - dataset_id: dataset_id (e.g., "team-smith-pmdbs-sn-rnaseq")
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future
        - datasetid_mapper: dict of existing dataset_id mappings
        - subjectid_mapper: dict of existing subject_id mappings
        - sampleid_mapper: dict of existing sample_id mappings
        - gp2id_mapper: dict of existing gp2_id mappings
        - sourceid_mapper: dict of existing source_subject_id mappings
    
    return updated id_mappers
    """

    if dataset_id.startswith("team-"):
        pass
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")
    
    _, datasetid_mapper = generate_asap_dataset_id(datasetid_mapper, dataset_id, source_for_ids)

    subjec_ids_df = clinpath_df[["subject_id", "source_subject_id", "GP2_id"]]

    # add ASAP_subject_id to the SUBJECT tables
    output = generate_human_subject_ids(
        subjectid_mapper, gp2id_mapper, sourceid_mapper, subjec_ids_df, source_for_ids
    )
    subjectid_mapper, gp2id_mapper, sourceid_mapper = output

    sample_ids_df = sample_df[["sample_id", "subject_id", "source_sample_id"]]
    sampleid_mapper = generate_human_sample_ids(
        subjectid_mapper, sampleid_mapper, sample_ids_df
    )

    return (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    )


# TODO: this function needs refactoring to be more modular and readable, but the general logic is:
# For each unique subject_id in the clinpath table, we check if we know the corresponding GP2_id or source_subject_id.
# If we do, we use the existing ASAP_subject_id. If not, we generate a new ASAP_subject_id and update the mappers accordingly.
# Issues to resolve
# 1) We need to decide how to handle cases where the same subject_id has multiple GP2_ids or source_subject_ids
#    (this should not happen and may indicate a data quality issue that needs to be logged).
# 2) We need to decide how to handle cases where we have no GP2_id or source_subject_id for a subject_id 
#    (this also may indicate a data quality issue that needs to be logged).
# 3) Using subject_id as the primary key across teams shouldn't be used, we should use source_subject_id but restricted to actual Biobank ID formats.

def generate_human_subject_ids(
    subjectid_mapper: dict,
    gp2id_mapper: dict,
    sourceid_mapper: dict,
    subject_df: pd.DataFrame,
    source_for_ids: str,
) -> tuple[dict, dict, dict]:
    """
    Generate new unique_ids for new subject_ids in subject_df table,
    update the id_mapper with the new ids from the data table

    Required fields:
        - subjectid_mapper: dict of existing subject_id mappings
        - gp2id_mapper: dict of existing gp2_id mappings
        - sourceid_mapper: dict of existing source_subject_id mappings
        - subject_df: subject metadata table as a dataframe, must contain columns "subject_id", "source_subject_id", and "GP2_id"
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future

    Returns:
        - Updated id_mappers
    """

    # force NA to be Nan
    subject_df = subject_df.replace("NA", pd.NA)

    # extract the max value of the mapper's third (last) section ([2] or [-1]) to get our n
    if bool(subjectid_mapper):
        n = max([int(v.split("_")[2]) for v in subjectid_mapper.values() if v]) + 1
    else:
        n = 1
    nstart = n

    # ids_df = subject_df[['subject_id','source_subject_id', 'AMPPD_id', 'GP2_id']].copy()
    ids_df = subject_df.copy()

    # might want to use 'source_subject_id' instead of 'subject_id' since we want to find matches across teams
    # shouldn't actually matter but logically cleaner
    uniq_subj = ids_df["subject_id"].unique()
    dupids_mapper = dict(
        zip(uniq_subj, [num + nstart for num in range(len(uniq_subj))])
    )

    n_asap_id_add = 0
    n_gp2_id_add = 0
    n_source_id_add = 0

    df_dup_chunks = []
    id_source = []
    for subj_id, samp_n in dupids_mapper.items():
        df_dups_subset = ids_df[ids_df.subject_id == subj_id].copy()

        # check if gp2_id is known
        # NOTE:  the gp2_id _might_ not be the GP2ID, but instead the GP2sampleID
        #        we might want to check for a trailing _s\d+ and remove it
        #        need to check w/ GP2 team about this.  The RepNo might be sample timepoint...
        #        and hence be a "subject" in our context
        #    # df['GP2ID'] = df['GP2sampleID'].apply(lambda x: ("_").join(x.split("_")[:-1]))
        #    # df['SampleRepNo'] = df['GP2sampleID'].apply(lambda x: x.split("_")[-1])#.replace("s",""))

        gp2_id = None
        add_gp2_id = False
        # force skipping of null GP2_ids
        if df_dups_subset["GP2_id"].nunique() > 1:
            print(
                f"subj_id: {subj_id} has multiple gp2_ids: {df_dups_subset['GP2_id'].to_list()}... something is wrong"
            )
            # TODO: log this
        elif not df_dups_subset["GP2_id"].dropna().empty:  # we have a valide GP2_id
            gp2_id = df_dups_subset["GP2_id"].values[
                0
            ]  # values because index was not reset

        if gp2_id in set(gp2id_mapper.keys()):
            asap_subj_id_gp2 = gp2id_mapper[gp2_id]
        else:
            add_gp2_id = True
            asap_subj_id_gp2 = None

        # check if source_id is known
        source_id = None
        add_source_id = False
        if df_dups_subset["source_subject_id"].nunique() > 1:
            print(
                f"subj_id: {subj_id} has multiple source ids: {df_dups_subset['source_subject_id'].to_list()}... something is wrong"
            )
            # TODO: log this
        elif df_dups_subset["source_subject_id"].isnull().any():
            print(f"subj_id: {subj_id} has no source_id... something is wrong")
            # TODO: log this
        else:  # we have a valide source_id
            # TODO: check for `source_subject_id` naming collisions with other teams
            #      e.g. check the `biobank_name`
            source_id = df_dups_subset["source_subject_id"].values[0]

        if source_id in set(sourceid_mapper.keys()):
            asap_subj_id_source = sourceid_mapper[source_id]
        else:
            add_source_id = True
            asap_subj_id_source = None

        # TODO: add AMPPD_id test/mapper

        # check if subj_id is known
        add_subj_id = False
        # check if subj_id (subject_id) is known
        if subj_id in set(subjectid_mapper.keys()):  # duplicate!!
            # TODO: log this
            # TODO: check for `subject_id` naming collisions with other teams
            asap_subj_id = subjectid_mapper[subj_id]
        else:
            add_subj_id = True
            asap_subj_id = None

        # TODO:  improve the logic here so gp2 is the default if it exists.?
        #        we need to check the team_id to make sure it's not a naming collision on subject_id
        #        we need to check the biobank_name to make sure it's not a naming collision on source_subject_id

        testset = set((asap_subj_id, asap_subj_id_gp2, asap_subj_id_source))
        if None in testset:
            testset.remove(None)

        # check that asap_subj_id is not disparate between the maps
        if len(testset) > 1:
            print(
                f"collission between our ids: {(asap_subj_id, asap_subj_id_gp2, asap_subj_id_source)=}"
            )
            print(
                f"this is BAAAAD. could be a naming collision with another team on `subject_id` "
            )

        if len(testset) == 0:  # generate a new asap_subj_id
            # print(samp_n)
            asap_subject_id = f"ASAP_{source_for_ids.upper()}_{samp_n:06}"
            # df_dups_subset.insert(0, 'ASAP_subject_id', asap_subject_id, inplace=True)
        else:  # testset should have the asap_subj_id
            asap_subject_id = testset.pop()  # but where did it come from?
            # print(f"found {subj_id }:{asap_subject_id} in the maps")

        src = []
        if add_subj_id:
            # TODO:  instead of just adding we should check if it exists...
            subjectid_mapper[subj_id] = asap_subject_id
            n_asap_id_add += 1
            src.append("asap")

        if add_gp2_id and gp2_id is not None:
            # TODO:  instead of just adding we should check if it exists...
            gp2id_mapper[gp2_id] = asap_subject_id
            n_gp2_id_add += 1
            src.append("gp2")

        if add_source_id and source_id is not None:
            # TODO:  instead of just adding we should check if it exists...
            sourceid_mapper[source_id] = asap_subject_id
            n_source_id_add += 1
            src.append("source")

        df_dup_chunks.append(df_dups_subset)
        id_source.append(src)

    df_dups_wids = pd.concat(df_dup_chunks)
    assert df_dups_wids.sort_index().equals(subject_df)
    print(f"added {n_asap_id_add} new asap_subject_ids")
    print(f"added {n_gp2_id_add} new gp2_ids")
    print(f"added {n_source_id_add} new source_ids")

    return subjectid_mapper, gp2id_mapper, sourceid_mapper


def generate_human_sample_ids(
    subjectid_mapper: dict, 
    sampleid_mapper: dict, 
    sample_df: pd.DataFrame
) -> dict:
    """
    Generate new unique_ids for new sample_ids in sample_df table,
    update the id_mapper with the new ids from the data table

    Required fields:
        - subjectid_mapper: dict of existing subject_id mappings
        - sampleid_mapper: dict of existing sample_id mappings
        - sample_df: sample metadata table as a dataframe, must contain columns "sample_id" and "subject_id"

    Returns:
        - Updated sampleid_mapper with new mappings from sample_df
    """

    ud_sampleid_mapper = sampleid_mapper.copy()

    uniq_samp = sample_df.sample_id.unique()
    if samp_intersec := set(uniq_samp) & set(ud_sampleid_mapper.keys()):
        print(
            f"found {len(samp_intersec)} sample_id's that have already been mapped!! BEWARE a sample_id naming collision!! If you are just reprocessing tables, it shoud be okay."
        )

    to_map = sample_df[
        ~sample_df["sample_id"].apply(lambda x: x in samp_intersec)
    ].copy()

    if not bool(to_map.shape[0]):
        print(
            "Nothing to see here... move along... move along .... \nNo new sample_ids to map"
        )
        return ud_sampleid_mapper

    uniq_subj = to_map.subject_id.unique()
    # check for subject_id collisions in the sampleid_mapper
    if subj_intersec := set(uniq_subj) & set(ud_sampleid_mapper.values()):
        print(
            f"found {len(subj_intersec)} subject_id collisions in the sampleid_mapper"
        )

    df_chunks = []
    for subj_id in uniq_subj:

        df_subset = to_map[to_map.subject_id == subj_id].copy()
        asap_id = subjectid_mapper[subj_id]

        dups = (
            df_subset[df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )
        nodups = (
            df_subset[~df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )

        asap_id = subjectid_mapper[subj_id]
        if bool(ud_sampleid_mapper):
            # see if there are any samples already with this asap_id
            sns = [
                get_sampr(v)
                for v in ud_sampleid_mapper.values()
                if get_id(v) == asap_id
            ]
            if len(sns) > 0:
                rep_n = max(sns) + 1
            else:
                rep_n = 1  # start incrimenting from 1
        else:  # empty dicitonary. starting from scratch
            rep_n = 1

        if nodups.shape[0] > 0:
            # ASSIGN IDS
            asap_nodups = [f"{asap_id}_s{rep_n+i:03}" for i in range(nodups.shape[0])]
            # nodups['ASAP_sample_id'] = asap_nodups
            nodups.loc[:, "ASAP_sample_id"] = asap_nodups
            rep_n = rep_n + nodups.shape[0]
            samples_nodups = nodups["sample_id"].unique()

            nodup_mapper = dict(zip(nodups["sample_id"], asap_nodups))

            df_chunks.append(nodups)
        else:
            samples_nodups = []

        if dups.shape[0] > 0:
            for dup_id in dups["sample_id"].unique():
                # first peel of any sample_ids that were already named in nodups,

                if dup_id in samples_nodups:
                    asap_dup = nodup_mapper[dup_id]
                else:
                    # then assign ids to the rest.
                    asap_dup = f"{asap_id}_s{rep_n:03}"
                    dups.loc[dups.sample_id == dup_id, "ASAP_sample_id"] = asap_dup
                    rep_n += 1
            df_chunks.append(dups)

    df_wids = pd.concat(df_chunks)
    id_mapper = dict(zip(df_wids["sample_id"], df_wids["ASAP_sample_id"]))

    ud_sampleid_mapper.update(id_mapper)

    # the n_add is NOT correct...
    n_add = len(ud_sampleid_mapper.keys()) - len(id_mapper.keys())

    print(f"added {n_add} new sample_ids")

    return ud_sampleid_mapper


#####################
# MOUSE specific functions
#####################

def export_mouse_id_mappers(
    map_path: Path | str,
    suffix: str,
    datasetid_mapper: dict,
    mouseid_mapper: dict,
    sampleid_mapper: dict,
):
    """
    Export the ID mappers to JSON files for the MOUSE source type.

    Required fields:
        - map_path: path to the directory where the ID mapper files will be saved
        - suffix: a string suffix to append to the filenames (e.g., "ids")
        - datasetid_mapper: dict of dataset ID mappings to export
        - mouseid_mapper: dict of mouse subject ID mappings to export
        - sampleid_mapper: dict of sample ID mappings to export
    
    """
    map_path = Path(map_path)
    source = "MOUSE"
    sample_mapper_path = os.path.join(map_path, f"ASAP_{source}_samp_{suffix}.json")
    mouse_mapper_path = os.path.join(map_path, f"ASAP_{source}_{suffix}.json")
    dataset_mapper_path = os.path.join(map_path, f"ASAP_dataset_{suffix}.json")
    # update the dataset_id_mapper
    write_id_mapper(datasetid_mapper, dataset_mapper_path)
    write_id_mapper(mouseid_mapper, mouse_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)


def update_mouse_id_mappers(
    subject_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    dataset_id: str,
    source_for_ids: str,
    datasetid_mapper: dict,
    mouseid_mapper: dict,
    sampleid_mapper: dict,
):
    """
    Read in the SUBJECT and SAMPLE data tables, generate new IDs, update the ID mappers

    Required fields:
        - subject_df: SUBJECT metadata table as a dataframe
        - sample_df: SAMPLE metadata table as a dataframe
        - dataset_id: dataset ID (e.g., "team-smith-pmdbs-sn-rnaseq")
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future
        - datasetid_mapper: dict of existing dataset ID mappings
        - mouseid_mapper: dict of existing mouse subject ID mappings
        - sampleid_mapper: dict of existing sample ID mappings

    return updated id_mappers
    """

    if dataset_id.startswith("team-"):
        pass
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    _, datasetid_mapper = generate_asap_dataset_id(datasetid_mapper, dataset_id, source_for_ids)

    subject_ids_df = subject_df[["subject_id"]]

    # add ASAP_subject_id to the SUBJECT tables
    mouseid_mapper = generate_mouse_subject_ids(mouseid_mapper, subject_ids_df)
    sample_ids_df = sample_df[["sample_id", "subject_id"]]
    sampleid_mapper = generate_mouse_sample_ids(
        mouseid_mapper, sampleid_mapper, sample_ids_df
    )

    return datasetid_mapper, mouseid_mapper, sampleid_mapper


def generate_mouse_subject_ids(mouseid_mapper: dict, subject_df: pd.DataFrame) -> dict:
    """
    generate new unique_ids for new subject_ids in subject_df table,
    update the id_mapper with the new ids from the data table

    Args:
        mouseid_mapper (dict): Existing subject ID mapper
        subject_df (pd.DataFrame): DataFrame containing subject information

    Returns:
        dict: Updated subject ID mapper"""
    # Initialize the mapper if it's None
    if mouseid_mapper is None:
        mouseid_mapper = {}

    source = "mouse"
    # Make a copy to avoid modifying the original
    mapper = mouseid_mapper.copy()

    # Get the next available ID number
    existing_ids = [
        id for id in mapper.values() if id.startswith(f"ASAP_{source.upper()}_")
    ]
    if existing_ids:
        # Extract numbers from existing IDs and find the max
        max_num = (
            max([int(id.split("_")[-1]) for id in existing_ids]) if existing_ids else 0
        )
        next_num = max_num + 1
    else:
        next_num = 1

    # Process each subject
    for _, row in subject_df.iterrows():
        subject_id = row.get("subject_id")
        if subject_id and subject_id not in mapper:
            # Generate new ID
            asap_id = f"ASAP_{source.upper()}_{next_num:06d}"
            mapper[subject_id] = asap_id
            next_num += 1

    return mapper


def generate_mouse_sample_ids(
    mouseid_mapper: dict, sampleid_mapper: dict, sample_df: pd.DataFrame
) -> dict:
    """
    generate new unique_ids for new sample_ids in sample_df table,
    update the id_mapper with the new ids from the data table

    Args:
        mouseid_mapper (dict): Existing subject ID mapper
        sampleid_mapper (dict): Existing sample ID mapper
        sample_df (pd.DataFrame): DataFrame containing sample + subject information
        source (str): Source identifier for the IDs

    Returns:
        dict: Updated sample ID mapper
    """

    ud_sampleid_mapper = sampleid_mapper.copy()

    uniq_samp = sample_df.sample_id.unique()
    if samp_intersec := set(uniq_samp) & set(ud_sampleid_mapper.keys()):
        print(
            f"found {len(samp_intersec)} sample_id's that have already been mapped!! BEWARE a sample_id naming collision!! If you are just reprocessing tables, it shoud be okay."
        )

    to_map = sample_df[
        ~sample_df["sample_id"].apply(lambda x: x in samp_intersec)
    ].copy()

    if not bool(to_map.shape[0]):
        print(
            "Nothing to see here... move along... move along .... \nNo new sample_ids to map"
        )
        return ud_sampleid_mapper

    uniq_subj = to_map.subject_id.unique()
    # check for subject_id collisions in the sampleid_mapper
    if subj_intersec := set(uniq_subj) & set(ud_sampleid_mapper.values()):
        print(
            f"found {len(subj_intersec)} subject_id collisions in the sampleid_mapper"
        )

    df_chunks = []
    for subj_id in uniq_subj:

        df_subset = to_map[to_map.subject_id == subj_id].copy()
        asap_id = mouseid_mapper[subj_id]

        dups = (
            df_subset[df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )
        nodups = (
            df_subset[~df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )

        asap_id = mouseid_mapper[subj_id]
        if bool(ud_sampleid_mapper):
            # see if there are any samples already with this asap_id
            sns = [
                get_sampr(v)
                for v in ud_sampleid_mapper.values()
                if get_id(v) == asap_id
            ]
            if len(sns) > 0:
                rep_n = max(sns) + 1
            else:
                rep_n = 1  # start incrimenting from 1
        else:  # empty dicitonary. starting from scratch
            rep_n = 1

        if nodups.shape[0] > 0:
            # ASSIGN IDS
            asap_nodups = [f"{asap_id}_s{rep_n+i:03}" for i in range(nodups.shape[0])]
            # nodups['ASAP_sample_id'] = asap_nodups
            nodups.loc[:, "ASAP_sample_id"] = asap_nodups
            rep_n = rep_n + nodups.shape[0]
            samples_nodups = nodups["sample_id"].unique()

            nodup_mapper = dict(zip(nodups["sample_id"], asap_nodups))

            df_chunks.append(nodups)
        else:
            samples_nodups = []

        if dups.shape[0] > 0:
            for dup_id in dups["sample_id"].unique():
                # first peel of any sample_ids that were already named in nodups,

                if dup_id in samples_nodups:
                    asap_dup = nodup_mapper[dup_id]
                else:
                    # then assign ids to the rest.
                    asap_dup = f"{asap_id}_s{rep_n:03}"
                    dups.loc[dups.sample_id == dup_id, "ASAP_sample_id"] = asap_dup
                    rep_n += 1
            df_chunks.append(dups)

    df_wids = pd.concat(df_chunks)
    id_mapper = dict(zip(df_wids["sample_id"], df_wids["ASAP_sample_id"]))

    ud_sampleid_mapper.update(id_mapper)
    # print(ud_sampleid_mapper)
    return ud_sampleid_mapper

#####################
# INVITRO specific functions
#####################

def export_cell_id_mappers(
    map_path: Path | str,
    suffix: str,
    datasetid_mapper: dict,
    cellid_mapper: dict,
    sampleid_mapper: dict,
):
    map_path = Path(map_path)
    source = "INVITRO"
    sample_mapper_path = os.path.join(map_path, f"ASAP_{source}_samp_{suffix}.json")
    cell_mapper_path = os.path.join(map_path, f"ASAP_{source}_{suffix}.json")
    dataset_mapper_path = os.path.join(map_path, f"ASAP_dataset_{suffix}.json")
    # update the dataset_id_mapper
    write_id_mapper(datasetid_mapper, dataset_mapper_path)
    write_id_mapper(cellid_mapper, cell_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)


def update_cell_id_mappers(
    cell_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    dataset_id: str,
    source_for_ids: str,
    datasetid_mapper: dict,
    cellid_mapper: dict,
    sampleid_mapper: dict,
):
    """
    Read in the CELL (cell_df) and SAMPLE (sample_df) data tables, generate new ids, update the id_mappers

    Required fields:
        - cell_df: CELL metadata table as a dataframe
        - sample_df: SAMPLE metadata table as a dataframe
        - dataset_id: dataset_id (e.g., "team-smith-pmdbs-sn-rnaseq")
        - source_for_ids: normalized source for ID mappers. One of ["pmdbs", "mouse", "invitro"],
                          which is currently determined by organism/source but will be updated in the future
        - datasetid_mapper: dict of existing dataset_id mappings
        - cellid_mapper: dict of existing cell_id mappings
        - sampleid_mapper: dict of existing sample_id mappings

    Returns:
         updated id_mappers
    """

    if dataset_id.startswith("team-"):
        pass
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    _, datasetid_mapper = generate_asap_dataset_id(datasetid_mapper, dataset_id, source_for_ids)

    # subject_id is the alias for cell_id
    if "subject_id" not in cell_df.columns:
        cell_col = "cell_id"
    else:
        cell_col = "subject_id"

    cell_ids_df = cell_df[[cell_col]]

    # add ASAP_subject_id to the SUBJECT tables
    cellid_mapper = generate_cell_ids(cellid_mapper, cell_ids_df)

    sample_ids_df = sample_df[["sample_id", cell_col]]
    sampleid_mapper = generate_cell_sample_ids(
        cellid_mapper, sampleid_mapper, sample_ids_df
    )

    return datasetid_mapper, cellid_mapper, sampleid_mapper


def generate_cell_ids(
        cellid_mapper: dict, 
        cell_df: pd.DataFrame) -> dict:
    """
    Generate new unique_ids for new cell_ids (subject_id) in cell_df table,
    update the id_mapper with the new ids from the data table

    Required fields:
        - cellid_mapper: dict of existing cell ID mappings
        - cell_df: cell metadata table as a dataframe, must contain a column "cell_id" or "subject_id" (which is the alias for cell_id)
    
    Returns:
        dict: Updated cell ID mapper
    """
    # Initialize the mapper if it's None
    if cellid_mapper is None:
        cellid_mapper = {}

    # FORCE "cell" as the generic term
    source_for_outfile = "cell"
    # Make a copy to avoid modifying the original
    mapper = cellid_mapper.copy()

    # Get the next available ID number
    existing_ids = [
        id for id in mapper.values() if id.startswith(f"ASAP_{source_for_outfile.upper()}_")
    ]
    if existing_ids:
        # Extract numbers from existing IDs and find the max
        max_num = (
            max([int(id.split("_")[-1]) for id in existing_ids]) if existing_ids else 0
        )
        next_num = max_num + 1
        print(f"found {len(existing_ids)} existing cell IDs. starting from {next_num}")
    else:
        next_num = 1
        print(f"no existing cell IDs found. starting from {next_num}")

    # subject_id is the alias for cell_id
    if "subject_id" not in cell_df.columns:
        cell_col = "cell_id"
    else:
        cell_col = "subject_id"

    # Process each subject
    for _, row in cell_df.iterrows():

        cell_id = row.get(cell_col)
        if cell_id and cell_id not in mapper:
            # Generate new ID
            asap_id = f"ASAP_{source_for_outfile.upper()}_{next_num:06d}"
            mapper[cell_id] = asap_id
            next_num += 1

    return mapper


def generate_cell_sample_ids(
    cellid_mapper: dict, sampleid_mapper: dict, sample_df: pd.DataFrame
) -> dict:
    """
    generate new unique_ids for new sample_ids in sample_df table,
    update the id_mapper with the new ids from the data table

    Args:
        cellid_mapper (dict): Existing cell ID mapper
        sampleid_mapper (dict): Existing sample ID mapper
        sample_df (pd.DataFrame): DataFrame containing sample + subject information

    Returns:
        dict: Updated sample ID mapper
    """

    ud_sampleid_mapper = sampleid_mapper.copy()

    uniq_samp = sample_df.sample_id.unique()
    if samp_intersec := set(uniq_samp) & set(ud_sampleid_mapper.keys()):
        print(
            f"found {len(samp_intersec)} sample_id's that have already been mapped!! BEWARE a sample_id naming collision!! If you are just reprocessing tables, it shoud be okay."
        )

    # subject_id is the alias for cell_id
    if "subject_id" not in sample_df.columns:
        cell_col = "cell_id"
    else:
        cell_col = "subject_id"

    to_map = sample_df[~sample_df[cell_col].apply(lambda x: x in samp_intersec)].copy()

    if not bool(to_map.shape[0]):
        print(
            "Nothing to see here... move along... move along .... \nNo new sample_ids to map"
        )
        return ud_sampleid_mapper

    uniq_cell = to_map[cell_col].unique()
    # check for subject_id collisions in the sampleid_mapper
    if cell_intersec := set(uniq_cell) & set(ud_sampleid_mapper.values()):
        print(f"found {len(cell_intersec)} cell_id collisions in the cellid_mapper")

    df_chunks = []
    for cell_id in uniq_cell:

        df_subset = to_map[to_map[cell_col] == cell_id].copy()
        asap_id = cellid_mapper[cell_id]

        dups = (
            df_subset[df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )
        nodups = (
            df_subset[~df_subset.duplicated(keep=False, subset=["sample_id"])]
            .sort_values("sample_id")
            .reset_index(drop=True)
            .copy()
        )

        asap_id = cellid_mapper[cell_id]
        if bool(ud_sampleid_mapper):
            # see if there are any samples already with this asap_id
            sns = [
                get_sampr(v)
                for v in ud_sampleid_mapper.values()
                if get_id(v) == asap_id
            ]
            if len(sns) > 0:
                rep_n = max(sns) + 1
            else:
                rep_n = 1  # start incrimenting from 1
        else:  # empty dicitonary. starting from scratch
            rep_n = 1

        if nodups.shape[0] > 0:
            # ASSIGN IDS
            asap_nodups = [f"{asap_id}_s{rep_n+i:03}" for i in range(nodups.shape[0])]
            # nodups['ASAP_sample_id'] = asap_nodups
            nodups.loc[:, "ASAP_sample_id"] = asap_nodups
            rep_n = rep_n + nodups.shape[0]
            samples_nodups = nodups[cell_col].unique()

            nodup_mapper = dict(zip(nodups[cell_col], asap_nodups))

            df_chunks.append(nodups)
        else:
            samples_nodups = []

        if dups.shape[0] > 0:
            for dup_id in dups[cell_col].unique():
                # first peel of any sample_ids that were already named in nodups,

                if dup_id in samples_nodups:
                    asap_dup = nodup_mapper[dup_id]
                else:
                    # then assign ids to the rest.
                    asap_dup = f"{asap_id}_s{rep_n:03}"
                    dups.loc[dups.sample_id == dup_id, "ASAP_sample_id"] = asap_dup
                    rep_n += 1
            df_chunks.append(dups)

    df_wids = pd.concat(df_chunks)
    id_mapper = dict(zip(df_wids["sample_id"], df_wids["ASAP_sample_id"]))

    ud_sampleid_mapper.update(id_mapper)
    return ud_sampleid_mapper


#####################
# Functions used by crn-utils/release_util.py only
#####################

def load_pmdbs_id_mappers(map_path, suffix):
    source = "PMDBS"

    prototypes = ["dataset", "subj", "samp", "gp2", "sourcesubj"]

    outputs = ()
    for prot in prototypes:
        if prot == "dataset":
            fname = f"ASAP_{prot}_{suffix}.json"
        else:
            fname = f"ASAP_{source}_{prot}_{suffix}.json"

        try:
            id_mapper = load_id_mapper(os.path.join(map_path, fname))
        except FileNotFoundError:
            id_mapper = {}
            print(f"{os.path.join(map_path, fname)} not found... starting from scratch")
        outputs += (id_mapper,)

    return outputs


def update_pmdbs_meta_tables_with_asap_ids(
    dfs: dict,
    dataset_id: str,
    asap_ids_schema: pd.DataFrame,
    datasetid_mapper: dict,
    subjectid_mapper: dict,
    sampleid_mapper: dict,
    gp2id_mapper: dict,
    sourceid_mapper: dict,
    pmdbs_tables: list | None = None,
) -> dict:
    """
    Process the metadata tables to add ASAP_IDs to the tables with the mappers

    Required fields:
        - dfs: dict of dataframes for each table in the dataset
        - dataset_id: dataset ID (e.g., "team-smith-pmdbs-sn-rnaseq")
        - asap_ids_schema: DataFrame containing the schema for ASAP IDs, with columns "Table" and "Field" indicating which tables should have which ASAP IDs
        - datasetid_mapper: dict mapping long dataset names to ASAP dataset IDs
        - subjectid_mapper: dict mapping subject_ids to ASAP_subject_ids
        - sampleid_mapper: dict mapping sample_ids to ASAP_sample_ids
        - gp2id_mapper: dict mapping GP2_ids to ASAP_subject_ids
        - sourceid_mapper: dict mapping source_subject_ids to ASAP_subject_ids
        - pmdbs_tables: list of table names to process for adding ASAP IDs (defaults to all PMDBS tables plus "SPATIAL")

    PMDBS tables:
        ['PMDBS', 'CONDITION', 'CLINPATH', 'SUBJECT', 'ASSAY_RNAseq', 'SAMPLE', 'DATA', 'STUDY', 'PROTOCOL']
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")


    if pmdbs_tables is None:
        pmdbs_tables = PMDBS_TABLES.copy() + ["SPATIAL"]
        print(f"default {pmdbs_tables=}")

    ASAP_sample_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_sample_id"
    ]["Table"].to_list()
    ASAP_subject_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_subject_id"
    ]["Table"].to_list()

    DATASET_ID = datasetid_mapper[dataset_name]

    if "STUDY" in dfs.keys():  # un-necessary check
        TEAM_ID = dfs["STUDY"]["ASAP_team_name"].str.upper().str.replace("-", "_")[0]
    else:  # this should NEVER happen
        print(f"STUDY table not found in dataset {dataset_name}")
        TEAM_ID = "TEAM_" + dataset_name.split("-")[0].upper()

    # now we add the IDs
    for tab in pmdbs_tables:
        if tab not in dfs:
            print(f"Table {tab} not found in dataset {dataset_name}")
            continue

        if tab in ASAP_subject_id_tables:
            # first do the ASAP_subject_id
            ASAP_subject_id = dfs[tab]["subject_id"].map(subjectid_mapper)
            dfs[tab].insert(0, "ASAP_subject_id", ASAP_subject_id)

        if tab in ASAP_sample_id_tables:
            # second do the ASAP_sample_id
            ASAP_sample_id = dfs[tab]["sample_id"].map(sampleid_mapper)
            dfs[tab].insert(0, "ASAP_sample_id", ASAP_sample_id)

        # insert the DATASET_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_dataset_id", DATASET_ID)
        # insert the TEAM_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_team_id", TEAM_ID)
    return dfs


def update_mouse_meta_tables_with_asap_ids(
    dfs: dict,
    dataset_id: str,
    asap_ids_schema: pd.DataFrame,
    datasetid_mapper: dict,
    mouseid_mapper: dict,
    sampleid_mapper: dict,
    mouse_tables: list | None = None,
) -> dict:
    """
    Process the metadata tables to add ASAP_IDs to the tables with the mappers

    Required fields:
        - dfs: dict of dataframes for each table in the dataset
        - dataset_id: dataset ID (e.g., "team-smith-pmdbs-sn-rnaseq")
        - asap_ids_schema: DataFrame containing the schema for ASAP IDs, with columns "Table" and "Field" indicating which tables should have which ASAP IDs
        - datasetid_mapper: dict mapping long dataset names to ASAP dataset IDs
        - mouseid_mapper: dict mapping mouse_ids to ASAP_mouse_ids
        - sampleid_mapper: dict mapping sample_ids to ASAP_sample_ids
        - mouse_tables: list of table names to process for adding ASAP IDs (defaults to all MOUSE_TABLES plus "SPATIAL")

    MOUSE_TABLES = [
        "STUDY",
        "PROTOCOL",
        "ASSAY_RNAseq",  # this is missing... try to construct...
        "SAMPLE",
        "MOUSE",
        "CONDITION",
        "DATA",
        ]
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    # default to mouse scPMDBS / bulkPMDBS
    if mouse_tables is None:
        mouse_tables = MOUSE_TABLES.copy() + ["SPATIAL"]

    ASAP_sample_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_sample_id"
    ]["Table"].to_list()
    ASAP_subject_id_tables = asap_ids_schema[
        (asap_ids_schema["Field"] == "ASAP_subject_id")
        | (asap_ids_schema["Field"] == "ASAP_mouse_id")
    ]["Table"].to_list()

    DATASET_ID = datasetid_mapper[dataset_name]

    if "STUDY" in dfs.keys():  # un-necessary check
        TEAM_ID = dfs["STUDY"]["ASAP_team_name"].str.upper().str.replace("-", "_")[0]
    else:  # this should NEVER happen
        print(f"STUDY table not found in dataset {dataset_name}")
        TEAM_ID = "TEAM_" + dataset_name.split("-")[0].upper()

    # now we add the IDs
    for tab in mouse_tables:
        if tab not in dfs:
            print(f"Table {tab} not found in dataset {dataset_name}")
            continue

        if tab in ASAP_subject_id_tables:
            # first do the ASAP_subject_id
            if "subject_id" in dfs[tab].columns:
                ASAP_subject_id = dfs[tab]["subject_id"].map(mouseid_mapper)
            elif "mouse_id" in dfs[tab].columns:
                ASAP_subject_id = dfs[tab]["mouse_id"].map(mouseid_mapper)
            else:
                print(f"subject_id or mouse_id not found in {tab}")
                continue
            dfs[tab].insert(0, "ASAP_mouse_id", ASAP_subject_id)

        if tab in ASAP_sample_id_tables:
            # second do the ASAP_sample_id
            ASAP_sample_id = dfs[tab]["sample_id"].map(sampleid_mapper)
            dfs[tab].insert(0, "ASAP_sample_id", ASAP_sample_id)

        # insert the DATASET_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_dataset_id", DATASET_ID)
        # insert the TEAM_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_team_id", TEAM_ID)

    return dfs


def load_mouse_id_mappers(map_path, suffix):
    source = "MOUSE"
    prototypes = ["dataset", "subj", "samp"]

    outputs = ()
    for prot in prototypes:
        if prot == "dataset":
            fname = f"ASAP_{prot}_{suffix}.json"
        elif prot == "subj":
            fname = f"ASAP_{source}_{suffix}.json"
        else:
            fname = f"ASAP_{source}_{prot}_{suffix}.json"

        try:
            id_mapper = load_id_mapper(os.path.join(map_path, fname))
        except FileNotFoundError:
            id_mapper = {}
            print(f"{os.path.join(map_path, fname)} not found... starting from scratch")

        print(f"loaded {fname}")
        outputs += (id_mapper,)

    return outputs


def load_cell_id_mappers(
    map_path: Path | str,
    suffix: str,
) -> tuple[dict, dict, dict]:

    map_path = Path(map_path)
    source = "INVITRO"
    prototypes = ["dataset", "cell", "samp"]
    outputs = ()
    for prot in prototypes:
        if prot == "dataset":
            fname = f"ASAP_{prot}_{suffix}.json"
        elif prot == "cell":
            fname = f"ASAP_{source}_{suffix}.json"
        else:
            fname = f"ASAP_{source}_{prot}_{suffix}.json"

        try:
            id_mapper = load_id_mapper(os.path.join(map_path, fname))
        except FileNotFoundError:
            id_mapper = {}
            print(f"{os.path.join(map_path, fname)} not found... starting from scratch")
        outputs += (id_mapper,)

    return outputs


def update_cell_meta_tables_with_asap_ids(
    dfs: dict,
    dataset_id: str,
    asap_ids_schema: pd.DataFrame,
    datasetid_mapper: dict,
    cellid_mapper: dict,
    sampleid_mapper: dict,
    cell_tables: list | None = None,
) -> dict:
    """
    Process the metadata tables to add ASAP_IDs to the tables with the mappers

    Required fields:
        - dfs: dict of dataframes for each table in the dataset
        - dataset_id: dataset ID (e.g., "team-smith-pmdbs-sn-rnaseq")
        - asap_ids_schema: DataFrame containing the schema for ASAP IDs, with   columns "Table" and "Field" indicating which tables should have which ASAP IDs
        - datasetid_mapper: dict mapping long dataset names to ASAP dataset IDs
        - cellid_mapper: dict mapping cell_ids to ASAP_cell_ids
        - sampleid_mapper: dict mapping sample_ids to ASAP_sample_ids
        - cell_tables: list of table names to process for adding ASAP IDs (defaults to all CELL_TABLES)


    CELL_TABLES = [
        "STUDY",
        "PROTOCOL",
        "ASSAY_RNAseq",  # this is missing... try to construct...
        "SAMPLE",
        "CELL",
        "CONDITION",
        "DATA",
        ]
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")

    cell_tables = list(set(dfs.keys()))
    print(f"{cell_tables=}")

    ASAP_sample_id_tables = asap_ids_schema[
        asap_ids_schema["Field"] == "ASAP_sample_id"
    ]["Table"].to_list()
    ASAP_subject_id_tables = asap_ids_schema[
        (asap_ids_schema["Field"] == "ASAP_subject_id")
        | (asap_ids_schema["Field"] == "ASAP_cell_id")
    ]["Table"].to_list()

    DATASET_ID = datasetid_mapper[dataset_name]

    if "STUDY" in dfs.keys():  # un-necessary check
        TEAM_ID = dfs["STUDY"]["ASAP_team_name"].str.upper().str.replace("-", "_")[0]
    else:  # this should NEVER happen
        print(f"STUDY table not found in dataset {dataset_name}")
        TEAM_ID = "TEAM_" + dataset_name.split("-")[0].upper()

    # now we add the IDs
    for tab in cell_tables:
        if tab not in dfs:
            print(f"Table {tab} not found in dataset {dataset_name}")
            continue

        if tab in ASAP_subject_id_tables:

            # first do the ASAP_subject_id
            if "subject_id" in dfs[tab].columns:
                ASAP_cell_id = dfs[tab]["subject_id"].map(cellid_mapper)
            # elif "mouse_id" in dfs[tab].columns:
            #     ASAP_subject_id = dfs[tab]["mouse_id"].map(cellid_mapper)
            elif "cell_id" in dfs[tab].columns:
                ASAP_cell_id = dfs[tab]["cell_id"].map(cellid_mapper)
            else:
                print(f"subject_id or cell_id not found in {tab}")
                continue
            dfs[tab].insert(0, "ASAP_cell_id", ASAP_cell_id)

        if tab in ASAP_sample_id_tables:
            # second do the ASAP_sample_id
            ASAP_sample_id = dfs[tab]["sample_id"].map(sampleid_mapper)
            dfs[tab].insert(0, "ASAP_sample_id", ASAP_sample_id)

        # insert the DATASET_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_dataset_id", DATASET_ID)
        # insert the TEAM_ID at the beginning of the dataframe
        dfs[tab].insert(0, "ASAP_team_id", TEAM_ID)

    return dfs
