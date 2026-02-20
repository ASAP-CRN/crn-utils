import os
import docx
import json
import logging
import pandas as pd
from pathlib import Path
from numpy import nan as np_nan


from .util import (
    list_expected_metadata_tables,
    read_CDE,
    NULL,
    read_meta_table,
    read_CDE_asap_ids,
    export_meta_tables,
    load_tables,
    write_version,
)

from .asap_ids import *
from .validate import validate_table, ReportCollector, process_table

from .file_metadata import (
    gen_raw_bucket_summary,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
    gen_spatial_bucket_summary,
    make_file_metadata,
)
from .constants import *
from .doi import update_study_table_with_doi
from .google_spreadsheets import read_google_sheet

__all__ = [
    "prep_release_metadata",
    "load_and_process_table",
    "process_schema",
    "create_metadata_package",
    "get_stat_tabs_pmdbs",
    "get_stat_tabs_human_non_brain",
    "get_stat_tabs_mouse",
    "get_stat_tabs_mouse_non_brain",
    "get_stats_table",
    "get_cohort_stats_table",
    "get_crn_release_metadata",
]

def get_spatial_subtype_from_dataset_id(dataset_id: str) -> str:
    """
    Determine spatial subtype (i.e. visium, geomx or cosmx) from dataset_id.
    """
    if "visium" in dataset_id.lower():
        return "visium"
    elif "geomx" in dataset_id.lower():
        return "geomx"
    elif "cosmx" in dataset_id.lower():
        return "cosmx"
    else:
        raise KeyError(f"get_spatial_subtype_from_dataset_id: Unable to determine spatial subtype from dataset_id: {dataset_id}")


# The following is a refactor of the main function to prepare metadata for a 
# release. It combines the previous source-specific functions into a unified
# call, while removing the ID generation step which is now handled separately.
# Following PRs will continue to refactor/deprecate the previous workflow.
# ----

def prep_release_metadata(dataset_id: str,
                          source: str,
                          team: str,
                          modality: str,
                          cde_version: str,
                          release_version: str,
                          metadata_dir: Path,
                          dataset_dir: Path,
                          map_path: Path
                          ) -> None:
    """
    Prepares dataset metadata for release.
    
    Workflow:
      1. Load staged metadata tables from metadata_dir
      2. Inject ASAP IDs from master ID mappers
      3. Fetch and save file metadata and GCP URIs from GCP raw bucket
      4. Merge file metadata with DATA table
      5. Inject DOI and project info from intake docx into STUDY table
      6. Export final release tables to {dataset_dir}/metadata/release/{release_version}/
    
    Args:
        dataset_id: Full dataset identifier (e.g., "cragg-mouse-sn-rnaseq-striatum")
        source: Data source type ("pmdbs", "mouse", "invitro")
        team: Team name (e.g., "cragg")
        modality: Assay modality (e.g., "rna", "spatial")
        cde_version: CDE schema version (e.g., "v3.3")
        release_version: CRN release version (e.g., "v4.0.1")
        metadata_dir: Path to metadata directory
        dataset_dir: Path to dataset directory
    """
    
    # ---- Load metadata tables ----    
    logging.info(f"Loading metadata tables from {metadata_dir}...")
    
    # TODO: This is a temporary heuristic as we move away from list_expected_metadata_table()
    present_files = [f.stem for f in metadata_dir.glob("*.csv")]
    expected_tables = [t for t in list_expected_metadata_tables() if t in present_files]
    meta_tables = load_tables(metadata_dir, expected_tables)
    
    logging.info(f"Loaded {len(meta_tables)} metadata tables: {', '.join(meta_tables.keys())}")
    
    # ---- Inserting ASAP IDs ----
    logging.info("Inserting ASAP IDs into metadata tables...")
    

    id_mappers = load_all_id_mappers(map_path, source)
    
    # TODO: This is a temporary heuristic until read_CDE() is updated for googlesheets
    # asap_ids_df = read_CDE_asap_ids(schema_version=cde_version)
    asap_ids_df = read_google_sheet("1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc", tab_name=cde_version)
    asap_ids_df = asap_ids_df[asap_ids_df["Required"] == "Assigned"]
    asap_ids_schema = asap_ids_df[["Table", "Field"]]
    
    updated_meta_tables = update_meta_tables_with_asap_ids(
        meta_tables=meta_tables,
        dataset_id = dataset_id,
        source=source,
        team=team,
        id_mappers=id_mappers,
        asap_ids_schema=asap_ids_schema)
    
    logging.info("ASAP IDs injected successfully")
    
    # ---- Generating contents of file_metadata/ ----
    logging.info("Generating file metadata from GCP raw bucket...")
    
    file_metadata_path = dataset_dir / "file_metadata"
    file_metadata_path.mkdir(exist_ok=True)
    
    raw_bucket_name = f"asap-raw-team-{dataset_id}"
    
    gen_raw_bucket_summary(
        raw_bucket_name=raw_bucket_name,
        dl_path=file_metadata_path,
        dataset_name=dataset_id,
        flatten=False   # TODO: confirm if this is ever True?
    )
    
    if "spatial" in modality.lower():
        gen_spatial_bucket_summary(
            raw_bucket_name=raw_bucket_name,
            dl_path=file_metadata_path,
            dataset_name=dataset_id
        )
        
    make_file_metadata(
        ds_path=dataset_dir,
        dl_path=file_metadata_path,
        data_df=updated_meta_tables["DATA"],
        spatial=("spatial" in modality.lower())
    )
        
    logging.info(f"File metadata summaries saved to [{file_metadata_path}]")
    
    # ---- Merging file metadata with DATA table ----  
    logging.info("Merging file metadata with DATA table...")

  
    updated_meta_tables["DATA"] = update_data_table_with_gcp_uri(
        data_df=updated_meta_tables["DATA"],
        ds_path=dataset_dir
    )
    
    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if "spatial" in modality.lower():
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_id)
        if not spatial_subtype == "cosmx":
            updated_meta_tables["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=updated_meta_tables["SPATIAL"],
                ds_path=dataset_dir,
                spatial_subtype=spatial_subtype
            )
    
    logging.info("File metadata merged with DATA table")
    
    # ---- Inserting DOI/project informatioin into STUDY ----
    logging.info("Injecting DOI and project metadata into STUDY table...")
    
    updated_meta_tables["STUDY"] = update_study_table_with_doi(
        study_df=updated_meta_tables["STUDY"],
        ds_path=dataset_dir
    )
    
    logging.info("DOI and project metadata injected into STUDY table")
    
    #  ---- Exporting release metadata ----
    out_dir = dataset_dir / "metadata" / "release" / release_version
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Exporting release metadata tables to [{out_dir}]...")
    
    for table_name, df in updated_meta_tables.items():
        out_path = out_dir / f"{table_name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")

    version_info = f"CDE_VERSION={cde_version}\nRELEASE_VERSION={release_version}\n"
    (out_dir / "VERSION").write_text(version_info)
    
    logging.info(f"Release metadata saved: {len(updated_meta_tables)} tables written to [{out_dir}]")

    return None


# The following are the older source-specific prep_release_metadata_* functionality
# ----


def create_metadata_package(
    dfs: dict[str, pd.DataFrame], metadata_path: str | Path, schema_version: str
):

    metadata_path = Path(metadata_path)
    final_metadata_path = os.path.join(metadata_path, schema_version)
    os.makedirs(final_metadata_path, exist_ok=True)

    export_meta_tables(dfs, final_metadata_path)
    # export_meta_tables(dfs, metadata_path)
    write_version(schema_version, os.path.join(metadata_path, "cde_version"))
    write_version(schema_version, os.path.join(final_metadata_path, "cde_version"))



# !!!NOTE!!! This was used up to and including the December 2025 release. 
# Importantly, it includes ID generation which is now handled separately.
### this is a wrapper to call source specific prep_release_metadata_* functions
def old_prep_release_metadata(
    ds_path: str | Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
    source: str = "pmdbs",
    flatten: bool = False,
):
    """
    prepare the metadata for a release.  This includes:
    - mapping to ASAP IDs
    - adding file metadata
    - updating the mappers

    """
    ds_path = Path(ds_path)

    if source == "pmdbs":
        prep_release_metadata_pmdbs(
            ds_path, schema_version, map_path, suffix, spatial, proteomics, flatten
        )
    elif source == "mouse":
        prep_release_metadata_mouse(
            ds_path, schema_version, map_path, suffix, spatial, proteomics, flatten
        )
    elif source in ["cell", "invitro", "ipsc"]:
        prep_release_metadata_cell(
            ds_path, schema_version, map_path, suffix, proteomics, flatten
        )
    else:
        raise ValueError(f"old_prep_release_metadata: Unknown source {source}")


def prep_release_metadata_cell(
    ds_path: str | Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    proteomics: bool = False,
    flatten: bool = False,
    map_only: bool = False,
):
    ds_path = Path(ds_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    print(asap_ids_schema)

    datasetid_mapper, cellid_mapper, sampleid_mapper = load_cell_id_mappers(
        map_path, suffix
    )

    mdata_path = os.path.join(ds_path, "metadata", schema_version)
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    req_tables = PROTEOMICS_TABLES if proteomics else CELL_TABLES

    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    if not map_only:
        datasetid_mapper, cellid_mapper, sampleid_mapper = update_cell_id_mappers(
            dfs["CELL"],
            dfs["SAMPLE"],
            dataset_name,
            datasetid_mapper,
            cellid_mapper,
            sampleid_mapper,
        )

    dfs = update_cell_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        cellid_mapper,
        sampleid_mapper,
        table_names,
    )

    file_metadata_path = os.path.join(ds_path, "file_metadata")
    os.makedirs(file_metadata_path, exist_ok=True)

    gen_raw_bucket_summary(
        raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
    )

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=False)

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    # HACK:
    # need to change file_metadata so artifacts.csv and (eventually curated_files.csv) point to curated bucket

    # export the tables to the metadata directory in a release subdirectory
    out_dir = os.path.join(ds_path, "metadata/release")
    os.makedirs(out_dir, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, os.path.join(out_dir, "cde_version"))

    if not map_only:
        export_map_path = map_path  # os.path.join(root_path, "asap-ids/temp")
        export_cell_id_mappers(
            export_map_path,
            suffix,
            datasetid_mapper,
            cellid_mapper,
            sampleid_mapper,
        )


def prep_release_metadata_mouse(
    ds_path: str | Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
    flatten: bool = False,
    map_only: bool = False,
):
    ds_path = Path(ds_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    datasetid_mapper, mouseid_mapper, sampleid_mapper = load_mouse_id_mappers(
        map_path, suffix
    )

    # os.makedirs(ds_path, exist_ok=True)
    mdata_path = os.path.join(ds_path, "metadata", schema_version)

    # this is broken due to deprication of the pathlib...
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    # TODO: make this accomodate proteomics
    req_tables = MOUSE_TABLES.copy()

    if spatial:
        req_tables.append("SPATIAL")

    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    if not map_only:
        datasetid_mapper, mouseid_mapper, sampleid_mapper = update_mouse_id_mappers(
            dfs["MOUSE"],
            dfs["SAMPLE"],
            dataset_name,
            datasetid_mapper,
            mouseid_mapper,
            sampleid_mapper,
        )

    dfs = update_mouse_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        mouseid_mapper,
        sampleid_mapper,
        table_names,
    )

    file_metadata_path = os.path.join(ds_path, "file_metadata")
    os.makedirs(file_metadata_path, exist_ok=True)

    print(
        f"release_util: Generating raw bucket summary for {raw_bucket_name}, flatten={flatten}    "
    )
    gen_raw_bucket_summary(
        raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
    )

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=spatial)

    if spatial:
        gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
    
    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if spatial:
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_name)
        if not spatial_subtype == "cosmx":
            dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=dfs["SPATIAL"],
                ds_path=ds_path,
                spatial_subtype=spatial_subtype
            )

    # HACK:
    # need to change file_metadata so artifacts.csv and (eventually curated_files.csv) point to curated bucket

    # export the tables to the metadata directory in a release subdirectory
    out_dir = os.path.join(ds_path, "metadata/release")
    os.makedirs(out_dir, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, os.path.join(out_dir, "cde_version"))

    if not map_only:
        export_map_path = map_path  # os.path.join(root_path, "asap-ids/temp")
        export_mouse_id_mappers(
            export_map_path,
            suffix,
            datasetid_mapper,
            mouseid_mapper,
            sampleid_mapper,
        )


def prep_release_metadata_pmdbs(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
    flatten: bool = False,
    map_only: bool = False,
):
    ds_path = Path(ds_path)
    map_path = Path(map_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    visium = "geomx" not in dataset_name

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    ) = load_pmdbs_id_mappers(map_path, suffix)

    mdata_path = os.path.join(ds_path, "metadata", schema_version)
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    # TODO: make this accomodate proteomics
    req_tables = MOUSE_TABLES.copy()

    req_tables = PMDBS_TABLES.copy()
    if spatial:
        req_tables.append("SPATIAL")

    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    if not map_only:
        (
            datasetid_mapper,
            subjectid_mapper,
            sampleid_mapper,
            gp2id_mapper,
            sourceid_mapper,
        ) = update_pmdbs_id_mappers(
            dfs["CLINPATH"],
            dfs["SAMPLE"],
            dataset_name,
            datasetid_mapper,
            subjectid_mapper,
            sampleid_mapper,
            gp2id_mapper,
            sourceid_mapper,
        )

    dfs = update_pmdbs_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
        pmdbs_tables=table_names,
    )

    file_metadata_path = os.path.join(ds_path, "file_metadata")
    os.makedirs(file_metadata_path, exist_ok=True)

    gen_raw_bucket_summary(
        raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
    )
    if spatial:
        gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=spatial)

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if spatial:
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_name)
        if not spatial_subtype == "cosmx":
            dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=dfs["SPATIAL"],
                ds_path=ds_path,
                spatial_subtype=spatial_subtype
                )

    # export the tables to the metadata directory in a release subdirectory
    out_dir = os.path.join(ds_path, "metadata", "release")
    os.makedirs(out_dir, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, os.path.join(out_dir, "cde_version"))

    if not map_only:
        export_map_path = map_path  # / "asap-ids/master"
        export_pmdbs_id_mappers(
            map_path,
            suffix,
            datasetid_mapper,
            subjectid_mapper,
            sampleid_mapper,
            gp2id_mapper,
            sourceid_mapper,
        )


def get_crn_release_metadata(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
    source: str = "pmdbs",
):
    """
    only maps by default
    """
    ds_path = Path(ds_path)
    map_path = Path(map_path)

    if source == "pmdbs":
        dfs = get_release_metadata_pmdbs(
            ds_path,
            schema_version,
            map_path,
            suffix,
            spatial=spatial,
            proteomics=proteomics,
        )
    elif source == "human ":
        dfs = get_release_metadata_human(
            ds_path,
            schema_version,
            map_path,
            suffix,
            spatial=spatial,
            proteomics=proteomics,
        )
    elif source == "mouse":
        dfs = get_release_metadata_mouse(
            ds_path,
            schema_version,
            map_path,
            suffix,
            spatial=spatial,
            proteomics=proteomics,
        )

    elif source in ["cell", "invitro", "ipsc"]:
        dfs = get_release_metadata_cell(
            ds_path, schema_version, map_path, suffix, proteomics=proteomics
        )
    elif source == "proteomics":
        dfs = get_release_metadata_cell(
            ds_path, schema_version, map_path, suffix, proteomics=True
        )
    else:
        raise ValueError(f"get_crn_release_metadata: Unknown source {source}")

    return dfs


def get_release_metadata_cell(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    proteomics: bool = False,
) -> dict:
    ds_path = Path(ds_path)
    map_path = Path(map_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    datasetid_mapper, cellid_mapper, sampleid_mapper = load_cell_id_mappers(
        map_path, suffix
    )

    mdata_path = os.path.join(ds_path, "metadata", schema_version)
    tables = [table for table in os.listdir(mdata_path) if table.endswith(".csv")]

    req_tables = CELL_TABLES if not proteomics else PROTEOMICS_TABLES
    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    dfs = update_cell_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        cellid_mapper,
        sampleid_mapper,
        table_names,
    )

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    return dfs


def get_release_metadata_mouse(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
) -> dict:
    ds_path = Path(ds_path)
    map_path = Path(map_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    visium = "geomx" not in dataset_name

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]
    datasetid_mapper, mouseid_mapper, sampleid_mapper = load_mouse_id_mappers(
        map_path, suffix
    )

    mdata_path = os.path.join(ds_path, "metadata", schema_version)
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    req_tables = MOUSE_TABLES.copy()
    if spatial:
        req_tables.append("SPATIAL")
    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    dfs = update_mouse_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        mouseid_mapper,
        sampleid_mapper,
        table_names,
    )

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if spatial:
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_name)
        if not spatial_subtype == "cosmx":
            dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=dfs["SPATIAL"],
                ds_path=ds_path,
                spatial_subtype=spatial_subtype
                )

    # TODO add proteoimics mouse stuff here

    return dfs


def get_release_metadata_pmdbs(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
) -> dict:
    ds_path = Path(ds_path)
    map_path = Path(map_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    visium = "geomx" not in dataset_name

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    ) = load_pmdbs_id_mappers(map_path, suffix)

    if schema_version == "v2.1":
        mdata_path = os.path.join(ds_path, "metadata", "v2")
    else:
        mdata_path = os.path.join(ds_path, "metadata", schema_version)
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    req_tables = PMDBS_TABLES.copy()
    if spatial:
        req_tables.append("SPATIAL")

    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    dfs = update_pmdbs_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
        pmdbs_tables=table_names,
    )

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if spatial:
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_name)
        if not spatial_subtype == "cosmx":
            dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=dfs["SPATIAL"],
                ds_path=ds_path,
                spatial_subtype=spatial_subtype
                )

    # TODO add proteoimics pmdbs stuff here

    return dfs


# TODO: add non PMDBS human wrinkles now
def get_release_metadata_human(
    ds_path: str | Path,
    schema_version: str,
    map_path: str | Path,
    suffix: str,
    spatial: bool = False,
    proteomics: bool = False,
) -> dict:

    ds_path = Path(ds_path)
    map_path = Path(map_path)

    dataset_name = ds_path.name
    print(f"release_util: Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

    visium = "geomx" not in dataset_name

    CDE = read_CDE(schema_version)
    asap_ids_df = read_CDE_asap_ids()
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    ) = load_pmdbs_id_mappers(map_path, suffix)

    if schema_version == "v2.1":
        mdata_path = os.path.join(ds_path, "metadata", "v2")
    else:
        mdata_path = os.path.join(ds_path, "metadata", schema_version)

    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    req_tables = PMDBS_TABLES.copy()
    if spatial:
        req_tables.append("SPATIAL")

    table_names = [
        os.path.splitext(os.path.basename(table))[0]
        for table in tables
        if os.path.splitext(os.path.basename(table))[0] in req_tables
    ]

    dfs = load_tables(mdata_path, table_names)

    dfs = update_pmdbs_meta_tables_with_asap_ids(
        dfs,
        dataset_name,
        asap_ids_schema,
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
        pmdbs_tables=table_names,
    )

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)

    # There is no SPATIAL table for CosMx datasets.
    # For CDE >v4.0 this needs to be modified for visium and geomx datasets as well as there is no longer SPATIAL table
    if spatial:
        spatial_subtype = get_spatial_subtype_from_dataset_id(dataset_name)
        if not spatial_subtype == "cosmx":
            dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=dfs["SPATIAL"],
                ds_path=ds_path,
                spatial_subtype=spatial_subtype
                )

    # TODO add proteoimics pmdbs stuff here

    return dfs


def load_and_process_table(
    table_name: str,
    tables_path: str | Path,
    cde_df: pd.DataFrame,
    print_log: bool = False,
):
    """
    Load and process a table from a given path according to a schema.

    Args:
        table_name (str): Name of the table to load and process.
        tables_path (Path): Path to the directory containing the tables.
        cde_df (pd.DataFrame): DataFrame containing the schema.
        print_log (bool, optional): Whether to print the validation log. Defaults to False.

    Returns:
        pd.DataFrame: Processed table.

    """
    table_path = Path(tables_path)

    table_path = os.path.join(tables_path, f"{table_name}.csv")
    schema = cde_df[cde_df["Table"] == table_name]
    report = ReportCollector(destination="NA")
    # full_table, report = validate_table(df.copy(), table_name, schema, report)
    if os.path.exists(table_path):
        df = read_meta_table(table_path)
    else:
        print(f"{table_name} table not found.  need to construct")
        df = pd.DataFrame(columns=schema["Field"])

    if print_log:
        report.print_log()
    df, df_aux, _ = process_table(df, table_name, CDE)
    return df, df_aux


def process_schema(
    tables: list[str],
    cde_version: str | Path,
    source_path: str | Path,
    export_path: str | Path | None = None,
    print_log: bool = False,
):
    """
    Load and process tables from a given path according to a schema.

    Args:
        tables (list[str]): List of table names to process.
        tables_path (Path): Path to the directory containing the tables.
        cde_df (pd.DataFrame): DataFrame containing the schema.
        print_log (bool, optional): Whether to print the validation log. Defaults to False.

    Returns:
        dict: Dictionary containing the processed tables.
        dict: Dictionary containing the auxiliary tables.
    """

    source_path = Path(source_path)
    if export_path is not None:
        export_path = Path(export_path)

    # load CDE
    cde_df = read_CDE(cde_version)

    # load and process tables
    tables_dict = {}
    aux_tables_dict = {}
    for table in tables:
        df, df_aux = load_and_process_table(table, source_path, cde_df, print_log)
        tables_dict[table] = df
        aux_tables_dict[table] = df_aux

        if export_path is not None:
            df.to_csv(os.path.join(export_path, f"{table}.csv"), index=False)
            if not df_aux.empty:
                df_aux.to_csv(
                    os.path.join(export_path, f"{table}_auxiliary.csv"), index=False
                )

    return tables_dict, aux_tables_dict


def ingest_ds_info_doc(
    intake_doc: Path | str, ds_path: Path | str, doc_path: Path | str
):
    """
    Ingest the dataset information from the docx file and export to json for zenodo upload.
    """
    intake_doc = Path(intake_doc)
    ds_path = Path(ds_path)
    doc_path = Path(doc_path)

    # should read this from ds_path/version
    # just read in as text
    with open(os.path.join(ds_path, "version"), "r") as f:
        ds_ver = f.read().strip()
    # ds_ver = "v2.0"

    # Load the document
    document = docx.Document(intake_doc)
    table_names = ["affiliations", "datasets", "projects", "extra1", "extra2"]
    for name, table in zip(table_names, document.tables):
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        # Assuming the first row is the header
        if name == "affiliations":
            fields = table_data[0]
            data = table_data[1:]
            print("made affiliation table")
        elif name == "datasets":
            dataset_title = table_data[0][1].strip().replace("\n", " ")
            dataset_description = table_data[1][1].strip().replace("\n", " ")
            print("got dataset title/description")
        elif name == "projects":
            project_title = table_data[0][1].strip().replace("\n", " ")
            project_description = table_data[1][1].strip().replace("\n", " ")
            print("got project title/description")
        else:
            print("what is this extra thing?")
            print(table_data)
    title = dataset_title
    upload_type = "dataset"

    creators = []
    for indiv in data:
        name = f"{indiv[0].strip()}, {indiv[1].strip()}"  # , ".join(indiv[:2])
        affiliation = indiv[2]
        oricid = indiv[3]

        to_append = {"name": name}
        if affiliation.strip() == "":
            affiliation = None
        else:
            to_append["affiliation"] = affiliation

        if oricid.strip() == "":
            oricid = None
        else:
            to_append["orcid"] = oricid
        creators.append(to_append)

    description = dataset_description
    # ASAP
    communities = [{"identifier": "asaphub"}]
    # version
    version = ds_ver
    # publication_date
    publication_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    export_data = {
        "metadata": {
            "title": title,
            "upload_type": upload_type,
            "description": description,
            "creators": creators,
            "communities": communities,
            "version": version,
            "publication_type": "other",
            "resource_type": "dataset",
            "publication_date": publication_date,
        }
    }

    # dump json
    doi_path = os.path.join(ds_path, "DOI")

    with open(os.path.join(doi_path, f"{long_dataset_name}.json"), "w") as f:
        json.dump(export_data, f, indent=4)

    ##  we've got everything now lets .md file to upload
    md_content = f"# {title}\n\n __Dataset Description:__  {description_}\n\n"
    md_content += f" ### ASAP Team: Team {team.capitalize()}:\n\n > *Project:* __{project_title}__:{project_description}\n\n"
    md_content += f"\n\n_____________________\n\n"
    md_content += f"*ASAP CRN Cloud Dataset Name:* {long_dataset_name}\n"
    ## add creators as "Authors:"
    md_content += f"\n\n### Authors:\n\n"
    for creator in creators:
        md_content += f"* {creator['name']}"
        if "orcid" in creator:
            # format as link
            md_content += (
                f"; [ORCID:{creator['orcid'].split("/")[-1]}]({creator['orcid']})"
            )
        if "affiliation" in creator:
            md_content += f"; {creator['affiliation']}"
        md_content += "\n"

    # neeed to add metadata informaiton...
    ## dump md_content
    # write to a text file
    doi_path = os.path.join(ds_path, "DOI")
    os.makedirs(doi_path, exist_ok=True)

    long_dataset_name = ds_path.name
    with open(os.path.join(doi_path, f"{long_dataset_name}.md"), "w") as f:
        f.write(md_content)

    # ## save a simple table to update STUDY table
    export_dict = {
        "project_name": f"{project_title.strip()}",  # protect the parkionson's apostrophe
        "project_description": f"{project_description.strip()}",
        "dataset_title": f"{dataset_title.strip()}",
        "dataset_description": f"{dataset_description.strip()}",
    }

    with open(os.path.join(doi_path, f"STUDY_fix.json"), "w") as f:
        json.dump(export_dict, f, indent=4)
    df = pd.DataFrame(export_dict, index=[0])
    df.to_csv(os.path.join(doi_path, f"{long_dataset_name}.csv"), index=False)


# fix STUDY
def fix_study_table(metadata_path: str | Path, doi_path: str | Path | None = None):
    """
    Update the STUDY table with the information from the DOI folder.
    """
    metadata_path = Path(metadata_path)
    if doi_path is not None:
        doi_path = Path(doi_path)

    table = "STUDY"
    STUDY = read_meta_table(os.path.join(metadata_path, f"{table}.csv"))

    if doi_path is None:
        doi_path = os.path.join(metadata_path.parent, "DOI")
    else:
        doi_path = Path(doi_path)

    STUDY_fix = pd.read_csv(os.path.join(doi_path, f"{metadata_path.parent.name}.csv"))

    STUDY["project_name"] = STUDY_fix["project_name"]
    STUDY["project_description"] = STUDY_fix["project_description"]
    STUDY["dataset_title"] = STUDY_fix["dataset_title"]
    STUDY["dataset_description"] = STUDY_fix["dataset_description"]

    # export STUDY
    STUDY.to_csv(os.path.join(metadata_path, "STUDY.csv"), index=False)


def get_stats_table(dfs: dict[pd.DataFrame], 
                    organism: str = None,
                    sample_source: str = None,
                    ) -> None:
    """
    Sorts organism and sample_source to get dataset stats.
    Note: starting release v4.0.1 (CDE v4.1) it's required to define organism and sample_source
          using controlled vocabularies from the CDE Google Spreadsheet tab ValidCategories.
    
    Parameters
    ----------
    dfs
        Dictionary of dataframes containing the metadata tables for a dataset release
    organism
        Organism name (e.g., "Human", "Mouse")
    sample_source
        Sample source (e.g., "Brain")
    """

    if organism == "Human":
        if sample_source == "Brain":
            return get_stat_tabs_pmdbs(dfs)
        elif sample_source in ["Cell lines", "Cell", "iPSC", "InVitro"]:
            return get_stat_tabs_cell(dfs)
        else:
            return get_stat_tabs_human_non_brain(dfs)
        
    elif organism == "Mouse":
        if sample_source == "Brain":
            return get_stat_tabs_mouse(dfs)
        elif sample_source in ["Cell lines", "Cell", "iPSC", "InVitro"]:
            return get_stat_tabs_cell(dfs)
        else:
            return get_stat_tabs_mouse_non_brain(dfs)
    
    else:
        raise ValueError(f"get_stats_table: Unexpected categories: organism {organism}, sample_source {sample_source}")


_brain_region_coder = {
    "Anterior_Cingulate_Gyrus": "ACG",
    "Anterior Cingulate Gyrus": "ACG",
    "Caudate": "CAU",
    "Putamen": "PUT",
    "Hippocampus": "HIP",
    "Substantia nigra": "SN",
    "Amygdala": "AMG",
    "Substantia_Nigra ": "SN",
    "Substantia_Nigra": "SN",
    "AMY": "AMG",  # team Jakobsson
    "SND": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNV": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "VTA": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNM": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNL": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "Prefrontal Cortex": "PFC",
    "Prefrontal cortex": "PFC",
    "inferior parietal lobe": "IPL",
    "Inferior Parietal Lobe": "IPL",
    "Anterior_Cingulate_Cortex": "ACC",
    "Anterior Cingulate Cortex": "ACC",
    "Antaerior Cortex": "ACC",
    "Antaerior Cingulate": "ACC",
    "Anterior_cingulate_cortex": "ACC",
    "Anterior cingulate cortex": "ACC",
    "Antaerior cortex": "ACC",
    "Antaerior cingulate": "ACC",
    "Frontal Cortex": "F_CTX",
    "Frontal_ctx": "F_CTX",
    "Frontal cortex": "F_CTX",
    "Frontal_Cortex": "F_CTX",
    "frontal_cortex": "F_CTX",
    "Frontal_Lobe": "F_CTX",
    "Frontal lobe": "F_CTX",
    "Parietal Cortex": "P_CTX",
    "Parietal cortex": "P_CTX",
    "Parietal_Cortex": "P_CTX",
    "Parietal lobe": "P_CTX",
    "Parietal_ctx": "P_CTX",
    "Parietal": "P_CTX",
    "Cingulate Cortex": "C_CTX",
    "Cingulate cortex": "C_CTX",
    "Cingulate_Cortex": "C_CTX",
    "Cingulate gyrus": "C_CTX",
    "temporal_ctx": "T_CTX",
    "Temporal Cortex": "T_CTX",
    "Temporal_ctx": "T_CTX",
    "Temporal cortex": "T_CTX",
    "Middle_Frontal_Gyrus": "MFG",
    "Middle frontal gyrus": "MFG",
    "Middle Frontal Gyrus": "MFG",
    "Middle Temporal Gyrus": "MTG",
    "Middle temporal gyrus": "MTG",
    "Parahippocampal Gyrus": "PARA",
}

_region_titles = {
    "ACG": "Anterior Cingulate Gyrus",
    "CAU": "Caudate",
    "PUT": "Putamen",
    "HIP": "Hippocampus",
    "SN": "Substantia Nigra",
    "AMG": "Amygdala",
    "PFC": "Prefrontal Cortex",
    "IPL": "Inferior Parietal Lobe",
    "ACC": "Antaerior Cingulate Cortex",
    "F_CTX": "Frontal Cortex",
    "P_CTX": "Parietal Cortex",
    "C_CTX": "Cingulate Cortex",
    "T_CTX": "Temporal Cortex",
    "MFG": "Middle Frontal Gyrus",
    "MTG": "Middle Temporal Gyrus",
    "PARA": "Para-Hippocampal Gyrus",
}


def make_stats_df_pmdbs(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """ """
    # do joins to get the stats we need.
    # first JOIN SAMPLE and CONDITION on "condition_id" how=left to get our "intervention_id" or PD / control
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "replicate_count",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "assay_type",
        "condition_id",
    ]

    subject_cols = [
        "ASAP_subject_id",
        "source_subject_id",
        "biobank_name",
        "sex",
        "race",
        "primary_diagnosis",
        "primary_diagnosis_text",
    ]

    pmdbs_cols = [
        "ASAP_sample_id",
        "brain_region",
        "hemisphere",
        "region_level_1",
        "region_level_2",
        "region_level_3",
    ]

    condition_cols = [
        "condition_id",
        "intervention_name",
    ]

    if "age_at_collection" in dfs["SUBJECT"].columns:
        subject_cols.append("age_at_collection")
    elif "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")
    else:
        raise ValueError(f"get_stats_pmdbs: No age_at_collection column found in SUBJECT or SAMPLE")

    SAMPLE_ = dfs["SAMPLE"][sample_cols]

    if "gp2_phenotype" in dfs["SUBJECT"].columns:
        subject_cols.append("gp2_phenotype")
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
    else:
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
        SUBJECT_["gp2_phenotype"] = SUBJECT_["primary_diagnosis"]

    PMDBS_ = dfs["PMDBS"][pmdbs_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")

    # then JOIN the result with SUBJECT on "ASAP_subject_id" how=left to get "age_at_collection", "sex", "primary_diagnosis"
    df = pd.merge(df, SUBJECT_, on="ASAP_subject_id", how="left")

    # then JOIN the result with PMDBS on "ASAP_subject_id" how=left to get "brain_region"
    df = (
        pd.merge(df, PMDBS_, on="ASAP_sample_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_pmdbs(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_pmdbs(dfs)
    report = get_stats_pmdbs(df)
    return report, df


def get_stats_pmdbs(df: pd.DataFrame) -> dict:
    # should be the same as df.shape[0]
    n_samples = df[["ASAP_sample_id", "replicate"]].drop_duplicates().shape[0]
    n_subjects = df["ASAP_subject_id"].nunique()

    # get stats for the dataset
    # 0. total number of samples
    # SAMPLE wise
    sw_df = df[
        [
            "ASAP_sample_id",
            "ASAP_subject_id",
            "replicate",
            "gp2_phenotype",
            "primary_diagnosis",
            "age_at_collection",
            "brain_region",
            "condition_id",
            "sex",
        ]
    ].drop_duplicates()

    print(f"shape df: {df.shape}, shape sw_df: {sw_df.shape}")

    brain_code = (
        sw_df["brain_region"].replace(_brain_region_coder).value_counts().to_dict()
    )
    brain_region = (
        sw_df["brain_region"]
        .replace(_brain_region_coder)
        .map(_region_titles)
        .value_counts()
        .to_dict()
    )

    age_at_collection = (
        sw_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    )
    sex = (sw_df["sex"].value_counts().to_dict(),)
    PD_status = (sw_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sw_df["condition_id"].value_counts().to_dict(),)
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    # does this copy the values?
    samples = dict(
        n_samples=n_samples,
        brain_region=brain_region,
        brain_code=brain_code,
        PD_status=PD_status,
        condition_id=condition_id,
        age_at_collection=age,
        sex=sex,
    )

    # SUBJECT wise
    sw_df = df[
        [
            "ASAP_subject_id",
            "gp2_phenotype",
            "primary_diagnosis",
            "sex",
            "age_at_collection",
            "condition_id",
        ]
    ].drop_duplicates()
    # fill in primary_diagnosis if gp2_phenotype is not in df
    PD_status = (sw_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sw_df["condition_id"].value_counts().to_dict(),)
    diagnosis = (sw_df["primary_diagnosis"].value_counts().to_dict(),)
    sex = (sw_df["sex"].value_counts().to_dict(),)
    age_at_collection = (
        sw_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    )

    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    subject = dict(
        n_subjects=n_subjects,
        PD_status=PD_status,
        condition_id=condition_id,
        diagnosis=diagnosis,
        age_at_collection=age,
        sex=sex,
    )

    report = dict(
        subject=subject,
        samples=samples,
    )
    # SAMPLE wise
    return report


def make_stats_df_human_non_brain(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a flat stats DataFrame for human non-brain datasets (e.g. Blood, Skin,
    Gastrointestinal tissue, etc.) by joining SAMPLE + CONDITION + SUBJECT.

    Unlike make_stats_df_pmdbs there is no PMDBS or CLINPATH join and therefore
    no brain-region columns.  Compatible with both CDE v4.1 (age_at_collection in
    SAMPLE, primary_diagnosis removed) and older schema versions (age_at_collection
    in SUBJECT, primary_diagnosis present).
    """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "condition_id",
    ]

    # replicate_count and assay_type were removed in CDE v4.1; include only if present
    for optional_col in ["replicate_count", "assay_type"]:
        if optional_col in dfs["SAMPLE"].columns:
            sample_cols.append(optional_col)

    subject_cols = [
        "ASAP_subject_id",
        "source_subject_id",
        "biobank_name",
        "sex",
    ]

    # race is Human-specific and required in v4.1 SUBJECT; include if present
    if "race" in dfs["SUBJECT"].columns:
        subject_cols.append("race")

    # primary_diagnosis / primary_diagnosis_text existed pre-v4.1
    for legacy_col in ["primary_diagnosis", "primary_diagnosis_text"]:
        if legacy_col in dfs["SUBJECT"].columns:
            subject_cols.append(legacy_col)

    condition_cols = [
        "condition_id",
        "intervention_name",
    ]

    # age_at_collection: in SUBJECT for pre-v4.1 datasets, in SAMPLE for v4.1+
    if "age_at_collection" in dfs["SUBJECT"].columns:
        subject_cols.append("age_at_collection")
    elif "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")
    else:
        raise ValueError(f"get_stats_human_non_brain: No age_at_collection column found in SUBJECT or SAMPLE")

    SAMPLE_ = dfs["SAMPLE"][sample_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    if "gp2_phenotype" in dfs["SUBJECT"].columns:
        subject_cols.append("gp2_phenotype")
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
    else:
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
        if "primary_diagnosis" in SUBJECT_.columns:
            SUBJECT_ = SUBJECT_.copy()
            SUBJECT_["gp2_phenotype"] = SUBJECT_["primary_diagnosis"]
        else:
            SUBJECT_ = SUBJECT_.copy()
            SUBJECT_["gp2_phenotype"] = "NA"

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")
    df = (
        pd.merge(df, SUBJECT_, on="ASAP_subject_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_human_non_brain(dfs: dict[pd.DataFrame]):
    """
    Compute summary statistics for human non-brain datasets (e.g. Blood, Skin,
    Gastrointestinal tissue, etc.).

    Returns
    -------
    report : dict
        Nested statistics dictionary (same structure as get_stat_tabs_pmdbs but
        without brain_region / brain_code entries).
    df : pd.DataFrame
        Flat stats DataFrame produced by make_stats_df_human_non_brain.
    """
    df = make_stats_df_human_non_brain(dfs)
    report = get_stats_human_non_brain(df)
    return report, df


def get_stats_human_non_brain(df: pd.DataFrame) -> dict:
    """
    Compile summary statistics for human non-brain datasets from a flat stats
    DataFrame (as returned by make_stats_df_human_non_brain).

    The report structure mirrors get_stats_pmdbs but omits brain_region and
    brain_code since those are not applicable outside brain sample sources.
    """
    n_samples = df[["ASAP_sample_id", "replicate"]].drop_duplicates().shape[0]
    n_subjects = df["ASAP_subject_id"].nunique()

    # --- Sample-level stats (one row per unique sample × replicate) ---
    sw_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "replicate",
        "gp2_phenotype",
        "age_at_collection",
        "condition_id",
        "sex",
    ]
    sw_df = df[[c for c in sw_cols if c in df.columns]].drop_duplicates()

    print(f"shape df: {df.shape}, shape sw_df: {sw_df.shape}")

    PD_status = (sw_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sw_df["condition_id"].value_counts().to_dict(),)
    sex = (sw_df["sex"].value_counts().to_dict(),)

    age_at_collection = sw_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    samples = dict(
        n_samples=n_samples,
        PD_status=PD_status,
        condition_id=condition_id,
        age_at_collection=age,
        sex=sex,
    )

    # --- Subject-level stats (one row per unique subject) ---
    sub_cols = [
        "ASAP_subject_id",
        "gp2_phenotype",
        "sex",
        "age_at_collection",
        "condition_id",
    ]
    sub_df = df[[c for c in sub_cols if c in df.columns]].drop_duplicates()

    PD_status = (sub_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sub_df["condition_id"].value_counts().to_dict(),)
    sex = (sub_df["sex"].value_counts().to_dict(),)

    age_at_collection = sub_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    subject = dict(
        n_subjects=n_subjects,
        PD_status=PD_status,
        condition_id=condition_id,
        age_at_collection=age,
        sex=sex,
    )

    report = dict(
        subject=subject,
        samples=samples,
    )
    return report


def get_stat_tabs_mouse_non_brain(dfs: dict[pd.DataFrame]):
    """
    Compute summary statistics for mouse non-brain datasets.

    Mouse datasets do not carry brain-region columns regardless of sample source,
    so this is a direct alias for get_stat_tabs_mouse.
    """
    return get_stat_tabs_mouse(dfs)


def make_stats_df_cell(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """ """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_cell_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "replicate_count",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "assay_type",
        "condition_id",
    ]

    subject_cols = [
        "ASAP_cell_id",
        "cell_line",
    ]

    condition_cols = [
        "condition_id",
        "intervention_name",
        "intervention_id",
        "protocol_id",
        "intervention_aux_table",
    ]

    SAMPLE_ = dfs["SAMPLE"][sample_cols]

    SUBJECT_ = dfs["CELL"][subject_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")

    # then JOIN the result with SUBJECT on "ASAP_subject_id" how=left to get "age_at_collection", "sex", "primary_diagnosis"
    df = (
        pd.merge(df, SUBJECT_, on="ASAP_cell_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_cell(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_cell(dfs)
    report = get_stats_cell(df)
    return report, df


def get_stats_cell(df: pd.DataFrame) -> dict:
    """
    get stats for the dataset from the stats table (tab)
    """

    # collapse to remove replicates...
    uq_df = df[
        [
            "ASAP_sample_id",
            "condition_id",
        ]
    ].drop_duplicates()
    print(f"shape df: {df.shape}, shape suq_dfw_df: {uq_df.shape}")

    # get stats for the dataset
    N = uq_df["ASAP_sample_id"].nunique()
    condition_id = (uq_df["condition_id"].value_counts().to_dict(),)
    report = dict(
        N=N,
        condition_id=condition_id,
    )
    return report

def make_stats_df_mouse(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a flat stats DataFrame for mouse datasets by joining SAMPLE + CONDITION
    + SUBJECT (or the legacy MOUSE table for pre-v4.1 datasets).

    Schema changes in CDE v4.1:
    - MOUSE table removed; subject info (sex, strain) moved to the universal SUBJECT table.
    - ASAP_subject_id is the join key between SAMPLE and SUBJECT in v4.1.
    - ASAP_mouse_id is a secondary identifier now in SAMPLE (not the join key to SUBJECT).
    - age renamed to age_at_collection (now in SAMPLE).
    - replicate_count and assay_type removed from SAMPLE.
    """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "condition_id",
    ]

    # ASAP_subject_id: join key to universal SUBJECT table in v4.1+
    if "ASAP_subject_id" in dfs["SAMPLE"].columns:
        sample_cols.append("ASAP_subject_id")

    # replicate_count and assay_type were removed in CDE v4.1
    for optional_col in ["replicate_count", "assay_type"]:
        if optional_col in dfs["SAMPLE"].columns:
            sample_cols.append(optional_col)

    # ASAP_mouse_id: secondary ID, include for reference if present
    if "ASAP_mouse_id" in dfs["SAMPLE"].columns:
        sample_cols.append("ASAP_mouse_id")

    # age_at_collection: in SAMPLE for v4.1+; legacy datasets use age in MOUSE
    if "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")

    condition_cols = [
        "condition_id",
        "intervention_name",
    ]
    for optional_col in ["intervention_id", "protocol_id", "intervention_aux_table"]:
        if optional_col in dfs["CONDITION"].columns:
            condition_cols.append(optional_col)

    SAMPLE_ = dfs["SAMPLE"][sample_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")

    # v4.1+: MOUSE table removed, subject info now in universal SUBJECT table.
    # Join key is ASAP_subject_id (present in both SAMPLE and SUBJECT).
    # Pre-v4.1: MOUSE table exists; join key is ASAP_mouse_id.
    if "MOUSE" in dfs:
        subject_df = dfs["MOUSE"]
        subject_join_key = "ASAP_mouse_id"
        subject_cols = ["ASAP_mouse_id"]
        for col in ["sex", "strain", "age"]:
            if col in subject_df.columns:
                subject_cols.append(col)
    else:
        subject_df = dfs["SUBJECT"]
        subject_join_key = "ASAP_subject_id"
        if subject_join_key not in df.columns:
            raise ValueError(f"get_stats_mouse: Cannot join SAMPLE to SUBJECT: {subject_join_key} not found in SAMPLE")
        subject_cols = [subject_join_key]
        for col in ["sex", "strain"]:
            if col in subject_df.columns:
                subject_cols.append(col)

    SUBJECT_ = subject_df[subject_cols]

    df = (
        pd.merge(df, SUBJECT_, on=subject_join_key, how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_mouse(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_mouse(dfs)
    # get stats for the dataset
    # 0. total number of samples
    report = get_stats_mouse(df)
    return report, df


def get_stats_mouse(df: pd.DataFrame) -> dict:
    """
    Compile stats from stats table (tab).
    Compatible with CDE v4.1 (age_at_collection in SAMPLE) and pre-v4.1 (age in MOUSE).
    """
    # age field: age_at_collection for v4.1+, age for pre-v4.1
    if "age_at_collection" in df.columns:
        age_col = "age_at_collection"
    elif "age" in df.columns:
        age_col = "age"
    else:
        raise ValueError(f"get_stats_mouse: No age column found in stats DataFrame (expected 'age_at_collection' or 'age')")

    uq_cols = ["ASAP_sample_id", "condition_id", "sex", age_col]
    uq_df = df[[c for c in uq_cols if c in df.columns]].drop_duplicates()

    print(f"shape df: {df.shape}, shape uq_df: {uq_df.shape}")

    age_series = uq_df[age_col].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_series.mean():.1f}",
        median=f"{age_series.median():.1f}",
        max=f"{age_series.max():.1f}",
        min=f"{age_series.min():.1f}",
    )

    N = uq_df["ASAP_sample_id"].nunique()
    condition_id = (uq_df["condition_id"].value_counts().to_dict(),)
    sex = (uq_df["sex"].value_counts().to_dict(),)

    report = dict(
        N=N,
        condition_id=condition_id,
        age=age,
        sex=sex,
    )
    return report


def get_cohort_stats_table(dfs: dict[pd.DataFrame], source: str = None):
    """ """
    if source == "pmdbs":
        # get stats_df by dataset, concatenate and then get stats
        datasets = dfs["STUDY"]["ASAP_dataset_id"].unique()
        stat_df = pd.DataFrame()
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_pmdbs(dfs_)
            stat_df = pd.concat([stat_df, df])

        report = get_stats_pmdbs(stat_df)

        N_datasets = stat_df["ASAP_dataset_id"].nunique()
        N_teams = stat_df["ASAP_team_id"].nunique()
        report["N_datasets"] = N_datasets
        report["N_teams"] = N_teams

    elif source == "mouse":
        datasets = dfs["STUDY"]["ASAP_dataset_id"].unique()
        stat_df = pd.DataFrame()
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_mouse(dfs_)
            stat_df = pd.concat([stat_df, df])

        report = get_stats_mouse(stat_df)

        N_datasets = stat_df["ASAP_dataset_id"].nunique()
        N_teams = stat_df["ASAP_team_id"].nunique()
        report["N_datasets"] = N_datasets
        report["N_teams"] = N_teams
    else:
        raise ValueError(f"get_cohort_stats_table: Unknown source {source}")

    return report, stat_df
