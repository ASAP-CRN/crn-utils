import os
import sys
import logging
import pandas as pd
from pathlib import Path

from .util import (
    list_expected_metadata_tables,
    read_CDE,
    read_meta_table,
    read_CDE_asap_ids,
    export_meta_tables,
    load_tables,
    write_version,
)

from .asap_ids import (
    load_all_id_mappers,
    update_meta_tables_with_asap_ids,
    # The following are used by get_crn_release_metadata and get_release_metadata_* functions.
    # These are candidates for deprecation once those functions are retired.
    load_pmdbs_id_mappers,
    update_pmdbs_meta_tables_with_asap_ids,
    load_mouse_id_mappers,
    update_mouse_meta_tables_with_asap_ids,
    load_cell_id_mappers,
    update_cell_meta_tables_with_asap_ids,
)

from .file_metadata import (
    gen_bucket_summary,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
    gen_spatial_bucket_summary,
    make_file_metadata,
)
from .constants import *  # List of tables expected (CDE <= v4.0)
from .doi import update_study_table_with_doi
from .google_spreadsheets import read_google_sheet
from .validate import ReportCollector, process_table
from .asap_ids import normalize_source_for_ids

__all__ = [
    # Release metadata preparation
    "prep_release_metadata",
    "load_and_process_table",
    "process_schema",
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


# The following is a refactor of the main function to prepare metadata for a release.
# It combines the previous source-specific functions into a unified call,
# while removing the ID generation step which is now handled separately.
# Following PRs will continue to refactor/deprecate the previous workflow.
# ----
#TODO: Here functions normalize_source_for_ids and prep_release_metadata
#      work together to normalize source_for_ids.
#      It's STILL one of ["pmdbs", "mouse", "invitro"].
#      but now it's based on CDE ValidCategories (organism and sample_source).
#      This is a temporary hack for the Feb2026/March2026 releases which use PMDBS/MOUSE/CELL ASAP IDs
#      A fututre implementation will fully transition to general SUBJECT ASAP IDs.
def prep_release_metadata(dataset_id: str,
                          organism: str,
                          source: str,
                          assay: str,
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

    Required fields:
        - dataset_id: (e.g., "team-smith-pmdbs-sn-rnaseq")
        - organism: organism type of the dataset (e.g., "Human", "Mouse")
        - source: source type of the dataset (e.g., "Brain", "Fecal", "Cell lines", "iPSC")
        - assay: assay type of the dataset (e.g., "bulk_rna_seq", "single_nucleus_rna_seq")
        - cde_version: CDE schema version to prepare for (e.g., "v4.0")
        - release_version: CRN release version (e.g., "v4.0.1")
        - metadata_dir: Path to metadata directory
        - dataset_dir: Path to dataset directory
        - map_path: Path to master ID mappers

    """

    # normalize source
    source_for_ids = normalize_source_for_ids(
        dataset_id=dataset_id, 
        organism=organism, 
        source=source,
        release_version=release_version
    )

    # ---- Load metadata tables ----
    logging.info(f"Loading metadata tables from {metadata_dir}...")

    # TODO: This is a temporary heuristic as we move away from list_expected_metadata_table()
    present_files = [f.stem for f in metadata_dir.glob("*.csv")]
    expected_tables = [t for t in list_expected_metadata_tables() if t in present_files]
    meta_tables = load_tables(metadata_dir, expected_tables)

    logging.info(f"Loaded {len(meta_tables)} metadata tables: {', '.join(meta_tables.keys())}")

    # ---- Inserting ASAP IDs ----
    logging.info("Inserting ASAP IDs into metadata tables...")

    # Get CDE fields assigned by ASAP curators
    id_mappers = load_all_id_mappers(map_path, source_for_ids)
    # TODO: Replace hardcoded IDs by a local config ingestion
    asap_ids_df = read_google_sheet("1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc", tab_name=cde_version)
    asap_ids_df = asap_ids_df[asap_ids_df["Required"] == "Assigned"]
    asap_ids_schema = asap_ids_df[["Table", "Field"]]

    updated_meta_tables = update_meta_tables_with_asap_ids(
        meta_tables=meta_tables,
        dataset_id=dataset_id,
        id_mappers=id_mappers,
        asap_ids_schema=asap_ids_schema)

    logging.info("ASAP IDs injected successfully")

    # ---- Generating contents of file_metadata/ ----
    logging.info("Generating file metadata from GCP raw bucket...")

    file_metadata_path = dataset_dir / "file_metadata"
    file_metadata_path.mkdir(exist_ok=True)

    raw_bucket_name = f"asap-raw-{dataset_id}"

    gen_bucket_summary(
        dl_path=file_metadata_path,
        dataset_id=dataset_id,
        env_type="raw",
    )

    if "spatial" in assay.lower():
        gen_spatial_bucket_summary(
            dl_path=file_metadata_path,
            dataset_id=dataset_id
        )

    make_file_metadata(
        ds_path=dataset_dir,
        dl_path=file_metadata_path,
        data_df=updated_meta_tables["DATA"],
        spatial=("spatial" in assay.lower())
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
    if "spatial" in assay.lower():
        spatial_subtype = get_spatial_subtype_from_dataset_id(assay)
        if not spatial_subtype == "cosmx":
            updated_meta_tables["SPATIAL"] = update_spatial_table_with_gcp_uri(
                spatial_df=updated_meta_tables["SPATIAL"],
                ds_path=dataset_dir,
                spatial_subtype=spatial_subtype
            )

    logging.info("File metadata merged with DATA table")

    # ---- Inserting DOI/project information into STUDY ----
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

    # TODO add proteomics mouse stuff here

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

    # TODO add proteomics pmdbs stuff here

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

    # TODO add proteomics pmdbs stuff here

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
