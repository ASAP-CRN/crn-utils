import pandas as pd
from pathlib import Path
import os
import docx
import json
from numpy import nan as np_nan


from .util import (
    read_CDE,
    NULL,
    prep_table,
    read_meta_table,
    read_CDE_asap_ids,
    export_meta_tables,
    load_tables,
    write_version,
)

from .asap_ids import *
from .validate import validate_table, ReportCollector, process_table

from .checksums import extract_md5_from_details2, get_md5_hashes
from .bucket_util import authenticate_with_service_account
from .file_metadata import (
    gen_raw_bucket_summary,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
    gen_spatial_bucket_summary,
    make_file_metadata,
)
from .constants import *
from .doi import update_study_table_with_doi


__all__ = [
    "prep_sc_release_metadata",
    # "prep_release_metadata_mouse",
    # "prep_release_metadata_pmdbs",
    "get_crn_release_metadata",
    "load_and_process_table",
    "process_schema",
    "create_metadata_package",
    "get_stat_tabs_pmdbs",
    "get_stat_tabs_mouse",
    "get_stats_table",
    "get_cohort_stats_table",
]


def create_metadata_package(
    dfs: dict[str, pd.DataFrame], metadata_path: Path, schema_version: str
):

    final_metadata_path = metadata_path / schema_version
    if not final_metadata_path.exists():
        final_metadata_path.mkdir()

    export_meta_tables(dfs, final_metadata_path)
    # export_meta_tables(dfs, metadata_path)
    write_version(schema_version, metadata_path / "cde_version")
    write_version(schema_version, final_metadata_path / "cde_version")


def prep_sc_release_metadata(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    source: str = "pmdbs",
    flatten: bool = False,
):

    if source == "pmdbs":
        prep_release_metadata_pmdbs(
            ds_path, schema_version, map_path, suffix, spatial, flatten
        )
    elif source == "mouse":
        prep_release_metadata_mouse(
            ds_path, schema_version, map_path, suffix, spatial, flatten
        )
    else:
        raise ValueError(f"Unknown source {source}")


def prep_release_metadata_mouse(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    flatten: bool = False,
    map_only: bool = False,
):
    # source
    # spatial

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

    # # %%
    datasetid_mapper, mouseid_mapper, sampleid_mapper = load_mouse_id_mappers(
        map_path, suffix
    )

    # ds_path.mkdir(parents=True, exist_ok=True)
    mdata_path = ds_path / "metadata" / schema_version
    tables = [
        table
        for table in mdata_path.iterdir()
        if table.is_file() and table.suffix == ".csv"
    ]

    req_tables = MOUSE_TABLES
    if spatial:
        req_tables.append("SPATIAL")
    table_names = [table.stem for table in tables if table.stem in req_tables]

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

    # TODO: need to make this read an env variable or something safe.
    #### CREATE file metadata summaries
    key_file_path = Path.home() / f"Projects/ASAP/{team}-credentials.json"
    res = authenticate_with_service_account(key_file_path)

    file_metadata_path = ds_path / "file_metadata"
    if not file_metadata_path.exists():
        file_metadata_path.mkdir(parents=True, exist_ok=True)

    gen_raw_bucket_summary(
        raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
    )

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=spatial)

    if spatial:
        gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
    if spatial:

        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=visium
        )

    # HACK:
    # need to change file_metadata so artifacts.csv and (eventually curated_files.csv) point to curated bucket

    # export the tables to the metadata directory in a release subdirectory
    out_dir = ds_path / "metadata/release"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, out_dir / "cde_version")

    if not map_only:
        export_map_path = map_path  # root_path / "asap-ids/temp"
        export_mouse_id_mappers(
            export_map_path,
            suffix,
            datasetid_mapper,
            mouseid_mapper,
            sampleid_mapper,
        )


def prep_release_metadata_pmdbs(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    flatten: bool = False,
    map_only: bool = False,
):
    # source
    # spatial

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

    # # %%
    (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    ) = load_pmdbs_id_mappers(map_path, suffix)

    # ds_path.mkdir(parents=True, exist_ok=True)
    mdata_path = ds_path / "metadata" / schema_version
    tables = [
        table
        for table in mdata_path.iterdir()
        if table.is_file() and table.suffix == ".csv"
    ]

    req_tables = PMDBS_TABLES
    if spatial:
        req_tables.append("SPATIAL")

    table_names = [table.stem for table in tables if table.stem in req_tables]

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

    # TODO: need to make this read an env variable or something safe.
    #### CREATE file metadata summaries
    key_file_path = Path.home() / f"Projects/ASAP/{team}-credentials.json"
    res = authenticate_with_service_account(key_file_path)

    file_metadata_path = ds_path / "file_metadata"
    if not file_metadata_path.exists():
        file_metadata_path.mkdir(parents=True, exist_ok=True)

    gen_raw_bucket_summary(
        raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
    )
    if spatial:
        gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=spatial)

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)

    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
    if spatial:
        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=visium
        )

    # export the tables to the metadata directory in a release subdirectory
    out_dir = ds_path / "metadata/release"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, out_dir / "cde_version")

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
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    source: str = "pmdbs",
):
    """
    only maps by default
    """

    if source == "pmdbs":
        dfs = get_release_metadata_pmdbs(
            ds_path, schema_version, map_path, suffix, spatial
        )
    elif source == "mouse":
        dfs = get_release_metadata_mouse(
            ds_path, schema_version, map_path, suffix, spatial
        )

    elif source == "cell":
        # prep_release_metadata_cell(
        #     ds_path, schema_version, map_path, suffix, spatial, flatten, map_only
        # )
        print(f"WARNING.. cell source not implimented")
        return {}
    else:
        raise ValueError(f"Unknown source {source}")

    return dfs


def get_release_metadata_mouse(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
) -> dict:
    # source
    # spatial

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

    # # %%
    datasetid_mapper, mouseid_mapper, sampleid_mapper = load_mouse_id_mappers(
        map_path, suffix
    )

    # ds_path.mkdir(parents=True, exist_ok=True)
    mdata_path = ds_path / "metadata" / schema_version
    tables = [
        table
        for table in mdata_path.iterdir()
        if table.is_file() and table.suffix == ".csv"
    ]

    req_tables = MOUSE_TABLES
    if spatial:
        req_tables.append("SPATIAL")
    table_names = [table.stem for table in tables if table.stem in req_tables]

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
    if spatial:
        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=visium
        )

    return dfs


def get_release_metadata_pmdbs(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
) -> dict:

    # source
    # spatial

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

    # # %%
    (
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    ) = load_pmdbs_id_mappers(map_path, suffix)

    # ds_path.mkdir(parents=True, exist_ok=True)
    if schema_version == "v2.1":
        mdata_path = ds_path / "metadata" / "v2"
    else:
        mdata_path = ds_path / "metadata" / schema_version

    tables = [
        table
        for table in mdata_path.iterdir()
        if table.is_file() and table.suffix == ".csv"
    ]

    req_tables = PMDBS_TABLES
    if spatial:
        req_tables.append("SPATIAL")

    table_names = [table.stem for table in tables if table.stem in req_tables]

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
    if spatial:
        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=visium
        )

    return dfs


def load_and_process_table(
    table_name: str, tables_path: Path, cde_df: pd.DataFrame, print_log: bool = False
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

    table_path = tables_path / f"{table_name}.csv"
    schema = cde_df[cde_df["Table"] == table_name]
    report = ReportCollector(destination="NA")
    full_table, report = validate_table(df.copy(), table_name, schema, report)
    if table_path.exists():
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
    source_path: Path,
    export_path: Path | None = None,
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
    # load CDE
    # if isinstance(cde_version, str):
    #     if cde_version in ["v3.1", "v3.2", "v3.1"]:
    #         cde_df = CDEv3
    #     else:
    #         cde_df = read_CDE(cde_version)
    # else:
    cde_df = read_CDE(cde_version)

    # load and process tables
    tables_dict = {}
    aux_tables_dict = {}
    for table in tables:
        df, df_aux = load_and_process_table(table, source_path, cde_df, print_log)
        tables_dict[table] = df
        aux_tables_dict[table] = df_aux

        if export_path is not None:
            df.to_csv(export_path / f"{table}.csv", index=False)
            if not df_aux.empty:
                df_aux.to_csv(export_path / f"{table}_auxiliary.csv", index=False)

    return tables_dict, aux_tables_dict


# if __name__ == "__main__":
#     # Set up the argument parser

#     parser = argparse.ArgumentParser(description="A command-line tool  to update tables to a new schema version ")

#     # Add arguments
#     parser.add_argument("--tables", default=Path.cwd(),
#                         help="Path to the directory containing meta TABLES. Defaults to the current working directory.")
#     parser.add_argument("--vin", default="v1",
#                         help="Input version.  Defaults to v1.")
#     parser.add_argument("--vout", default="v3",


#     tables = MOUSE_TABLES + ["SPATIAL"]
#     cde_version = "v3.1"
#     source_path = metadata_path / "og"
#     export_path = metadata_path / "v3.1"
#     tables_dict, aux_tables_dict = process_schema(tables, cde_version, source_path, export_path, print_log=True)


def ingest_ds_info_doc(intake_doc: Path | str, ds_path: Path, doc_path: Path):
    """
    Ingest the dataset information from the docx file and export to json for zenodo upload.
    """

    # should read this from ds_path/version
    # just read in as text
    with open(ds_path / "version", "r") as f:
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
            # affiliations = pd.DataFrame(table_data[1:], columns=table_data[0])
            # if affiliations.shape[0] == 1:
            #     affiliations = affiliations.iloc[0, 0]

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

    # created
    # timestamp	Creation time of deposition (in ISO8601 format).
    # doi
    # string	Digital Object Identifier (DOI). When you publish your deposition, we register a DOI in DataCite for your upload, unless you manually provided us with one. This field is only present for published depositions.
    # doi_url
    # url	Persistent link to your published deposition. This field is only present for published depositions.
    # files
    # array	A list of deposition files resources.
    # id
    # integer	Deposition identifier
    # metadata
    # object	A deposition metadata resource
    # modified
    # timestamp	Last modification time of deposition (in ISO8601 format).
    # owner
    # integer	User identifier of the owner of the deposition.
    # record_id
    # integer	Record identifier. This field is only present for published depositions.
    # record_url
    # url	URL to public version of record for this deposition. This field is only present for published depositions.
    # state
    # string	One of the values:
    # * inprogress: Deposition metadata can be updated. If deposition is also unsubmitted (see submitted) files can be updated as well.
    # * done: Deposition has been published.
    # * error: Deposition is in an error state - contact our support.

    # submitted
    # bool	True if the deposition has been published, False otherwise.

    # title
    # string	Title of deposition (automatically set from metadata). Defaults to empty string.
    title = dataset_title

    # upload_type  string	Yes	Controlled vocabulary:
    # * publication: Publication
    # * poster: Poster
    # * presentation: Presentation
    # * dataset: Dataset
    # * image: Image
    # * video: Video/Audio
    # * software: Software
    # * lesson: Lesson
    # * physicalobject: Physical object
    # * other: Other
    upload_type = "dataset"

    # creators
    # array of objects	Yes	The creators/authors of the deposition. Each array element is an object with the attributes:
    # * name: Name of creator in the format Family name, Given names
    # * affiliation: Affiliation of creator (optional).
    # * orcid: ORCID identifier of creator (optional).
    # * gnd: GND identifier of creator (optional).
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
        # creators.append({"name": name, "affiliation": affiliation, "orcid": oricid})

    # description
    # string (allows HTML)	Yes	Abstract or description for deposition.
    description = dataset_description
    # ASAP
    communities = [{"identifier": "asaphub"}]
    # version
    version = ds_ver  # "2.0"?  also do "v1.0"
    # publication_date
    publication_date = pd.Timestamp.now().strftime("%Y-%m-%d")  # "2.0"?  also do "v1.0"

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
    doi_path = ds_path / "DOI"

    with open(doi_path / f"{long_dataset_name}.json", "w") as f:
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
    doi_path = ds_path / "DOI"
    if not doi_path.exists():
        doi_path.mkdir()

    long_dataset_name = ds_path.name
    with open(doi_path / f"{long_dataset_name}.md", "w") as f:
        f.write(md_content)

    # ## save a simple table to update STUDY table
    export_dict = {
        "project_name": f"{project_title.strip()}",  # protect the parkionson's apostrophe
        "project_description": f"{project_description.strip()}",
        "dataset_title": f"{dataset_title.strip()}",
        "dataset_description": f"{dataset_description.strip()}",
    }

    with open(doi_path / f"STUDY_fix.json", "w") as f:
        json.dump(export_dict, f, indent=4)

    df = pd.DataFrame(export_dict, index=[0])

    df.to_csv(doi_path / f"{long_dataset_name}.csv", index=False)


# fix STUDY
def fix_study_table(metadata_path: Path, doi_path: Path | None = None):
    """
    Update the STUDY table with the information from the DOI folder.
    """

    table = "STUDY"
    STUDY = read_meta_table(metadata_path / f"{table}.csv")

    if doi_path is None:
        doi_path = metadata_path.parent / "DOI"
    else:
        doi_path = Path(doi_path)

    STUDY_fix = pd.read_csv(doi_path / f"{metadata_path.parent.name}.csv")

    STUDY["project_name"] = STUDY_fix["project_name"]
    STUDY["project_description"] = STUDY_fix["project_description"]
    STUDY["dataset_title"] = STUDY_fix["dataset_title"]
    STUDY["dataset_description"] = STUDY_fix["dataset_description"]

    # export STUDY
    STUDY.to_csv(metadata_path / "STUDY.csv", index=False)


def get_stats_table(dfs: dict[pd.DataFrame], source: str = "pmdbs"):
    """ """
    if source == "pmdbs":
        return get_stat_tabs_pmdbs(dfs)
    elif source == "mouse":
        return get_stat_tabs_mouse(dfs)
    else:
        raise ValueError(f"Unknown source {source}")
        return {}, pd.DataFrame()


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


def get_stat_tabs_pmdbs(dfs: dict[pd.DataFrame]):
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
        # "age_at_collection",
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
        "intervention_id",
        "protocol_id",
        "intervention_aux_table",
    ]

    if "age_at_collection" in dfs["SUBJECT"].columns:
        subject_cols.append("age_at_collection")
    elif "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")
    else:
        raise ValueError("No age_at_collection column found in SUBJECT or SAMPLE")

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
    df = pd.merge(df, PMDBS_, on="ASAP_sample_id", how="left")

    # get stats for the dataset
    # 0. total number of samples

    age_at_collection = df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    n_samples = df["ASAP_sample_id"].nunique()

    n_subjects = df["ASAP_subject_id"].nunique()
    brain_code = (
        df["brain_region"].replace(_brain_region_coder).value_counts().to_dict()
    )
    brain_region = (
        df["brain_region"]
        .replace(_brain_region_coder)
        .map(_region_titles)
        .value_counts()
        .to_dict()
    )

    sex = (df["sex"].value_counts().to_dict(),)
    PD_status = (df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (df["condition_id"].value_counts().to_dict(),)
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

    return report, df


def get_stat_tabs_mouse(dfs: dict[pd.DataFrame]):
    """ """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_mouse_id",
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
        "ASAP_mouse_id",
        "sex",
        "age",
        "strain",
    ]

    condition_cols = [
        "condition_id",
        "intervention_name",
        "intervention_id",
        "protocol_id",
        "intervention_aux_table",
    ]

    SAMPLE_ = dfs["SAMPLE"][sample_cols]

    SUBJECT_ = dfs["MOUSE"][subject_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")

    # then JOIN the result with SUBJECT on "ASAP_subject_id" how=left to get "age_at_collection", "sex", "primary_diagnosis"
    df = pd.merge(df, SUBJECT_, on="ASAP_mouse_id", how="left")

    # get stats for the dataset
    # 0. total number of samples

    age_at_collection = df["age"].astype("float")
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    N = df["ASAP_sample_id"].nunique()

    # brain_region = (df["brain_region"].value_counts().to_dict(),)
    # PD_status = (df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (df["condition_id"].value_counts().to_dict(),)
    # diagnosis = (df["primary_diagnosis"].value_counts().to_dict(),)
    sex = (df["sex"].value_counts().to_dict(),)

    report = dict(
        N=N,
        condition_id=condition_id,
        age=age,
        sex=sex,
    )
    return report, df


def get_cohort_stats_table(dfs: dict[pd.DataFrame], source: str = "pmdbs"):
    """ """
    if source == "pmdbs":
        report, df = get_stat_tabs_pmdbs(dfs)
        N_datasets = df["ASAP_dataset_id"].nunique()
        N_teams = df["ASAP_team_id"].nunique()
        report["N_datasets"] = N_datasets
        report["N_teams"] = N_teams

    elif source == "mouse":
        report, df = get_stat_tabs_mouse(dfs)
        # TODO:
    else:
        raise ValueError(f"Unknown source {source}")
        report = {}
        df = pd.DataFrame()

    return report, df
