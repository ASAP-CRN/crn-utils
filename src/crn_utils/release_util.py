import pandas as pd
from pathlib import Path
import os
import docx
import json

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
from .file_metadata import *

from .checksums import extract_md5_from_details2, get_md5_hashes
from .bucket_util import authenticate_with_service_account
from .file_metadata import (
    get_raw_bucket_summary,
    get_raw_bucket_summary_flat,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
)
from .constants import *
from .doi import update_study_table_with_doi


__all__ = [
    "prep_release_metadata_mouse",
    "prep_release_metadata_pmdbs",
    "load_and_process_table",
    "process_schema",
    "create_metadata_package",
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


def prep_release_metadata_mouse(
    ds_path: Path,
    schema_version: str,
    map_path: Path,
    suffix: str,
    spatial: bool = False,
    flatten: bool = False,
):
    # source
    # spatial

    dataset_name = ds_path.name
    print(f"Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

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

    if flatten:
        get_raw_bucket_summary_flat(raw_bucket_name, file_metadata_path, dataset_name)
    else:
        get_raw_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"])

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
    if spatial:
        get_image_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=False
        )

    # export the tables to the metadata directory in a release subdirectory
    out_dir = ds_path / "metadata/release"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, out_dir / "cde_version")
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
):
    # source
    # spatial

    dataset_name = ds_path.name
    print(f"Processing {ds_path.name}")
    ds_parts = dataset_name.split("-")
    team = ds_parts[0]
    source = ds_parts[1]
    short_dataset_name = "-".join(ds_parts[2:])
    raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"

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
    )

    # TODO: need to make this read an env variable or something safe.
    #### CREATE file metadata summaries
    key_file_path = Path.home() / f"Projects/ASAP/{team}-credentials.json"
    res = authenticate_with_service_account(key_file_path)

    file_metadata_path = ds_path / "file_metadata"
    if not file_metadata_path.exists():
        file_metadata_path.mkdir(parents=True, exist_ok=True)

    # if flatten:
    #     get_raw_bucket_summary_flat(raw_bucket_name, file_metadata_path, dataset_name)
    # else:
    #     get_raw_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

    make_file_metadata(ds_path, file_metadata_path, dfs["DATA"])

    dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)

    dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
    if spatial:
        get_image_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)
        dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
            dfs["SPATIAL"], ds_path, visium=False
        )

    # export the tables to the metadata directory in a release subdirectory
    out_dir = ds_path / "metadata/release"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_meta_tables(dfs, out_dir)
    write_version(schema_version, out_dir / "cde_version")
    export_map_path = map_path / "asap-ids/master"
    export_pmdbs_id_mappers(
        export_map_path,
        suffix,
        datasetid_mapper,
        subjectid_mapper,
        sampleid_mapper,
        gp2id_mapper,
        sourceid_mapper,
    )


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
