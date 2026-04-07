# release_util_deprecating.py
#
# This file contains functions that were previously in release_util.py but are
# no longer actively used. They are preserved here for reference in case they
# are needed to reconstruct historical release workflows.
#
# IMPORTANT: Do NOT import from this module in active code. Instead, if functions from here are needed,
# they should be moved back to release_util.py or a new module in utils/ as appropriate.
#
# Functions moved here:
#   - create_metadata_package        (archive scripts only)
#   - old_prep_release_metadata      (no callers outside this file)
#   - prep_release_metadata_cell     (only called by old_prep_release_metadata)
#   - prep_release_metadata_mouse    (only called by old_prep_release_metadata + archive)
#   - prep_release_metadata_pmdbs    (only called by old_prep_release_metadata + archive)
#   - ingest_ds_info_doc             (no callers anywhere)
#   - fix_study_table                (no callers anywhere)

from .util import (
    read_CDE,
    read_meta_table,
    read_CDE_asap_ids,
    export_meta_tables,
    load_tables,
    write_version,
)

from .asap_ids import (
    load_pmdbs_id_mappers,
    update_pmdbs_id_mappers,
    update_pmdbs_meta_tables_with_asap_ids,
    export_pmdbs_id_mappers,
    load_mouse_id_mappers,
    update_mouse_id_mappers,
    update_mouse_meta_tables_with_asap_ids,
    export_mouse_id_mappers,
    load_cell_id_mappers,
    update_cell_id_mappers,
    update_cell_meta_tables_with_asap_ids,
    export_cell_id_mappers,
)

from .file_metadata import (
    gen_raw_bucket_summary,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
    gen_spatial_bucket_summary,
    make_file_metadata,
)
from .constants import *  # List of tables expected (CDE <= v4.0)
from .doi import update_study_table_with_doi

# get_spatial_subtype_from_dataset_id is still used in release_util.py;
# import it from there to avoid duplication.
from .release_util import get_spatial_subtype_from_dataset_id


def create_metadata_package(
    dfs: dict[str, pd.DataFrame], metadata_path: str | Path, schema_version: str
):
    metadata_path = Path(metadata_path)
    final_metadata_path = os.path.join(metadata_path, schema_version)
    os.makedirs(final_metadata_path, exist_ok=True)

    export_meta_tables(dfs, final_metadata_path)
    write_version(schema_version, os.path.join(metadata_path, "cde_version"))
    write_version(schema_version, os.path.join(final_metadata_path, "cde_version"))


# !!!NOTE!!! This was used up to and including the December 2025 release.
# Importantly, it includes ID generation which is now handled separately.
# this is a wrapper to call source specific prep_release_metadata_* functions
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

    # this is broken due to deprecation of the pathlib...
    tables = [
        os.path.join(mdata_path, table)
        for table in os.listdir(mdata_path)
        if os.path.isfile(os.path.join(mdata_path, table)) and table.endswith(".csv")
    ]

    # TODO: make this accommodate proteomics
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

    # TODO: make this accommodate proteomics
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
                f"; [ORCID:{creator['orcid'].split('/')[-1]}]({creator['orcid']})"
            )
        if "affiliation" in creator:
            md_content += f"; {creator['affiliation']}"
        md_content += "\n"

    # need to add metadata information...
    ## dump md_content
    # write to a text file
    doi_path = os.path.join(ds_path, "DOI")
    os.makedirs(doi_path, exist_ok=True)

    long_dataset_name = ds_path.name
    with open(os.path.join(doi_path, f"{long_dataset_name}.md"), "w") as f:
        f.write(md_content)

    # ## save a simple table to update STUDY table
    export_dict = {
        "project_name": f"{project_title.strip()}",  # protect the Parkinson's apostrophe
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
