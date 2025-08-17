import json
import io

# import pypandoc
from xhtml2pdf import pisa
import docx
from pathlib import Path
import os
import pandas as pd
from markdown import markdown

from .zenodo_util import *
from .util import read_meta_table

# import markdown
# from html import escape

__all__ = [
    "setup_DOI_info",
    "ingest_DOI_doc",
    "finalize_DOI",
    "make_readme_file",
    "make_pdf_file",
    "create_draft_doi",
    "add_anchor_file_to_doi",
    "update_doi_metadata",
    "publish_doi",
    "archive_deposition_local",
    "update_study_table_with_doi",
    "get_published_deposition",
    "setup_zenodo",
    "get_doi_from_dataset",
    "bump_doi_version",
    "create_draft_metadata",
    "replace_anchor_file_in_doi",
    "add_anchor_file_to_doi",
]

# def md_to_html(text):
#     """Convert markdown text to HTML"""
#     if not text:
#         return ""
#     # First escape any HTML to prevent injection
#     escaped_text = escape(str(text))
#     # Then convert markdown to HTML
#     html = markdown.markdown(escaped_text)
#     return html


def setup_DOI_info(
    ds_path: str | Path,
    doi_doc_path: str | Path,
    publication_date: None | str = None,
):

    study_df = read_meta_table(ds_path / "metadata/STUDY.csv")
    ingest_DOI_doc(ds_path, doi_doc_path, study_df, publication_date=publication_date)
    make_readme_file(ds_path)
    update_study_table(ds_path)


def ingest_DOI_doc(
    ds_path: str | Path,
    doi_doc_path: str | Path,
    study_df: pd.DataFrame,
    publication_date: None | str = None,
):
    """
    read docx, extract the information, and save in dataset/DOI subdirectory
    """
    ds_path = Path(ds_path)
    doi_doc_path = Path(doi_doc_path)
    long_dataset_name = ds_path.name

    # get details from the study df
    ASAP_lab_name = study_df["ASAP_lab_name"].values[0]
    PI_full_name = study_df["PI_full_name"].values[0]
    PI_email = study_df["PI_email"].values[0]
    submitter_name = study_df["submitter_name"].values[0]
    submitter_email = study_df["submitter_email"].values[0]
    publication_DOI = study_df["publication_DOI"].values[0]
    grant_ids = study_df["ASAP_grant_id"].values[0]
    print(grant_ids)
    team_name = (
        study_df["ASAP_team_name"].values[0].lower().replace("team-", "").capitalize()
    )

    # read the docx

    # should read this from ds_path/version
    # just read in as text
    with open(ds_path / "version", "r") as f:
        ds_ver = f.read().strip()
    # ds_ver = "v2.0"
    # Load the document
    document = docx.Document(doi_doc_path)

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
            dataset_title = (
                table_data[0][1].strip().replace("\n", " ").replace("\u2019", "'")
            )
            dataset_description = (
                table_data[1][1].strip().replace("\n", " ").replace("\u2019", "'")
            )
            print("got dataset title/description")
        elif name == "projects":
            project_title = (
                table_data[0][1].strip().replace("\n", " ").replace("\u2019", "'")
            )
            project_description = (
                table_data[1][1].strip().replace("\n", " ").replace("\u2019", "'")
            )
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
    title = dataset_title.strip().replace("Singel", "Single")

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
        affiliation = indiv[2].strip()
        oricid = indiv[3].strip()

        if name == ", ":  # this should block empty names
            continue

        to_append = {"name": name}
        if affiliation == "":
            affiliation = None
        else:
            to_append["affiliation"] = affiliation

        if oricid == "":
            oricid = None
        else:
            to_append["orcid"] = oricid.lstrip("https://orcid.org/")
        creators.append(to_append)

        # creators.append({"name": name, "affiliation": affiliation, "orcid": oricid})

    # description
    # string (allows HTML)	Yes	Abstract or description for deposition.
    # description = dataset_description

    dataset_description = dataset_description.strip()
    project_description = project_description.strip()
    # fix description to enable the numbered and bulletted lists...
    for i in range(10):
        rep_from = f" {i}. "
        rep_to = f"\n\n{i}. "
        project_description = project_description.strip().replace(rep_from, rep_to)
        dataset_description = dataset_description.strip().replace(rep_from, rep_to)
    project_description = project_description.strip().replace("* ", "\n\t* ")
    dataset_description = dataset_description.strip().replace("* ", "\n\t* ")

    description = f"""This Zenodo deposit contains a publicly available description of the Dataset:

**Title:** "{title}".

**Description:** {dataset_description}

--------------------------

> This dataset is made available to researchers via the ASAP CRN Cloud: [cloud.parkinsonsroadmap.org](https://cloud.parkinsonsroadmap.org). Instructions for how to request access can be found in the [User Manual](https://storage.googleapis.com/asap-public-assets/wayfinding/ASAP-CRN-Cloud-User-Manual.pdf).

> This research was funded by the Aligning Science Across Parkinson's Collaborative Research Network (ASAP CRN), through the Michael J. Fox Foundation for Parkinson's Research (MJFF).

> This Zenodo deposit was created by the ASAP CRN Cloud staff on behalf of the dataset authors. It provides a citable reference for a CRN Cloud Dataset

"""

    # Convert to html for good formatting
    description = markdown(description)

    # ASAP
    communities = [{"identifier": "asaphub"}]
    # version
    version = ds_ver  # "2.0"?  also do "v1.0"
    # license
    license = {"id": "cc-by-4.0"}
    refrences = [
        "Aligning Science Across Parkinson's Collaborative Research Network Cloud, https://cloud.parkinsonsroadmap.org/collections, RRID:SCR_023923",
        f"Team {team_name}",
    ]

    # publication_date
    if publication_date is None:
        publication_date = pd.Timestamp.now().strftime(
            "%Y-%m-%d"
        )  # "2.0"?  also do "v1.0"

    metadata = {
        "title": title,
        "upload_type": upload_type,
        "description": description,
        "publication_date": publication_date,
        "version": version,
        # "access_right": "open",
        "creators": creators,
        "resource_type": "dataset",
        "communities": communities,
        "references": refrences,
        "license": license,
    }

    if not pd.isna(grant_ids):
        if "," in grant_ids:
            grant_ids = grant_ids.split(",")
        elif ";" in grant_ids:
            grant_ids = grant_ids.split(";")
        else:
            grant_ids = [grant_ids]

        grants = [{"id": f"10.13039/100018231::{grant_id}"} for grant_id in grant_ids]
        metadata["grants"] = grants

    else:
        print("Warning: No grant ids found")

    export_data = {"metadata": metadata}

    # dump json
    doi_path = ds_path / "DOI"

    if not doi_path.exists():
        doi_path.mkdir()

    with open(doi_path / f"{long_dataset_name}.json", "w") as f:
        json.dump(export_data, f, indent=4)

    # also dump the table to make the documents and
    # ## save a simple table to update STUDY table
    project_dict = {
        "project_name": f"{project_title.strip()}",  # protect the parkionson's apostrophe
        "project_description": f"{project_description.strip()}",
        "dataset_title": f"{dataset_title.strip()}",
        "dataset_description": f"{dataset_description}",
        "creators": creators,
        "publication_date": publication_date,
        "version": version,
        "title": title,
        ### add the additional stuff from the study df
        "ASAP_lab_name": ASAP_lab_name,
        "PI_full_name": PI_full_name,
        "PI_email": PI_email,
        "submitter_name": submitter_name,
        "submitter_email": submitter_email,
        "publication_DOI": publication_DOI,
        "grant_ids": grant_ids,
        "team_name": team_name,
    }

    with open(doi_path / f"project.json", "w") as f:
        json.dump(project_dict, f, indent=4)

    # df = pd.DataFrame(project_dict, index=[0])
    # df.to_csv(doi_path / f"{long_dataset_name}.csv", index=False)
    # write the files.


def make_readme_file(ds_path: Path):
    """
    Make the stereotyped .md from the

    """
    # TODO:  add grant_ids

    # Aligning Science Across Parkinson's: 10.13039/100018231
    # grants = [{'id': f"10.13039/100018231::{grant_id}"}]

    long_dataset_name = ds_path.name

    team = long_dataset_name.split("-")[0]

    # load jsons
    doi_path = ds_path / "DOI"
    with open(doi_path / f"project.json", "r") as f:
        data = json.load(f)
    # data = clean_json_read(doi_path / f"project.json")

    title = data.get("title")
    project_title = data.get("project_name")
    project_description = data.get("project_description")
    dataset_title = data.get("dataset_title")
    dataset_description = data.get("dataset_description")
    creators = data.get("creators")
    publication_date = data.get("publication_date")
    version = data.get("version")
    ASAP_lab_name = data.get("ASAP_lab_name")
    PI_full_name = data.get("PI_full_name")
    PI_email = data.get("PI_email")
    submitter_name = data.get("submitter_name")
    submitter_email = data.get("submitter_email")
    publication_DOI = data.get("publication_DOI")
    grant_ids = data.get("grant_ids")
    team_name = data.get("team_name")

    # # avoid unicodes that mess up latex
    # rep_from = "α"
    # rep_to = "alpha"
    # project_description = project_description.strip().replace(rep_from, rep_to)
    # dataset_description = dataset_description.strip().replace(rep_from, rep_to)
    # rep_from = "₂"
    # rep_to = "2"
    # project_description = project_description.strip().replace(rep_from, rep_to)
    # dataset_description = dataset_description.strip().replace(rep_from, rep_to)

    description = f"""This Zenodo deposit contains a publicly available description of the Dataset:

# "{title}".

## Dataset Description:
 
{dataset_description.strip()}

"""
    readme_content = description

    readme_content += f"\n**Authors:**\n\n"
    for creator in creators:
        readme_content += f"* {creator['name']}"
        if "orcid" in creator:
            # format as link
            readme_content += (
                f"; [ORCID:{creator['orcid'].split("/")[-1]}]({creator['orcid']})"
            )
        if "affiliation" in creator:
            readme_content += f"; {creator['affiliation']}"
        readme_content += "\n"

    readme_content += f"\n\n**ASAP Team:** {team_name}\n\n"
    readme_content += f"**Dataset Name:** {ds_path.name}, v{version}\n\n"

    readme_content += f"**Principal Investigator:** {PI_full_name}, {PI_email}\n\n"
    readme_content += f"**Dataset Submitter:** {submitter_name}, {submitter_email}\n\n"
    readme_content += f"**Publication DOI:** {publication_DOI}\n\n"
    readme_content += f"**Grant IDs:** {grant_ids}\n\n"
    readme_content += f"**ASAP Lab:** {ASAP_lab_name}\n\n"
    readme_content += f"**ASAP Project:** {project_title}\n\n"
    readme_content += f"**Project Description:** {project_description}\n\n"
    readme_content += f"**Submission Date:** {publication_date}\n\n"
    readme_content += f"__________________________________________\n"

    readme_content += f"""

> This dataset is made available to researchers via the ASAP CRN Cloud: [cloud.parkinsonsroadmap.org](https://cloud.parkinsonsroadmap.org). Instructions for how to request access can be found in the [User Manual](https://storage.googleapis.com/asap-public-assets/wayfinding/ASAP-CRN-Cloud-User-Manual.pdf).

> This research was funded by the Aligning Science Across Parkinson's Collaborative Research Network (ASAP CRN), through the Michael J. Fox Foundation for Parkinson's Research (MJFF).

> This Zenodo deposit was created by the ASAP CRN Cloud staff on behalf of the dataset authors. It provides a citable reference for a CRN Cloud Dataset

"""

    readme_content_HTML = markdown(readme_content)

    print(f"{long_dataset_name=}")
    print(f"{doi_path=}")
    with open(doi_path / f"{long_dataset_name}_README.md", "w") as f:
        f.write(readme_content)

    make_pdf_file(readme_content_HTML, doi_path / f"{long_dataset_name}_README.pdf")


# def make_pdf_file(ds_path: Path):
#     """
#     Make the stereotyped .pdf from the .md file
#     """
#     if not "/Library/TeX/texbin:" in os.environ["PATH"]:
#         os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]
#     long_dataset_name = ds_path.name
#     doi_path = ds_path / "DOI"
#     file_path = doi_path / f"{long_dataset_name}.md"
#     pdf_path = doi_path / f"{long_dataset_name}.pdf"
#     output = pypandoc.convert_file(file_path, "pdf", outputfile=pdf_path)
#     return output


def make_pdf_file(html_content: str, output_filepath: str | Path):
    # Open a file to write the PDF to
    output_filepath = Path(output_filepath)
    # Convert HTML to PDF
    # The `io.BytesIO` is crucial for handling the content in memory before writing to file
    with open(output_filepath, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            io.BytesIO(html_content.encode("utf-8")),  # HTML needs to be bytes
            dest=result_file,
        )

    return not pisa_status.err  # True if conversion was successful


def update_study_table(ds_path: str | Path):
    """ """
    ds_path = Path(ds_path)
    metadata_path = ds_path / "metadata"
    STUDY = read_meta_table(metadata_path / "STUDY.csv")

    # load jsons
    doi_path = ds_path / "DOI"
    with open(doi_path / f"project.json", "r") as f:
        data = json.load(f)
    # data = clean_json_read(doi_path / f"project.json")

    STUDY["project_name"] = data["project_name"]
    STUDY["project_description"] = data["project_description"]
    STUDY["dataset_title"] = data["dataset_title"]
    STUDY["dataset_description"] = data["dataset_description"]
    # export STUDY
    STUDY.to_csv(metadata_path / "STUDY.csv", index=False)


def setup_zenodo(sandbox: bool = None):
    """Setup zenodo client.

    Args:
        sandbox (bool, optional): Whether to use the sandbox. Defaults to True.

    Returns:
        ZenodoClient: Zenodo client
    """
    zenodo = ZenodoClient(sandbox=sandbox)
    return zenodo


def get_published_deposition(zenodo: ZenodoClient, doi: str) -> dict:
    """Get the published deposition for a DOI.

    Args:
        zenodo (ZenodoClient): Zenodo client
        doi (str): DOI to get the deposition for

    Returns:
        dict: Zenodo deposition
    """
    record_id = zenodo._get_record_id_from_doi(doi)
    print(f"Record ID: {record_id}")
    deposition = zenodo._get_depositions_by_id(record_id)
    return deposition


def get_doi_from_dataset(ds_path: Path, version: bool = True):
    """Get the doi from the dataset.

    Args:
        ds_path (Path): Path to the dataset
        version (bool, optional): Whether to return the versioned doi. Defaults to False.

    Returns:
        str: DOI
    """
    doi_path = ds_path / "DOI"
    doi_file = "version.doi" if version else "dataset.doi"

    # fall back to doi if version does not exist
    if not (doi_path / doi_file).exists():
        doi_file = "doi"
        print(f"Warning: {doi_file} does not exist. Falling back to old format 'doi' ")

    with open(doi_path / doi_file, "r") as f:
        doi_id = f.read().strip()
    doi_id = doi_id.split(".")[-1]
    return doi_id


def create_draft_metadata(ds_path: Path, version: str = "0.1") -> dict:
    """Create a draft DOI on zenodo.

    Args:
        ds_path (Path): Path to the dataset
        version (str, optional): Version to use. Defaults to None. in which case force to be "0.1"

    Returns:
        dict: Zenodo deposition
    """
    with open(ds_path / f"DOI/{ds_path.name}.json", "r") as f:
        export_data = json.load(f)
    # export_data = clean_json_read(ds_path / f"DOI/{ds_path.name}.json")
    metadata = export_data["metadata"]

    if not version.endswith(".1"):
        print(f"Warning: Version {version} does not end with .1. Forcing to 0.1")
        metadata["version"] = "0.1"
    # metadata["license"] = {"id": "cc-by-4.0"}
    # metadata["communities"] = [{"id": "asaphub"}]

    return metadata


def create_draft_doi(zenodo: ZenodoClient, ds_path: Path, version: str = "0.1") -> dict:
    """Create a draft DOI on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client
        ds_path (Path): Path to the dataset
        version (str, optional): Version to use. Defaults to None. in which case force to be "0.1"

    Returns:
        dict: Zenodo deposition
    """

    with open(ds_path / f"DOI/{ds_path.name}.json", "r") as f:
        export_data = json.load(f)
    metadata = export_data["metadata"]

    if version == "0.1":
        print(f"Warning Draft DOI is defaulting to v0.1")

    metadata["version"] = version
    zenodo.create_new_deposition(metadata)

    return zenodo.deposition, metadata


def add_anchor_file_to_doi(
    zenodo: ZenodoClient, file_path: Path, doi_id: str | int | None = None
) -> dict:
    if isinstance(doi_id, int):
        print(f"Warning: You are using the record id {doi_id} instead of the doi")
        doi_id = str(doi_id)

    zenodo.set_deposition_id(doi_id)  # forces .bucket .title .deposition_id update

    # upload file to zenodo
    zenodo.upload_file(file_path)
    return zenodo.deposition


def replace_anchor_file_in_doi(
    zenodo: ZenodoClient,
    ds_path: Path,
    doi_id: str | None = None,
    new_file: str | None = None,
    old_file: str | None = None,
) -> dict:
    # upload file to zenodo

    if doi_id is not None:
        zenodo.set_deposition_id(doi_id)
        # zenodo.deposition_id = doi_id

    # add new file first
    if new_file is None:
        new_file = f"{ds_path.name}_README.pdf"
    new_file_path = ds_path / "DOI" / new_file

    # add anchor file
    deposition = add_anchor_file_to_doi(zenodo, new_file_path, doi_id=doi_id)

    # else use current deposition
    if old_file is None:
        old_file = f"{ds_path.name}.pdf"
    # find file_id
    file_ids = zenodo.get_file_ids(doi_id)
    file_id = file_ids.get(old_file, "")
    if file_id == "":
        print(f"Could not find file {old_file} in deposition {doi_id}")

    if file_id != "":
        # delete old file
        msg = zenodo.delete_file(file_id)
        print(msg)
    else:
        # no file to delete...
        print(f"no file to delete...")

    return zenodo.deposition


def bump_doi_version(zenodo: ZenodoClient, doi_id: str | int) -> dict:
    """Bump the version of a DOI on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client
        doi_id (str | int): DOI to bump
        new_version (str): New version to bump to

    Returns:
        dict: Zenodo deposition
    """
    if isinstance(doi_id, int):
        print(f"Warning: You are using the record id {doi_id} instead of the doi")
        doi_id = str(doi_id)

    zenodo.deposition_id = doi_id
    # deposition = zenodo.all_depositions[doi_id] #get_published_deposition(zenodo,doi_id)
    # zenodo.set_deposition_iddoi_id)
    deposition = zenodo.make_new_version()
    return deposition

    # metadata = deposition["metadata"]
    # metadata["version"] = new_version
    # return zenodo.change_metadata(metadata)

    # return zenodo.deposition


def update_doi_metadata(
    zenodo: ZenodoClient, doi_id: str | int, metadata: dict
) -> dict:
    """update the metadata on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client
        metadata (dict, optional): Metadata to update.
    Returns:
        dict: Zenodo deposition
    """
    if isinstance(doi_id, int):
        print(f"Warning: You are using the record id {doi_id} instead of the doi")
        doi_id = str(doi_id)

    zenodo.deposition_id = doi_id
    deposition = zenodo.deposition
    if deposition["state"] == "done":
        print("Deposition is already published. unlocking for update.")
        deposition = zenodo.unlock_deposition()

    # # add missing keys from deposition to metadata
    # for key in deposition.get("metadata", {}).keys():
    #     if key not in metadata:
    #         metadata[key] = deposition["metadata"][key]
    #         print(f"Warning: Adding missing key {key} to metadata")

    deposition = zenodo.change_metadata(metadata)
    return deposition


def publish_doi(zenodo: ZenodoClient, doi_id: str | int) -> dict:
    """Publish a DOI on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client

    Returns:
        dict: Zenodo deposition
    """
    if isinstance(doi_id, int):
        print(f"Warning: You are using the record id {doi_id} instead of the doi")
        doi_id = str(doi_id)

    zenodo.set_deposition_id(doi_id)

    deposition = zenodo.deposition
    if deposition.get("state", {}) == "done":
        print("Deposition is already published.")
        return deposition

    return zenodo.publish()


def finalize_DOI(ds_path: Path, deposition: dict, prerelease: bool = False):
    """
    Write the DOI information to the dataset/DOI directory
    """
    doi = deposition.get("doi", "")
    doi_url = deposition.get("doi_url", "")

    if doi == "":
        prerelease = True
        tmp = deposition["metadata"]["prereserve_doi"]
        doi = tmp["doi"]
        doi_url = f"https://doi.org/{doi}"
        # 'prereserve_doi': {'doi': '10.5281/zenodo.15578088', 'recid': 15578088}},
    # else:
    #     prerelease = False

    conceptdoi = deposition["conceptdoi"]
    conceptdoi_url = doi_url.replace(doi, conceptdoi)

    doi_path = ds_path / "DOI"
    with open(doi_path / "version.doi", "w") as f:
        # write doi to file as text
        f.write(doi)

    with open(doi_path / "dataset.doi", "w") as f:
        # write doi to file as text
        f.write(conceptdoi)

    with open(doi_path / f"{conceptdoi.replace('/','_')}", "w") as f:
        # write doi to file
        f.write(f"ALL_VERSIONS        : {conceptdoi_url}\n")
        if prerelease:
            f.write(f"CURRENT (prerelease): {doi_url}")
        else:
            f.write(f"CURRENT             : {doi_url}")


def archive_deposition_local(ds_path: Path, arch_name: str, deposition: dict):
    doi_path = ds_path / "DOI"
    with open(doi_path / f"{arch_name}.json", "w") as f:
        json.dump(deposition, f, indent=2)


def update_study_table_with_doi(study_df: pd.DataFrame, ds_path: str | Path):
    """ """
    ds_path = Path(ds_path)
    metadata_path = ds_path / "metadata"
    STUDY = read_meta_table(metadata_path / "STUDY.csv")

    # load jsons
    doi_path = ds_path / "DOI"

    with open(doi_path / "dataset.doi", "r") as f:
        ds_doi = f.read().strip()
    study_df["dataset_DOI"] = ds_doi
    study_df["dataset_DOI_url"] = f"https://doi.org/{ds_doi}"

    # get dataset version from version file
    with open(ds_path / "version", "r") as f:
        ds_ver = f.read().strip()
    study_df["dataset_version"] = ds_ver

    return study_df


# def clean_json_read(file_path):
#     """
#     Read a JSON file and clean non-breaking spaces and other problematic characters.

#     Args:
#         file_path (str): Path to the JSON file

#     Returns:
#         dict: The cleaned JSON data
#     """
#     # Read the file as text first
#     with open(file_path, "r", encoding="utf-8") as f:
#         json_text = f.read()

#     # Clean the text by replacing non-breaking spaces with regular spaces
#     cleaned_text = json_text.replace("\xa0", " ")

#     # Also clean other common problematic characters
#     cleaned_text = re.sub(r"[\u2018\u2019]", "'", cleaned_text)  # Smart single quotes
#     cleaned_text = re.sub(r"[\u201c\u201d]", '"', cleaned_text)  # Smart double quotes
#     cleaned_text = cleaned_text.replace("\u2013", "-")  # En dash
#     cleaned_text = cleaned_text.replace("\u2014", "--")  # Em dash
#     cleaned_text = cleaned_text.replace("\u2026", "...")  # Ellipsis

#     # Parse the cleaned JSON
#     try:
#         data = json.loads(cleaned_text)
#         return data
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         # If there's still an error, try a more aggressive cleaning approach
#         cleaned_text = re.sub(
#             r"[^\x00-\x7F]+", " ", cleaned_text
#         )  # Remove all non-ASCII
#         try:
#             data = json.loads(cleaned_text)
#             return data
#         except json.JSONDecodeError as e2:
#             print(f"Still couldn't decode JSON after aggressive cleaning: {e2}")
#             raise
