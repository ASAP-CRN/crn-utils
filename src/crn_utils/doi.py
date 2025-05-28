import json
import pypandoc
import docx
from pathlib import Path
import os
import pandas as pd

from .zenodo_util import *
from .util import read_meta_table

# import markdown
# from html import escape


# %load_ext dotenv
# %dotenv


__all__ = [
    "setup_DOI_info",
    "ingest_DOI_doc",
    "make_readme_file",
    "make_pdf_file",
    "create_draft_doi",
    "add_anchor_file_to_doi",
    "update_doi",
    "publish_doi",
    "archive_deposition_local",
    "update_study_table_with_doi",
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
    # make_pdf_file(ds_path)
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

        if name == ", ":  # this should block empty names
            continue

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
    # description = dataset_description

    description = f"""
This Zenodo deposit contains a publicly available description of the dataset: 

    “{title}". ({ds_path.name}, v{ds_ver}) submitted by ASAP Team: {team_name}.

This dataset will be made available to researchers via the ASAP CRN Cloud in approximately one month. Once available, the dataset will be accessible by going to https://cloud.parkinsonsroadmap.org.
This research was funded by the Aligning Science Across Parkinson’s Collaborative Research Network (ASAP CRN), through the Michael J. Fox Foundation for Parkinson’s Research (MJFF).

This Zenodo deposit was created by the ASAP CRN Cloud staff on behalf of the dataset Authors. It provides a citable reference for a CRN Cloud Dataset

- Aligning Science Across Parkinson’s 

"""

    # ASAP
    communities = [{"identifier": "asaphub"}]
    # version
    version = ds_ver  # "2.0"?  also do "v1.0"

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
        "publication_type": "other",
        "resource_type": "dataset",
        "communities": communities,
    }

    if not pd.isna(grant_ids):
        metadata["grants"] = [{"id": f"10.13039/100018231::{grant_ids}"}]

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
        "dataset_description": f"{dataset_description.strip()}",
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

    # Aligning Science Across Parkinson’s: 10.13039/100018231
    # grants = [{'id': f"10.13039/100018231::{grant_id}"}]

    long_dataset_name = ds_path.name

    team = long_dataset_name.split("-")[0]

    # load jsons
    doi_path = ds_path / "DOI"
    with open(doi_path / f"project.json", "r") as f:
        data = json.load(f)

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

    # fix description to enable the numbered and bulletted lists...
    for i in range(10):
        rep_from = f"{i}. "
        rep_to = f"\n\n{i}. "
        project_description = project_description.strip().replace(rep_from, rep_to)
        dataset_description = dataset_description.strip().replace(rep_from, rep_to)
    project_description = project_description.strip().replace("* ", "\n\t* ")
    dataset_description = dataset_description.strip().replace("* ", "\n\t* ")

    # # avoid unicodes that mess up latex
    # rep_from = "α"
    # rep_to = "alpha"
    # project_description = project_description.strip().replace(rep_from, rep_to)
    # dataset_description = dataset_description.strip().replace(rep_from, rep_to)
    # rep_from = "₂"
    # rep_to = "2"
    # project_description = project_description.strip().replace(rep_from, rep_to)
    # dataset_description = dataset_description.strip().replace(rep_from, rep_to)

    readme_content = (
        f"TITLE: {title}\n\nDataset Description:  {dataset_description}\n\n"
    )
    readme_content += f"ASAP Team: Team {team.capitalize()}\n"
    readme_content += f"ASAP CRN Cloud Dataset Name: {long_dataset_name}]\nDataset Version:v{version}\n"
    readme_content += f"Authors:\n"
    for creator in creators:
        readme_content += f"\t* {creator['name']}"
        if "orcid" in creator:
            # format as link
            readme_content += (
                f"; [ORCID:{creator['orcid'].split("/")[-1]}]({creator['orcid']})"
            )
        if "affiliation" in creator:
            readme_content += f"; {creator['affiliation']}"
        readme_content += "\n"

    readme_content += f"\nPrincipal Investigator: {PI_full_name}, {PI_email}\n"
    readme_content += f"Dataset Submitter: {submitter_name}, {submitter_email}\n"
    readme_content += f"Publication DOI: {publication_DOI}\n"
    readme_content += f"Grant IDs: {grant_ids}\n"
    readme_content += f"ASAP Lab: {ASAP_lab_name}\n"
    readme_content += (
        f"ASAP Project: {project_title}\nProject Description: {project_description}\n\n"
    )
    readme_content += f"Submission Date: {publication_date}\n"
    readme_content += f"__________________________________________\n"

    readme_content += f"""
    This Zenodo deposit contains a publicly available description of the dataset.

    This dataset will be made available to researchers via the ASAP CRN Cloud in approximately one month. Once available, the dataset will be accessible by going to https://cloud.parkinsonsroadmap.org.
    This research was funded by the Aligning Science Across Parkinson’s Collaborative Research Network (ASAP CRN), through the Michael J. Fox Foundation for Parkinson’s Research (MJFF).

    This Zenodo deposit was created by the ASAP CRN Cloud staff on behalf of the dataset Authors. It provides a citable reference for a CRN Cloud Dataset

    - Aligning Science Across Parkinson’s 

    """

    with open(doi_path / f"{long_dataset_name}_README.txt", "w") as f:
        f.write(readme_content)


def make_pdf_file(ds_path: Path):
    """
    Make the stereotyped .pdf from the .md file
    """
    if not "/Library/TeX/texbin:" in os.environ["PATH"]:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]
    long_dataset_name = ds_path.name
    doi_path = ds_path / "DOI"
    file_path = doi_path / f"{long_dataset_name}.md"
    pdf_path = doi_path / f"{long_dataset_name}.pdf"
    output = pypandoc.convert_file(file_path, "pdf", outputfile=pdf_path)
    return output


def update_study_table(ds_path: str | Path):
    """ """
    ds_path = Path(ds_path)
    metadata_path = ds_path / "metadata"
    STUDY = read_meta_table(metadata_path / "STUDY.csv")

    # load jsons
    doi_path = ds_path / "DOI"
    with open(doi_path / f"project.json", "r") as f:
        data = json.load(f)

    STUDY["project_name"] = data["project_name"]
    STUDY["project_description"] = data["project_description"]
    STUDY["dataset_title"] = data["dataset_title"]
    STUDY["dataset_description"] = data["dataset_description"]
    # export STUDY
    STUDY.to_csv(metadata_path / "STUDY.csv", index=False)


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
    metadata = ZenodoMetadata(**export_data["metadata"])

    if version == "0.1":
        print(f"Warning Draft DOI is defaulting to v0.1")

    metadata.version = version
    zenodo.create_deposition(metadata)

    return zenodo.deposition, metadata


def add_anchor_file_to_doi(zenodo: ZenodoClient, ds_path: Path) -> dict:
    # upload file to zenodo
    file_path = ds_path / f"DOI/{ds_path.name}_README.txt"
    zenodo.upload_file(file_path)
    return zenodo.deposition


def update_doi(zenodo: ZenodoClient, metadata: dict | ZenodoMetadata) -> dict:
    """Create a draft DOI on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client
        metadata (dict, optional): Metadata to update. Defaults to None.

    Returns:
        dict: Zenodo deposition
    """

    if isinstance(metadata, dict):
        metadata = ZenodoMetadata(**metadata)
    zenodo.change_metadata(metadata)

    return zenodo.deposition


def publish_doi(zenodo: ZenodoClient) -> dict:
    """Publish a DOI on zenodo.

    Args:
        zenodo (ZenodoClient): Zenodo client

    Returns:
        dict: Zenodo deposition
    """

    return zenodo.publish()


# def mint_DOI(ds_path: Path) -> dict:
#     """
#     Mint a DOI for the dataset.
#     """
#     api_token = setup_zenodo()
#     metadata = setup_DOI(ds_path)
#     deposition, upload_response = create_DOI(ds_path, metadata, api_token)
#     published_deposition = finalize_DOI(ds_path, deposition, api_token)
#     return published_deposition, upload_response


# def setup_DOI(ds_path: Path) -> dict:
#     # load json

#     with open(ds_path / f"DOI/{ds_path.name}.json", "r") as f:
#         export_data = json.load(f)

#     metadata = export_data["metadata"]
#     return metadata


# def create_DOI(ds_path: Path, metadata: dict, api_token: str) -> dict:
#     # Create a new deposition.
#     deposition = create_deposition(api_token)

#     # file_path = doi_path / f"{long_dataset_name}.md"
#     pdf_path = ds_path / f"DOI/{ds_path.name}.pdf"

#     # Upload the file.
#     upload_response = upload_file(deposition, pdf_path, api_token)
#     # Update the deposition metadata with the provided metadata.
#     deposition = update_metadata(deposition, api_token, metadata)
#     return deposition, upload_response


# def finalize_DOI(ds_path: Path, deposition: dict, api_token: str) -> dict:
#     # Publish the deposition to reserve a DOI.
#     print("Publishing deposition...")
#     published_deposition = publish_deposition(deposition, api_token)
#     # Print the final published deposition details.
#     print("\nFinal Published Deposition Details:")
#     print(json.dumps(published_deposition, indent=2))
#     doi = published_deposition["doi"]
#     doi_url = published_deposition["doi_url"]
#     # 10.5281/zenodo.15162835
#     doi_path = ds_path / "DOI"
#     with open(doi_path / "doi", "w") as f:
#         # write doi to file as text
#         f.write(doi)

#     with open(doi_path / f"{doi.replace('/','_')}", "w") as f:
#         # write doi to file
#         f.write(doi_url)

#     return published_deposition


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

    with open(doi_path / "doi", "r") as f:
        doi = f.read().strip()
    study_df["dataset_DOI"] = doi

    with open(doi_path / f"{doi.replace('/','_')}", "r") as f:
        doi_url = f.read().strip()
    study_df["dataset_DOI_url"] = doi_url

    # get dataset version from version file
    with open(ds_path / "version", "r") as f:
        ds_ver = f.read().strip()

    study_df["dataset_version"] = ds_ver

    return study_df


# # always start by creating a Client object
# zeno = zenodopy.Client(sandbox=True)

# # list project id's associated to zenodo account
# zeno.list_projects

# # create a project
# zeno.create_project(title="test_project", upload_type="other")
# # your zeno object now points to this newly created project

# # create a file to upload
# with open("~/test_file.txt", "w+") as f:
#     f.write("Hello from zenodopy")

# # upload file to zenodo
# zeno.upload_file("~/test.file.txt")

# # list files of project
# zeno.list_files

# # set project to other project id's
# zeno.set_project("<id>")

# # delete project
# zeno._delete_project(dep_id="<id>")
