import json
import pypandoc
import docx
from pathlib import Path
import os
import pandas as pd

from .zenodo_util import *
from .util import read_meta_table

# %load_ext dotenv
# %dotenv


__all__ = [
    "ingest_DOI_doc",
    "make_md_file",
    "make_pdf_file",
    "update_study_table",
    "setup_DOI",
    "mint_DOI",
    "create_DOI",
    "finalize_DOI",
    "archive_deposition_local",
]


def ingest_DOI_doc(
    ds_path: str | Path, doi_doc_path: str | Path, publication_date: None | str = None
):
    """
    read docx, extract the information, and save in dataset/DOI subdirectory
    """
    ds_path = Path(ds_path)
    doi_doc_path = Path(doi_doc_path)
    long_dataset_name = ds_path.name

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
    description = dataset_description
    # ASAP
    communities = [{"identifier": "asaphub"}]
    # version
    version = ds_ver  # "2.0"?  also do "v1.0"

    # publication_date
    if publication_date is None:
        publication_date = pd.Timestamp.now().strftime(
            "%Y-%m-%d"
        )  # "2.0"?  also do "v1.0"

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
    }

    with open(doi_path / f"project.json", "w") as f:
        json.dump(project_dict, f, indent=4)

    # df = pd.DataFrame(project_dict, index=[0])
    # df.to_csv(doi_path / f"{long_dataset_name}.csv", index=False)


def make_md_file(ds_path: Path):
    """
    Make the stereotyped .md from the

    """
    if not "/Library/TeX/texbin:" in os.environ["PATH"]:
        os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]

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

    # fix description to enable the numbered and bulletted lists...
    for i in range(10):
        rep_from = f"{i}. "
        rep_to = f"\n\n{i}. "
        project_description = project_description.strip().replace(rep_from, rep_to)
        dataset_description = dataset_description.strip().replace(rep_from, rep_to)
    project_description = project_description.strip().replace("* ", "\n\t* ")
    dataset_description = dataset_description.strip().replace("* ", "\n\t* ")

    md_content = f"# {title}\n\n __Dataset Description:__  {dataset_description}\n\n"
    md_content += f" ### ASAP Team: Team {team.capitalize()}:\n\n > *Project:* __{project_title}__:{project_description}\n\n"
    md_content += f"\n\n_____________________\n\n"
    md_content += f"*ASAP CRN Cloud Dataset Name:* {long_dataset_name}, v{version}\n"
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

    with open(doi_path / f"{long_dataset_name}.md", "w") as f:
        f.write(md_content)


def make_pdf_file(ds_path: Path):
    """
    Make the stereotyped .pdf from the .md file
    """
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


def mint_DOI(ds_path: Path) -> dict:
    """
    Mint a DOI for the dataset.
    """
    api_token = setup_zenodo()
    metadata = setup_DOI(ds_path)
    deposition, upload_response = create_DOI(ds_path, metadata, api_token)
    published_deposition = finalize_DOI(ds_path, deposition, api_token)
    return published_deposition, upload_response


def setup_DOI(ds_path: Path) -> dict:
    # load json

    with open(ds_path / f"DOI/{ds_path.name}.json", "r") as f:
        export_data = json.load(f)

    metadata = export_data["metadata"]
    return metadata


def create_DOI(ds_path: Path, metadata: dict, api_token: str) -> dict:
    # Create a new deposition.
    deposition = create_deposition(api_token)

    # file_path = doi_path / f"{long_dataset_name}.md"
    pdf_path = ds_path / f"DOI/{ds_path.name}.pdf"

    # Upload the file.
    upload_response = upload_file(deposition, pdf_path, api_token)
    # Update the deposition metadata with the provided metadata.
    deposition = update_metadata(deposition, api_token, metadata)
    return deposition, upload_response


def finalize_DOI(ds_path: Path, deposition: dict, api_token: str) -> dict:
    # Publish the deposition to reserve a DOI.
    print("Publishing deposition...")
    published_deposition = publish_deposition(deposition, api_token)
    # Print the final published deposition details.
    print("\nFinal Published Deposition Details:")
    print(json.dumps(published_deposition, indent=2))
    doi = published_deposition["doi"]
    doi_url = published_deposition["doi_url"]
    # 10.5281/zenodo.15162835
    doi_path = ds_path / "DOI"
    with open(doi_path / "doi", "w") as f:
        # write doi to file as text
        f.write(doi)

    with open(doi_path / f"{doi.replace('/','_')}", "w") as f:
        # write doi to file
        f.write(doi_url)

    return published_deposition


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
