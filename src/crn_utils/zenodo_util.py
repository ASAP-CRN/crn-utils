import json
import os
from pathlib import Path
import re
import requests
from datetime import datetime
import time
from dataclasses import dataclass, field
from typing import Optional, List
from dotenv import load_dotenv

__all__ = [
    "ZenodoMetadata",
    "ZenodoToken",
    "ZenodoClient",
]


# [
#     "created",
#     "modified",
#     "id",
#     "conceptrecid",
#     "metadata",
#     "title",
#     "links",
#     "record_id",
#     "owner",
#     "files",
#     "state",
#     "submitted",
# ]


@dataclass
class ZenodoDeposition:
    created: str
    modified: str
    id: int
    conceptrecid: int
    metadata: dict
    title: str
    links: dict
    record_id: int
    owner: int
    files: list
    state: str
    submitted: bool

    @classmethod
    def from_dict(cls, deposition_dict: dict) -> "ZenodoDeposition":
        return cls(**deposition_dict)

    @property
    def is_published(self):
        return self.submitted

    def to_zenodo_metadata(self):
        return ZenodoMetadata(**self.metadata)

    @property
    def is_draft(self):
        return not self.submitted

    @property
    def doi(self):
        return self.conceptrecid


@dataclass
class ZenodoMetadata:
    title: str
    upload_type: str = "other"
    description: str | None = None
    publication_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    version: str = "0.1.0"
    # access_right: str = "open"
    # license: str = "Apache-2.0"
    # keywords: List[str] = field(default_factory=lambda: ["zenodo", "github", "git"])
    creators: List[dict] = field(
        default_factory=lambda: [{"name": "Jhon, Doe", "orcid": "0000-0003-2584-3576"}]
    )
    publication_type: str = "other"
    resource_type: str = "dataset"
    communities: List[dict] = field(default_factory=lambda: [{"identifier": "asaphub"}])
    grants: List[dict] = field(default_factory=lambda: [])

    @classmethod
    def parse_metadata_from_json(cls, json_file_path: Path) -> "ZenodoMetadata":
        """Parse metadata from a JSON file into a ZenodoMetadata object."""
        json_file_path = Path(json_file_path).expanduser()
        if not json_file_path.exists():
            raise ValueError(
                f"{json_file_path} does not exist. Please check you entered the correct path."
            )

        with json_file_path.open("r") as json_file:
            data = json.load(json_file)

        metadata_dict = data.get("metadata", {})
        return cls(**metadata_dict)


class ZenodoToken(requests.auth.AuthBase):
    """Bearer Authentication"""

    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r


class ZenodoClient(object):
    """Zenodo Client object

    Use this class to instantiate a zenodopy object
    to interact with your Zenodo account

        ```
        import zenodopy
        zeno = zenodopy.Client()
        zeno.help()
        ```

    Setup instructions:
        ```
        zeno.setup_instructions
        ```
    """

    def __init__(
        self, title=None, bucket=None, deposition_id=None, sandbox=None, token=None
    ):
        """initialization method"""
        if sandbox:
            self._endpoint = "https://sandbox.zenodo.org/api"
        else:
            self._endpoint = "https://zenodo.org/api"

        self.title = title
        self.bucket = bucket
        self.deposition_id = deposition_id
        self.sandbox = sandbox
        self._token = self._load_from_env if token is None else token
        self._auth_token = ZenodoToken(self._token)
        # 'metadata/prereservation_doi/doi'
        self.all_depositions = self._get_depositions()  # list[dict] =

    def __repr__(self):
        return f"zenodoapi('{self.title}','{self.bucket}','{self.deposition_id}')"

    def __str__(self):
        return f"{self.title} --- {self.deposition_id}"

    # ---------------------------------------------
    # hidden functions
    # ---------------------------------------------

    @staticmethod
    def _get_upload_types():
        """Acceptable upload types

        Returns:
            list: contains acceptable upload_types
        """
        return [
            "Publication",
            "Poster",
            "Presentation",
            "Dataset",
            "Image",
            "Video/Audio",
            "Software",
            "Lesson",
            "Physical object",
            "Other",
        ]

    @staticmethod
    def _load_token():
        """reads the configuration file

        Configuration file should be ~/.zenodo_token

        Args:
            path (str): location of the file with ACCESS_TOKEN

        Returns:
            dict: dictionary with API ACCESS_TOKEN
        """

        load_dotenv()
        api_token = os.getenv("ZENODO_API_TOKEN")
        return api_token

    @property
    def _load_from_env(self):
        """reads the web3.storage token from env
        configuration file is ~/.web3_storage_token
        Returns:
            str: ACCESS_TOKEN to connect to web3 storage
        """
        key = self._load_token()
        return key

    def _get_depositions(self):
        """gets the current project deposition

        this provides details on the project, including metadata

        Returns:
            dict: dictionary containing project details
        """
        # get request, returns our response
        r = requests.get(f"{self._endpoint}/deposit/depositions", auth=self._auth_token)
        if r.ok:
            return r.json()
        else:
            return r.raise_for_status()

    def _get_depositions_by_id(self, dep_id: str | None = None):
        """gets the deposition based on project id

        this provides details on the project, including metadata

        Args:
            dep_id (str): project deposition ID, if None, uses the deposition_id of the class

        Returns:
            dict: dictionary containing project details
        """
        dep_id = dep_id if dep_id is not None else self.deposition_id
        # get request, returns our response
        if self.deposition_id is not None:
            r = requests.get(
                f"{self._endpoint}/deposit/depositions/{self.deposition_id}",
                auth=self._auth_token,
            )
        else:
            print(" ** no deposition id is set on the project ** ")
            return None
        if r.ok:
            return r.json()
        else:
            return r.raise_for_status()

    def _get_depositions_files(self):
        """gets the file deposition

        ** not used, can safely be removed **

        Returns:
            dict: dictionary containing project details
        """
        # get request, returns our response
        if self.deposition_id is not None:
            r = requests.get(
                f"{self._endpoint}/deposit/depositions/{self.deposition_id}/files",
                auth=self._auth_token,
            )
        else:
            print(" ** no deposition id is set on the project ** ")

        if r.ok:
            return r.json()
        else:
            return r.raise_for_status()

    def _get_bucket_by_title(self, title=None):
        """gets the bucket URL by project title

        This URL is what you upload files to

        Args:
            title (str): project title

        Returns:
            str: the bucket URL to upload files to
        """
        dic = self.list_projects
        dep_id = dic[title] if dic is not None else None

        # get request, returns our response, this the records metadata
        r = requests.get(
            f"{self._endpoint}/deposit/depositions/{dep_id}", auth=self._auth_token
        )

        if r.ok:
            return r.json()["links"]["bucket"]
        else:
            return r.raise_for_status()

    def _get_bucket_by_id(self, dep_id=None):
        """gets the bucket URL by project deposition ID

        This URL is what you upload files to

        Args:
            dep_id (str): project deposition ID

        Returns:
            str: the bucket URL to upload files to
        """
        # get request, returns our response
        if dep_id is not None:
            r = requests.get(
                f"{self._endpoint}/deposit/depositions/{dep_id}", auth=self._auth_token
            )
        else:
            r = requests.get(
                f"{self._endpoint}/deposit/depositions/{self.deposition_id}",
                auth=self._auth_token,
            )

        if r.ok:
            return r.json()["links"]["bucket"]
        else:
            return r.raise_for_status()

    def _get_api(self):
        # get request, returns our response
        r = requests.get(f"{self._endpoint}", auth=self._auth_token)

        if r.ok:
            return r.json()
        else:
            return r.raise_for_status()

    def _is_doi(self, string=None):
        """test if string is of the form of a zenodo doi
        10.5281.zenodo.[0-9]+

        Args:
            string (strl): string to test. Defaults to None.

        Returns:
           bool: true is string is doi-like
        """
        import re

        pattern = re.compile("10.5281/zenodo.[0-9]+")
        return pattern.match(string)

    def _get_record_id_from_doi(self, doi: str | None = None):
        """return the record id for given doi

        Args:
            doi (string, optional): the zenodo doi. Defaults to None.

        Returns:
            str: the record id from the doi (just the last numbers)
        """
        return doi.split(".")[-1]

    def _get_latest_record(self):
        """return the latest record id for given record id

        Returns:
            str: the latest record id or 'None' if not found
        """
        try:
            record = self._get_depositions_by_id()["links"]["latest"].split("/")[-1]
        except:
            record = "None"
        return record

    def _delete_project(self, dep_id: str | None = None):
        """delete a project from repository by ID

        Args:
            dep_id (str): The project deposition ID
        """
        print("")
        # if input("are you sure you want to delete this project? (y/n)") == "y":
        # delete requests, we are deleting the resource at the specified URL
        dep_id = dep_id if dep_id is not None else self.deposition_id
        r = requests.delete(
            f"{self._endpoint}/deposit/depositions/{dep_id}",
            auth=self._auth_token,
        )
        # response status
        print(r.status_code)

        # reset class variables to None
        self.title = None
        self.bucket = None
        self.deposition_id = None
        # else:
        #    print(f'Project title {self.title} is still available.')

    # ---------------------------------------------
    # user facing functions/properties
    # ---------------------------------------------

    @property
    def projects(self):
        return self.list_projects()

    def list_projects(self):
        """list projects connected to the supplied ACCESS_KEY

        prints to the screen the "Project Name" and "ID"
        """
        tmp = self._get_depositions()

        if isinstance(tmp, list):
            print("Project Name ---- ID ---- Status ---- Latest Published ID")
            print("---------------------------------------------------------")
            for file in tmp:
                status = "published" if file["submitted"] else "unpublished"
                latest = self._get_latest_record()
                print(f"{file['title']} ---- {file['id']} ---- {status} ---- {latest}")
        else:
            print(" ** need to setup ~/.zenodo_token file ** ")

        return tmp

    @property
    def list_files(self):
        """list files in current project

        prints filenames to screen
        """
        dep = self._get_depositions_by_id()
        if dep is not None:
            print("Files")
            print("------------------------")
            for file in dep["files"]:
                print(file["filename"])
        else:
            print(
                " ** the object is not pointing to a project. Use either .set_project() or .create_deposition() before listing files ** "
            )
            # except UserWarning:
            # warnings.warn("The object is not pointing to a project. Either create a project or explicity set the project'", UserWarning)

    def create_deposition(
        self,
        metadata: ZenodoMetadata,
    ):
        """Creates a new deposition

        After a deposition is created the zenodopy object will point to the deposition

        title is required. If upload_type or description
        are not specified, then default values will be used

        Args:
            title (str): new title of project
            metadata_json (str): path to json file with metadata
        """

        # get request, returns our response
        r = requests.post(
            f"{self._endpoint}/deposit/depositions",
            auth=self._auth_token,
            data=json.dumps({}),
            headers={"Content-Type": "application/json"},
        )

        if r.ok:

            self.deposition_id = r.json()["id"]
            self.bucket = r.json()["links"]["bucket"]

            self.change_metadata(
                metadata=metadata,
            )

        else:
            print(
                "** Project not created, something went wrong. Check that your ACCESS_TOKEN is in ~/.zenodo_token "
            )

    @property
    def deposition(self):
        return self._get_depositions_by_id()

    def set_project(self, dep_id: str | None = None):
        """set the project by id"""
        projects = self._get_depositions()

        if projects is not None:
            project_list = [
                d
                for d in projects
                if self._check_parent_doi(dep_id=dep_id, project_obj=d)
            ]
            if len(project_list) > 0:
                self.title = project_list[0]["title"]
                self.bucket = self._get_bucket_by_id(project_list[0]["id"])
                self.deposition_id = project_list[0]["id"]

        else:
            print(f" ** Deposition ID: {dep_id} does not exist in your projects  ** ")

    def _check_parent_doi(self, dep_id: str, project_obj: dict):
        if project_obj["id"] == int(dep_id):
            return True
        concept_doi = project_obj.get("conceptdoi", None)
        if concept_doi != None:
            return int(dep_id) == int(concept_doi.split(".")[-1])
        return False

    def change_metadata(self, metadata: ZenodoMetadata):
        """
        Change project's metadata.

        Args:
            metadata (ZenodoMetadata): The metadata to update.

        Returns:
            dict: Dictionary with the updated metadata if the request is successful.
                  Raises an error if the request fails.

        This function updates the project's metadata on Zenodo.
        The metadata is sent as a JSON payload to the Zenodo API endpoint using a PUT request.
        If the request is successful, it returns the updated metadata as a dictionary.
        If the request fails, it raises an error with the status of the failed request.
        """

        data = json.dumps({"metadata": metadata.__dict__})

        r = requests.put(
            f"{self._endpoint}/deposit/depositions/{self.deposition_id}",
            auth=self._auth_token,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        if r.ok:
            return r.json()
        else:
            return r.raise_for_status()

    def upload_file(self, file_path: Path | str | None = None, publish=False):
        """upload a file to a project

        Args:
            file_path (str): name of the file to upload
            publish (bool): whether implemente publish action or not
        """
        if file_path is None:
            print("You need to supply a path")
            return
        elif isinstance(file_path, str):
            file_path = Path(file_path)

        if not file_path.exists():
            print(
                f"{file_path} does not exist. Please check you entered the correct path"
            )

        if self.bucket is None:
            print(
                "You need to create a project with zeno.create_deposition() "
                "or set a project zeno.set_project() before uploading a file"
            )
        else:
            bucket_link = self.bucket

            with open(file_path, "rb") as fp:
                # text after last '/' is the filename
                filename = file_path.name
                r = requests.put(
                    f"{bucket_link}/{filename}",
                    auth=self._auth_token,
                    data=fp,
                )

                (
                    print(f"{file_path} successfully uploaded!")
                    if r.ok
                    else print("Oh no! something went wrong")
                )

            if publish:
                return self.publish()

    def update(
        self, metadata: ZenodoMetadata, source: Path | str | None = None, publish=False
    ):
        """update an existing record

        Args:
            source (str): path to directory or file to upload
            publish (bool): whether implemente publish action or not, argument for `upload_file`
        """
        # create a draft deposition
        url_action = self._get_depositions_by_id()["links"]["newversion"]
        r = requests.post(url_action, auth=self._auth_token)
        r.raise_for_status()

        # parse current project to the draft deposition
        new_dep_id = r.json()["links"]["latest_draft"].split("/")[-1]

        # adding this to let new id propogate in the backend
        time.sleep(2)

        self.set_project(new_dep_id)

        time.sleep(5)

        self.change_metadata(metadata=metadata)
        # invoke upload funcions
        source = Path(source) if source is not None else None

        if not source:
            print("You need to supply a path")

        if source.exists() and source.is_file():
            self.upload_file(source, publish=publish)
        else:
            raise FileNotFoundError(f"{source} does not exist")

    def publish(self):
        """publish a record"""
        url_action = self._get_depositions_by_id()["links"]["publish"]
        r = requests.post(url_action, auth=self._auth_token)
        r.raise_for_status()
        return r

    def get_urls_from_doi(self, doi: str | None = None):
        """the files urls for the given doi

        Args:
            doi (str): the doi you want the urls from. Defaults to None.

        Returns:
            list: a list of the files urls for the given doi
        """
        if self._is_doi(doi):
            record_id = self._get_record_id_from_doi(doi)
        else:
            print(f"{doi} must be of the form: 10.5281/zenodo.[0-9]+")

        # get request (do not need to provide access token since public
        r = requests.get(
            f"https://zenodo.org/api/records/{record_id}"
        )  # params={'access_token': ACCESS_TOKEN})
        return [f["links"]["self"] for f in r.json()["files"]]

    def delete_file(self, filename: Path | str | None = None):
        """delete a file from a project

        Args:
            filename (str): the name of file to delete
        """
        bucket_link = self.bucket
        if filename is None:
            print("You need to supply a filename")
            return
        elif isinstance(filename, str):
            filename = Path(filename)

        # with open(file_path, "rb") as fp:
        _ = requests.delete(f"{bucket_link}/{filename}", auth=self._auth_token)
