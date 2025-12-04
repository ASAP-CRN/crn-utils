import os
import subprocess

__all__ = [
    "gcloud_ls",
    "gcloud_rsync",
    "gcloud_mv",
    "gcloud_rm",
    "authenticate_with_service_account",
]


# create functions to list, rsync and delete files into GCP
# Updated to use gcloud instead of gsutil
def gcloud_ls(bucket_name, prefix, project: str | None = None):
    """
    prints the files in a GCS bucket matching a given prefix.

    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix to filter objects.
        project (str | None): GCP project name. If None, uses default project [dnastack-asap-parkinsons]

    Returns:
       list of files

    """
    default_project = "dnastack-asap-parkinsons"
    if project is None:
        project = default_project

    cmd = f"gcloud storage ls gs://{bucket_name}/{prefix} --billing-project={project}"

    print(f"IN: {cmd}")
    prefix = prefix + "/" if not prefix.endswith("/") else prefix
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"OUT: {result.stdout}")
    if result.returncode == 0:
        pass
    else:
        print(f"gcloud command failed: {result.stderr}")

    return result.stdout.split("\n")


def gcloud_rsync(
    source, destination, directory: bool = False, project: str | None = None
):
    """
    rsync files to/from local paths or GCS buckets

    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
        directory (bool): indicates if source the input is a directory
        project (str | None): GCP project name. If None, uses default project [dnastack-asap-parkinsons]

    Returns:
       None.
    """

    default_project = "dnastack-asap-parkinsons"
    if project is None:
        project = default_project

    if os.path.isdir(source) or source.endswith("/"):
        cmd = f"gcloud storage rsync --recursive '{source}' '{destination}' --billing-project={project}"
    else:
        cmd = (
            f"gcloud storage cp '{source}' '{destination}' --billing-project={project}"
        )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gcloud command succeeded: {cmd}")
    else:
        print(f"gcloud command failed: {result.stderr}")
    return result.stdout


def gcloud_mv(source, destination, directory=False, project: str | None = None):
    """
    moves the files between os.path.join(paths, GCS) bucket path


    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory
        project (str | None): GCP project name. If None, uses default project [dnastack-asap-parkinsons]

    Returns:
       None.
    """

    default_project = "dnastack-asap-parkinsons"
    if project is None:
        project = default_project

    if directory:
        cmd = f"gcloud storage mv --recursive '{source}' '{destination}' --billing-project={project}"
    else:
        cmd = (
            f"gcloud storage mv '{source}' '{destination}' --billing-project={project}"
        )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gcloud command succeeded: {cmd}")
    else:
        print(f"gcloud command failed: {result.stderr}")

    return result.stdout


# NOTE: this is deprecated
def authenticate_with_service_account(key_file_path):
    """
    Authenticates with a Google Cloud service account using a key file.

    Args:
        key_file_path (str): The path to the service account key file.
    """

    cmd = f"gcloud auth activate-service-account --key-file={key_file_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return result


def gcloud_rm(destination, directory=False, project: str | None = None):
    """
    copies the files to a GCS bucket path

    Args:
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory
        project (str | None): GCP project name. If None, uses default project [dnastack-asap-parkinsons]

    Returns:
       None.
    """

    default_project = "dnastack-asap-parkinsons"
    if project is None:
        project = default_project

    if directory:
        cmd = (
            f"gcloud storage rm --recursive '{destination}' --billing-project={project}"
        )
    else:
        cmd = f"gcloud storage rm '{destination}' --billing-project={project}"

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gcloud command succeeded: {cmd}")
    else:
        print(f"gcloud command failed: {result.stderr}")
    return result.stdout
