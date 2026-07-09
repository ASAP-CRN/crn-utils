# NOTE: This script will be deprecated/folded into wf-common's gcloud_ops.py in the near future.

import os
import subprocess
import logging
from pathlib import Path

__all__ = [
    "gcloud_ls",
    "gcloud_rsync",
    "gcloud_mv",
    "gcloud_rm",
    "gcloud_hash",
    "authenticate_with_service_account",
]


def gcloud_ls(bucket_name: str, prefix: str, project: str | None = None):
    """
    Lists the files in a GCS bucket_name/prefix path
    
    Args:
        bucket_name (str): GCS bucket name, without the "gs://" scheme.
        prefix (str): Object glob relative to the bucket root (e.g. "artifacts/**").
        project (str | None): GCP project name. If None, defaults to "dnastack-asap-parkinsons".
        
    Returns:
        list: List of files in the specified GCS bucket/prefix path. Empty list
        if no files are found or if the command fails.
    """
    default_project = "dnastack-asap-parkinsons"
    if project is None:
        project = default_project

    cmd = f"gcloud storage ls gs://{bucket_name}/{prefix} --billing-project={project}"
    logging.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        if "matched no objects" in result.stderr:
            logging.info(f"No objects found for gs://{bucket_name}/{prefix}")
        else:
            logging.error(f"gcloud command failed: {result.stderr.strip()}")
        return []

    files = result.stdout.split("\n")
    logging.info(f"Found {len(files)} objects for gs://{bucket_name}/{prefix}")
    return files


def gcloud_rsync(
    source: str | Path,
    destination: str | Path,
    directory: bool = False,
    project: str | None = None,
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

    if not isinstance(source, str):
        source = str(source)

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


def gcloud_mv(
    source: str | Path,
    destination: str | Path,
    directory=False,
    project: str | None = None,
):
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
    # probably not nescessary but defensive
    if not isinstance(source, str):
        source = str(source)
    if not isinstance(destination, str):
        destination = str(destination)

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


def gcloud_rm(destination: str | Path, directory=False, project: str | None = None):
    """
    copies the files to a GCS bucket path

    Args:
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory
        project (str | None): GCP project name. If None, uses default project [dnastack-asap-parkinsons]

    Returns:
       None.
    """
    if not isinstance(destination, str):
        destination = str(destination)

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


def gcloud_hash(
    bucket_name: str,
    prefix: str,
    project: str | None = None,
) -> dict[str, str]:
    """
    Fetch MD5 hashes for GCS objects matching a prefix.

    Wraps ``gcloud storage hash --hex`` and parses the line-oriented output
    into a mapping from object basename to hex MD5.

    Parameters
    ----------
    bucket_name : str
        GCS bucket name, without the ``gs://`` scheme.
    prefix : str
        Object glob relative to the bucket root (e.g. ``"artifacts/**"``).
    project : str or None, optional
        Billing project. Defaults to ``"dnastack-asap-parkinsons"`` when None.

    Returns
    -------
    dict of str to str
        Mapping ``{file_name: md5_hex}`` for each matched object. Empty when
        the prefix matches no objects or the command fails.

    Notes
    -----
    Objects uploaded via parallel-composite upload have no MD5 stored in GCS
    (only ``crc32c``) and will be absent from the returned mapping.
    """
    if project is None:
        project = "dnastack-asap-parkinsons"

    command = [
        "gcloud", "storage", "hash", "--hex",
        f"gs://{bucket_name}/{prefix}",
        f"--billing-project={project}",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        logging.error("gcloud storage hash failed: %s", exc.stderr.strip())
        return {}

    md5s: dict[str, str] = {}
    curr_md5: str | None = None
    for line in result.stdout.splitlines():
        if line.startswith("md5_hash:"):
            curr_md5 = line.split(":", 1)[1].strip()
        elif line.startswith("url:") and curr_md5 is not None:
            md5s[line.split("/")[-1].strip()] = curr_md5
    return md5s