import subprocess

# # create functions to transfer files to GCP....
def gsutil_ls( bucket_name, prefix):
    """
    prints the files in a GCS bucket matching a given prefix. 

    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix to filter objects.


    Returns:
       list of files

    """

    project = "dnastack-asap-parkinsons"
    cmd = f"gsutil -u {project} ls gs://{bucket_name}/{prefix}"

    prefix = prefix + "/" if not prefix.endswith("/") else prefix
    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")

    return result.stdout.split("\n")

def gsutil_ls2( bucket_name:str, prefix:str, project: str|None = None):
    """
    prints the files in a GCS bucket matching a given prefix. 

    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix to filter objects.

    Returns:
        list of files
    """

    # project = "dnastack-asap-parkinsons"
    if project is None:
        cmd = f"gsutil ls \"gs://{bucket_name}/{prefix}\""
    else:
        cmd = f"gsutil -u {project} ls \"gs://{bucket_name}/{prefix}\""

    # print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded")
        # print(f"gsutil command succeeded: {result.stdout}")
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")

    return result.stdout.split("\n")


def gsutil_cp( source, destination, directory=False):
    """
    copies the files to a GCS bucket path

    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory

    Returns:
       None.
    """

    project = "dnastack-asap-parkinsons"


    if directory:
        cmd = f"gsutil -u {project} cp -r {source} {destination}"
    else:
        cmd = f"gsutil -u {project} cp {source} {destination}"

    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")
    return result.stdout

def gsutil_cp2( source, destination, directory:bool=False, project: str|None = None):
    """
    copies the files to a GCS bucket path

    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory
        project (str): GCP project name

    Returns:
       None.
    """

    if project is None:
        project_flag = ""
    else:
        project_flag = f"-u {project}"


    if directory:
        cmd = f"gsutil {project_flag} cp -r {source} {destination}"
    else:
        cmd = f"gsutil {project_flag} cp {source} {destination}"

    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")
    return result.stdout

def gsutil_mv( source, destination, directory=False):
    """
    moves the files between paths / GCS bucket path


    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory

    Returns:
       None.
    """

    project = "dnastack-asap-parkinsons"
    

    if directory:
        cmd = f"gsutil -u {project} mv -r {source} {destination}"
    else:
        cmd = f"gsutil -u {project} mv {source} {destination}"
    
    print(cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")

    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")

    return result.stdout

def gsutil_rsync( source, destination):
    """
    rsync the files between paths / GCS bucket path


    Args:
        source (str): local file path or GCS bucket path
        destination (str): local file path or GCS bucket path
    Returns:
       stdout.=
    """

    project = "dnastack-asap-parkinsons"
    

    cmd = f"gsutil -u {project} -m rsync -d {source} {destination}"
    print(cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")

    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")

    return result.stdout



def authenticate_with_service_account(key_file_path):
    """
    Authenticates with a Google Cloud service account using a key file.

    Args:
        key_file_path (str): The path to the service account key file.
    """

    cmd = f"gcloud auth activate-service-account --key-file={key_file_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    return result 


# %%

def gsutil_rm( destination, directory=False):
    """
    copies the files to a GCS bucket path

    Args:
        destination (str): local file path or GCS bucket path
        directory (bool): is the source or destination a directory

    Returns:
       None.
    """

    project = "dnastack-asap-parkinsons"

    if directory:
        cmd = f"gsutil -u {project} rm -r {destination}"
    else:
        cmd = f"gsutil -u {project} rm  {destination}"

    print(cmd)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"gsutil command succeeded: {result.stdout}")
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")
    return result.stdout
