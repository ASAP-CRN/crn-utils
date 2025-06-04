# from google.oauth2.service_account import Credentials
# from google.cloud import storage

import subprocess


# # create functions to login to GCP and get hashes....
def get_md5_hashes(bucket_name, prefix):
    """
    Fetches MD5 hashes of objects in a GCS bucket matching a given prefix.

    Args:
        project_id (str): Your Google Cloud Project ID.
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix to filter objects.

    Returns:
        A dictionary mapping object names to their MD5 hashes.
    # # Example usage:
    # project_id = "your-project-id"
    # bucket_name = "your-bucket-name"
    # prefix = "path/to/your/files/*.gz"

    # md5_dict = get_md5_hashes(project_id, bucket_name, prefix)

    """
    # storage_client = storage.Client(credentials=credentials, project="dnastack-asap-parkinsons")
    # bucket = storage_client.bucket(bucket_name)

    # md5_hashes = {}

    # for blob in bucket.list_blobs(prefix=prefix):
    #     md5_hashes[blob.name] = blob.md5_hash

    project = "dnastack-asap-parkinsons"
    cmd = f'gsutil -u {project} hash -h "gs://{bucket_name}/{prefix}"'
    print(cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        md5_hashes = extract_md5_from_details2_lines(result.stdout.splitlines())
        return md5_hashes
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")
        return result


def get_md5_hashes_full(bucket_name, prefix):

    project = "dnastack-asap-parkinsons"
    cmd = f'gsutil -u {project} hash -h "gs://{bucket_name}/{prefix}"'
    print(cmd)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        md5_hashes = extract_md5_from_details2_lines_full(result.stdout.splitlines())
        return md5_hashes
    else:
        # raise RuntimeError(f"gsutil command failed: {result.stderr}")
        print(f"gsutil command failed: {result.stderr}")
        return result


# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """
#     Downloads a blob from the bucket.

#     Args:
#     bucket_name (str): The name of the bucket.
#     source_blob_name (str): The name of the blob in the bucket.
#     destination_file_name (str): The local file path to which the blob should be downloaded.
#     """
#     # Initialize a client
#     storage_client = storage.Client()

#     # Get the bucket
#     bucket = storage_client.bucket(bucket_name)

#     # Get the blob
#     blob = bucket.blob(source_blob_name)

#     # Download the blob to a local file
#     blob.download_to_filename(destination_file_name)

#     print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("gs://"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
    return md5s


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details2(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
    return md5s


def extract_md5_from_details2ful(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
    return md5s


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details2_df(md5_file):
    md5s = pd.DataFrame(columns=["file_name", "md5"])
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s = md5s.append(
                    {"file_name": current_file, "md5": line.split(":")[1].strip()},
                    ignore_index=True,
                )
                md5s[current_file] = line.split(":")[1].strip()
    return md5s


def extract_md5_from_details3(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip("Hashes [hex]: for")
                # remove " for "
                # current_file = current_file.lstrip(" for")
                current_file = current_file.strip()
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
            elif line.startswith("Hash (crc32c)"):
                pass
            else:
                print(f"cruff: {line.strip()}")

    return md5s


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details2_lines(lines):
    md5s = {}
    current_file = None
    for line in lines:
        if line.startswith("Hashes [hex]"):
            current_file = line.strip().rstrip(":")
            current_file = current_file.split("/")[-1]
        if "Hash (md5)" in line:
            md5s[current_file] = line.split(":")[1].strip()
    return md5s


def extract_md5_from_details2_lines_full(lines):
    md5s = {}
    current_file = None
    for line in lines:
        if line.startswith("Hashes [hex]"):
            current_file = line.split(" for ")[-1].rstrip(":")
        if "Hash (md5)" in line:
            md5s[current_file] = line.split(":")[1].strip()
            md5_list.append((current_file, md5s[current_file]))
    return md5s


# Function to parse the file to extract crc32c and filenames
def extract_crc32c_from_details2(md5_file):
    crcs = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (crc32c)" in line:
                crcs[current_file] = line.split(":")[1].strip()
    return crcs


# Function to parse the file to extract crc32c and filenames
def extract_hashes_from_gcloudstorage(source_hash):

    crcs = {}
    md5s = {}

    with open(source_hash, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:

            if line.startswith("crc32c_hash:"):
                curr_crc = line.split(":")[1].strip()

            elif line.startswith("md5_hash:"):
                curr_md5 = line.split(":")[1].strip()

            elif line.startswith("url:"):
                current_file = line.split("/")[-1].strip()
                crcs[current_file] = curr_crc
                md5s[current_file] = curr_md5
            # else:
            #     print(f'cruff:{line.strip()}')

    return crcs, md5s


# Function to parse the file to extract crc32c and filenames
def extract_hashes_from_gsutil(source_hash):

    crcs = {}
    md5s = {}

    with open(source_hash, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (crc32c)" in line:
                crcs[current_file] = line.split(":")[1].strip()
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()

    return crcs, md5s
