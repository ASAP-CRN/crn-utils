# create zenodo DOI programmically

# example 
example = [{'name':'Doe, John', 'affiliation': 'Zenodo'}, 
    {'name':'Smith, Jane', 'affiliation': 'Zenodo', 'orcid': '0000-0002-1694-233X'}, 
    {'name': 'Kowalski, Jack', 'affiliation': 'Zenodo', 'gnd': '170118215'}]


import os
import json
import requests
from dotenv import load_dotenv
import argparse

ZENODO_BASE_URL = "https://zenodo.org"

__all__ = ["setup_zenodo", "create_deposition", "upload_file", "update_metadata", "publish_deposition", "main"]

def setup_zenodo():
    """
    Load environment variables using python-dotenv and ensure that ZENODO_API_TOKEN is set.
    Returns:
        str: The API token.
    """
    load_dotenv()
    api_token = os.getenv("ZENODO_API_TOKEN")
    if not api_token:
        raise ValueError("ZENODO_API_TOKEN not found in environment variables. Please set it in your .env file.")
    print("Zenodo API token found.")
    return api_token


def create_deposition(api_token):
    """
    Create a new empty deposition on Zenodo.
    
    Args:
        api_token (str): Your Zenodo API token.
        
    Returns:
        dict: The JSON response from Zenodo containing the deposition details.
    """
    url = f"{ZENODO_BASE_URL}/api/deposit/depositions"
    params = {'access_token': api_token}
    headers = {"Content-Type": "application/json"}
    
    # Create an empty deposition.
    response = requests.post(url, params=params, json={}, headers=headers)
    if response.status_code != 201:
        raise Exception(f"Error creating deposition: {response.status_code}\n{response.text}")
    
    deposition = response.json()
    deposition_id = deposition.get("id")
    print(f"Deposition created with ID: {deposition_id}")
    return deposition


def upload_file(deposition, file_path, api_token):
    """
    Upload a file to the deposition.
    
    Args:
        deposition (dict): The deposition JSON (returned from create_deposition).
        file_path (str): Path to the file you wish to upload.
        api_token (str): Your Zenodo API token.
        
    Returns:
        dict: The JSON response from the file upload.
    """
    bucket_url = deposition.get("links", {}).get("bucket")
    if not bucket_url:
        raise Exception("Bucket URL not found in deposition response.")
    
    filename = os.path.basename(file_path)
    upload_url = f"{bucket_url}/{filename}"
    params = {'access_token': api_token}
    
    print(f"Uploading file '{filename}' to deposition bucket...")
    with open(file_path, "rb") as fp:
        response = requests.put(upload_url, data=fp, params=params)
    
    if response.status_code not in (200, 201):
        raise Exception(f"Error uploading file: {response.status_code}\n{response.text}")
    
    upload_response = response.json()
    print("File uploaded successfully:")
    print(json.dumps(upload_response, indent=2))
    return upload_response


def update_metadata(deposition, api_token, metadata_payload):
    """
    Update the deposition metadata with the provided fields.
    
    Args:
        deposition (dict): The deposition JSON.
        api_token (str): Your Zenodo API token.
        metadata_payload (dict): A dictionary containing the metadata fields (without the outer "metadata" key).
        
    Returns:
        dict: The updated deposition JSON.
    """
    deposition_id = deposition.get("id")
    if not deposition_id:
        raise Exception("Deposition ID not found.")
    
    url = f"{ZENODO_BASE_URL}/api/deposit/depositions/{deposition_id}"
    params = {'access_token': api_token}
    headers = {"Content-Type": "application/json"}
    
    # Wrap the provided metadata in the expected outer "metadata" key.
    data_payload = {"metadata": metadata_payload}
    
    print("Updating deposition metadata...")
    response = requests.put(url, params=params, data=json.dumps(data_payload), headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error updating metadata: {response.status_code}\n{response.text}")
    
    updated_dep = response.json()
    print("Metadata updated:")
    print(json.dumps(updated_dep, indent=2))
    return updated_dep


def publish_deposition(deposition, api_token):
    """
    Publish the deposition so that a DOI is reserved and the upload is final.
    
    Args:
        deposition (dict): The deposition JSON.
        api_token (str): Your Zenodo API token.
        
    Returns:
        dict: The JSON response from the publish action.
    """
    deposition_id = deposition.get("id")
    if not deposition_id:
        raise Exception("Deposition ID not found.")
    
    url = f"{ZENODO_BASE_URL}/api/deposit/depositions/{deposition_id}/actions/publish"
    params = {'access_token': api_token}
    
    print(f"Publishing deposition with ID: {deposition_id} ...")
    response = requests.post(url, params=params)
    if response.status_code != 202:
        raise Exception(f"Error publishing deposition: {response.status_code}\n{response.text}")
    
    published_dep = response.json()
    doi = published_dep.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    if doi:
        print(f"Deposition published. DOI: {doi}")
    else:
        print("Deposition published but DOI not found in the response.")
    return published_dep


def main():
    # Set up and get the API token.
    api_token = setup_zenodo()
    
    # Define a dict mapping file paths to their corresponding metadata.
    # Ensure that each file exists in your local directory.
    # Alternatively can define this in a json file and load it.
    files_metadata = {
        "test.txt": {
            "upload_type": "publication",
            "publication_type": "other",
            "resource_type": "publication",
            "title": "Test Deposition",
            "creators": [
                {"name": "Doe, John"}
            ],
            "publication_date": "2025-02-06"
        },
        "another_file.txt": {
            "upload_type": "dataset",
            "publication_type": "other",
            "resource_type": "dataset",
            "title": "Another Deposition",
            "creators": [
                {"name": "Smith, Jane"}
            ],
            "publication_date": "2025-02-06"
        }
    }
    
    # Process each file and its associated metadata separately.
    for file_path, metadata in files_metadata.items():
        print(f"\n=== Processing deposition for file: {file_path} ===")
        
        # Create a new deposition.
        deposition = create_deposition(api_token)
        
        # Upload the file.
        upload_file(deposition, file_path, api_token)
        
        # Update the deposition metadata with the provided metadata.
        deposition = update_metadata(deposition, api_token, metadata)
        
        # Publish the deposition to reserve a DOI.
        published_deposition = publish_deposition(deposition, api_token)
        
        # Print the final published deposition details.
        print("\nFinal Published Deposition Details:")
        print(json.dumps(published_deposition, indent=2))

def instructions():
    print("INSTRUCTIONS:")
    print("(1) Have all of your data available locally.")
    print("(2) Make a dictionary mapping data to metadata.")
    print("(3) Place a Zenodo API key with deposit:actions & deposite:write access in a .env file.")
    print("(4) Install dependencies 'pip install python-dotenv'.")
    print("(5) Comment out this function and uncomment main, run it and check your Zenodo account for published DOIs.")
    print("** Please walk through the code yourself, run a test trial on a single upload before running all, and feel ask any questions. **")


if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="A command-line tool to update tables from ASAP_CDEv1 to ASAP_CDEv2.")
    
    # Add arguments
    parser.add_argument("--dataset", default=".",
                        help="long_dataset_name: <team_name>_<source>_<dataset_name> i.e. the folder name in 'asap-crn-metadata'. Defaults to the current working directory.")
    parser.add_argument("--tables", default=Path.cwd(),
                        help="Path to the directory containing meta TABLES. Defaults to the current working directory.")
    parser.add_argument("--schema", default=Path.cwd(),
                        help="Path to the directory containing ASAP_ID schema.csv. Defaults to the current working directory.")
    parser.add_argument("--map", default=Path.cwd(),
                        help="Path to the directory containing path to mapper.json files. Defaults to the current working directory.")
    parser.add_argument("--suf", default="test",
                        help="suffix to mapper.json. Defaults to 'map' i.e. ASAP_{samp,subj}_map.json")
    parser.add_argument("--outdir", default="v2",
                        help="Path to the directory containing CSD.csv. Defaults to the current working directory.")
    
    # Parse the arguments
    args = parser.parse_args()

    asap_ids_schema = read_CDE_asap_ids(local_path=args.cde)
    table_root = Path(args.tables) 
    export_root= Path(args.outdir)

    # instructions()
    # main()



    process_meta_files( args.dataset,
                        table_root, 
                        asap_ids_schema,
                        args.map, 
                        args.suf,
                        export_path = export_root)