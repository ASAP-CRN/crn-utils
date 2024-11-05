import pandas as pd
import json
# import ijson
from pathlib import Path
import argparse

import requests

from util import read_CDE, read_meta_table

##  HARD CODED VARIABLES

COLLECTION_ID = "ASAP_PMBDS"
STUDY_PREFIX = f"{COLLECTION_ID}_"




def dump_CDE(cde_path:Path, version:str="v3.0") -> pd.DataFrame:
    """
    helper to dump the CDE from its ground-truth google sheet source
    """
    cde_df = read_CDE(metadata_version=version)

    print("read url")
    cde_df.to_csv(cde_path, index=False)
    print(f"dumped CDE to {cde_path}")
    return cde_df


def dump_file_manifest(fm_path:Path) -> None:
    #    https://docs.google.com/document/d/1hNz8ujcSgpDcf6VpFCdhr1G_Pob7o805mNfQIBk8Uis/edit?usp=sharing 
    # currently : ASAP CRN Cloud File Manifest - v2.0
    GOOGLE_SHEET_ID = "1hNz8ujcSgpDcf6VpFCdhr1G_Pob7o805mNfQIBk8Uis"
    file_manifest_url = f"https://docs.google.com/document/d/{GOOGLE_SHEET_ID}/export?format=pdf"

    response = requests.get(file_manifest_url)
    response.raise_for_status()  # Check if the request was successful

    with open(fm_path, 'wb') as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {fm_path}")

    # file_manifest_df = pd.read_csv(file_manifest_url)
    # print("read url")
    # file_manifest_df.to_csv(fm_path, index=False)
    # print(f"dumped file manifest to {fm_path}")
    # return file_manifest_df


def dump_data_dictionary(dd_path:Path) -> None:
    # https://docs.google.com/document/d/1A65aDHwis5pt_at4tjf0rF292TLw9sSnSXan8MLc4Os/edit?usp=sharing
    # currently ASAP CRN Data Dictionary - v2.1
    GOOGLE_SHEET_ID = "1A65aDHwis5pt_at4tjf0rF292TLw9sSnSXan8MLc4Os"
    dd_url = f"https://docs.google.com/document/d/{GOOGLE_SHEET_ID}/export?format=pdf"
    response = requests.get(dd_url)
    response.raise_for_status()  # Check if the request was successful

    with open(dd_path, 'wb') as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {dd_path}")

    # dd_df = pd.read_csv(dd_url)
    # print("read url")
    # dd_df.to_csv(dd_path, index=False)
    # print(f"dumped file manifest to {dd_path}")
    # return dd_df

def dump_readme(rm_path:Path) -> None:
    # https://zenodo.org/records/11585274
    rm_url = f"https://zenodo.org/records/11585274/files/ASAP%20CRN%20Cloud%20Platform%20README%20-%20v1.0.0.pdf?download=1"
    response = requests.get(rm_url)
    response.raise_for_status()  # Check if the request was successful

    with open(rm_path, 'wb') as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {rm_path}")
