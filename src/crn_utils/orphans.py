## orphan file for all depricated code

# resource_tools.py
import pandas as pd
import json

# import ijson
from pathlib import Path
import argparse

import requests

from .util.util import read_CDE, read_meta_table

##  HARD CODED VARIABLES

COLLECTION_ID = "ASAP_PMBDS"
STUDY_PREFIX = f"{COLLECTION_ID}_"


__all__ = ["dump_CDE", "dump_file_manifest", "dump_data_dictionary", "dump_readme"]


def dump_CDE(cde_path: Path, version: str = "v3.0") -> pd.DataFrame:
    """
    helper to dump the CDE from its ground-truth google sheet source
    """
    cde_df = read_CDE(metadata_version=version)

    print("read url")
    print(cde_df)
    cde_df.to_csv(cde_path, index=False)
    print(f"dumped CDE to {cde_path}")
    return cde_df


def dump_file_manifest(fm_path: Path, version: str = "v2.0.0") -> None:
    #    https://docs.google.com/document/d/1hNz8ujcSgpDcf6VpFCdhr1G_Pob7o805mNfQIBk8Uis/edit?usp=sharing
    # currently : ASAP CRN Cloud File Manifest - v2.0
    if version == "v1.0.0":  # doc is v2.0 for release v1.0.0
        GOOGLE_SHEET_ID = "1hNz8ujcSgpDcf6VpFCdhr1G_Pob7o805mNfQIBk8Uis"
    elif version == "v2.0.0":  # doc is versioned v3.0 for releae v2.0.0
        GOOGLE_SHEET_ID = "1V0TqEA-EQCrFFLJksnKJLiiQRxBN7j1EJ8AKtrQWHsA"
    else:
        GOOGLE_SHEET_ID = "1V0TqEA-EQCrFFLJksnKJLiiQRxBN7j1EJ8AKtrQWHsA"

    file_manifest_url = (
        f"https://docs.google.com/document/d/{GOOGLE_SHEET_ID}/export?format=pdf"
    )

    response = requests.get(file_manifest_url)
    response.raise_for_status()  # Check if the request was successful

    with open(fm_path, "wb") as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {fm_path}")

    # file_manifest_df = pd.read_csv(file_manifest_url)
    # print("read url")
    # file_manifest_df.to_csv(fm_path, index=False)
    # print(f"dumped file manifest to {fm_path}")
    # return file_manifest_df


def dump_data_dictionary(dd_path: Path, version: str = "v3.0") -> None:
    # https://docs.google.com/document/d/1A65aDHwis5pt_at4tjf0rF292TLw9sSnSXan8MLc4Os/edit?usp=sharing
    # currently ASAP CRN Data Dictionary - v2.1
    if version == "v2.0":
        GOOGLE_SHEET_ID = "1A65aDHwis5pt_at4tjf0rF292TLw9sSnSXan8MLc4Os"
    elif version == "v2.1":
        GOOGLE_SHEET_ID = "1A65aDHwis5pt_at4tjf0rF292TLw9sSnSXan8MLc4Os"
    elif version == "v3.0":
        GOOGLE_SHEET_ID = "1vmFivc9pRFdDRZRF0C3B3lwXFILiMCncNoZSojJSZQI"
    else:
        GOOGLE_SHEET_ID = "1vmFivc9pRFdDRZRF0C3B3lwXFILiMCncNoZSojJSZQI"

    dd_url = f"https://docs.google.com/document/d/{GOOGLE_SHEET_ID}/export?format=pdf"
    response = requests.get(dd_url)
    response.raise_for_status()  # Check if the request was successful

    with open(dd_path, "wb") as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {dd_path}")

    # dd_df = pd.read_csv(dd_url)
    # print("read url")
    # dd_df.to_csv(dd_path, index=False)
    # print(f"dumped file manifest to {dd_path}")
    # return dd_df


def dump_readme(rm_path: Path, version: str = "v2.0") -> None:
    # https://zenodo.org/records/11585274
    if version == "v1.0":
        rm_url = f"https://zenodo.org/records/11585274/files/ASAP%20CRN%20Cloud%20Platform%20README%20-%20v1.0.0.pdf?download=1"
    elif version == "v2.0":
        rm_url = f"https://zenodo.org/records/14270014/files/ASAP%20CRN%20Cloud%20Platform%20Release%20README%20-%20v2.0.0.pdf?download=1&preview=1"
    else:
        rm_url = f"https://zenodo.org/records/14270014/files/ASAP%20CRN%20Cloud%20Platform%20Release%20README%20-%20v2.0.0.pdf?download=1&preview=1"

    response = requests.get(rm_url)
    response.raise_for_status()  # Check if the request was successful

    with open(rm_path, "wb") as file:
        file.write(response.content)
    print(f"PDF downloaded and saved as {rm_path}")


## depricated from file_metadata.py
####################
## helpers
####################

BULK_MODES = ["alignment_mode", "mapping_mode"]


def get_bulk_manifests(
    workflow: str, bucket_name: str, dl_path: Path | str, team_name: str
):
    bulk_modes = BULK_MODES

    for mode in bulk_modes:
        ## UPSTREAM
        analysis = "upstream"
        prefix = f"{workflow}/{analysis}/{mode}"
        bucket_path = f"{bucket_name}/{prefix}"  # dev_bucket_name has gs:// prefix
        file_name = "MANIFEST.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/{analysis}-{mode}-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)

        ## DOWNSTREAM
        analysis = "downstream"
        prefix = f"{workflow}/{analysis}/{mode}"
        bucket_path = f"{bucket_name}/{prefix}"  # dev_bucket_name has gs:// prefix
        file_name = "MANIFEST.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/{analysis}-{mode}-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)

        ## COHORT ANALYSIS
        analysis = "cohort_analysis"
        prefix = f"{workflow}/{analysis}/{mode}"
        bucket_path = f"{bucket_name}/{prefix}"  # dev_bucket_name has gs:// prefix
        file_name = "MANIFEST.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/{analysis}-{mode}-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)

    # just need one sample_list.tsv
    file_name = f"{team_name}.sample_list.tsv"
    remote = f"{bucket_path}/{file_name}"
    local = f"{dl_path}/{analysis}-{mode}-{file_name}"
    # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
    gsutil_cp2(remote, local, directory=False)


def get_sc_manifests(
    workflow: str,
    bucket_name: str,
    dl_path: Path | str,
    team_name: str,
    raw: bool = False,
):
    if raw:
        print("no sample_list.tsv for minor releases or 'platforming exercises'")
        # TODO make this an "other" dataset to skip this step
    else:
        ############# sc_rna_seq datasets
        # get the preprocess file list from workflow/"preprocess/MANIFEST.tsv"
        # for curated files table
        prefix = f"{workflow}/preprocess"
        bucket_path = f"{bucket_name}/{prefix}"  # dev_bucket_name has gs:// prefix
        file_name = "MANIFEST.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/preprocess-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)

        # get the cohort_analysis file list from workflow/"cohort_analysis/"
        # for curated files table
        prefix = f"{workflow}/cohort_analysis"
        bucket_path = f"{bucket_name}/{prefix}"  # dev_bucket_name has gs:// prefix
        file_name = "MANIFEST.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/cohort_analysis-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)

        file_name = f"{team_name}.sample_list.tsv"
        remote = f"{bucket_path}/{file_name}"
        local = f"{dl_path}/cohort_analysis-{file_name}"
        # retval = gsutil_ls2(dev_bucket_name.split("/")[-1], prefix)
        gsutil_cp2(remote, local, directory=False)


# def gen_cohort_fastqs(datasets_table:pd.DataFrame, datasets_path:Path, cohorts:dict):
#     """
#     """
#     # get the preprocess file list from workflow/"preprocess/MANIFEST.tsv"
#     # for curated files table
#     make_cohort_fastqtables(datasets_table, datasets_path, cohorts)


def make_cohort_fastqtables(
    datasets_table: pd.DataFrame, datasets_path: Path, cohorts: dict
):

    for cohort_name, cohort_datasets in cohorts.items():
        if cohort_datasets == []:
            continue
        else:
            dfs = []
            for dataset_name in cohort_datasets:
                ds = datasets_table[datasets_table["dataset_name"] == dataset_name]
                ds_name = ds["full_dataset_name"].values[0]
                ds_path = datasets_path / f"{ds_name}"
                dl_path = ds_path / "intermediate"
                fastq_df = pd.read_csv(dl_path / f"{ds_name}-raw_fastqs.csv")
                # not sure if we need to add the cohort name to the fastq_df and save...
                fastq_df["cohort"] = cohort_name
                fastq_df.to_csv(dl_path / f"{ds_name}-raw_fastqs.csv", index=False)
                dfs.append(fastq_df)

            combined_df = pd.concat(dfs)

            ds = datasets_table[
                (datasets_table["collection"].apply(lambda x: x in cohort_name))
                & (datasets_table["cohort"])
            ]

            cohort_ds_name = ds["full_dataset_name"].values[0]
            # print(f"cohort_ds_name: {cohort_ds_name}")
            # print(f"cohort_name: {cohort_name}")
            assert cohort_ds_name == f"asap-{cohort_name}"

            ds_path = datasets_path / f"{cohort_ds_name}"
            dl_path = ds_path / "intermediate"

            combined_df.to_csv(
                f"{dl_path}/{cohort_ds_name}-raw_fastqs.csv", index=False
            )
            print(f"Saved {cohort_ds_name}-raw_fastqs.csv")


def gen_unitary_fastqs(ds_table: pd.DataFrame, datasets_path: Path):
    """ """
    # get the preprocess file list from workflow/"preprocess/MANIFEST.tsv"
    # for curated files table

    # NOTE: for now looking for the cohort datasets as well
    ##  now we need to go dataset by dataset to get the list of raw files, and list of curated files.
    for idx, ds in ds_table.iterrows():
        dataset_name = ds["full_dataset_name"]
        raw_bucket_name = ds["raw_bucket_name"]

        # team_name = ds['team_name']
        dl_path = datasets_path / f"{dataset_name}/intermediate"

        gen_raw_bucket_summary(raw_bucket_name, dl_path, dataset_name)


# TODO: update the naming to accomodate "urgent", "minor", "major"
def get_minor_manifests(ds_table: pd.DataFrame, datasets_path: Path):
    """
    for minor releases only 'platforming exercise'
    """

    bucket_name = ds["raw_bucket_name"]  # for minor releases

    print(f"Processing {bucket_name}")
    team_name = ds["team_name"]
    workflow = ds["workflow"]
    # print(f"Processing {dataset_name}, {team_name=}")
    # we'll get metadata from the raw bucket at "/metadata/release"

    dl_path = datasets_path / f"{dataset_name}/intermediate"
    dl_path.mkdir(parents=True, exist_ok=True)
    # get_sc_manifests(workflow, bucket_name, dl_path, team_name)
    gen_unitary_fastqs(ds_table, datasets_path)


def gen_unitary_manifests(
    datasets_table: pd.DataFrame, datasets_path: Path, promotion: str
):
    """ """

    ds_table = datasets_table[~datasets_table["cohort"]].copy()
    get_manifests(ds_table, datasets_path, promotion)
    gen_unitary_fastqs(ds_table, datasets_path)


def gen_cohort_manifests(
    datasets_table: pd.DataFrame, datasets_path: Path, cohorts: dict, promotion: str
):
    """ """

    ds_table = datasets_table[datasets_table["cohort"]].copy()
    get_manifests(
        ds_table, datasets_path, promotion
    )  # chort manifest files are in the same place as unitary datasets
    make_cohort_fastqtables(datasets_table, datasets_path, cohorts)


def get_manifests(ds_table: pd.DataFrame, datasets_path: Path, promotion: str):
    """ """
    # get the preprocess file list from workflow/"preprocess/MANIFEST.tsv"
    # for curated files table

    # NOTE: for now looking for the cohort datasets as well
    ##  now we need to go dataset by dataset to get the list of raw files, and list of curated files.
    for idx, ds in ds_table.iterrows():
        dataset_name = ds["full_dataset_name"]
        # collection = ds['collection']
        # collection_name = ds['collection_name']
        # collection_version = ds['collection_version']

        # raw_bucket_name = ds['raw_bucket_name']
        if promotion == "dev":
            bucket_name = ds["dev_bucket_name"]
        elif promotion == "prod":
            bucket_name = ds["prod_bucket_name"]
        elif promotion == "uat":
            bucket_name = ds["uat_bucket_name"]
        elif promotion == "minor":
            bucket_name = ds["raw_bucket_name"]  # for minor releases
            ds["workflow"] = "NA"
        else:
            raise ValueError(f"Invalid promotion: {promotion}")

        print(f"Processing {bucket_name}")
        team_name = ds["team_name"]
        workflow = ds["workflow"]
        # print(f"Processing {dataset_name}, {team_name=}")
        # we'll get metadata from the raw bucket at "/metadata/release"

        dl_path = datasets_path / f"{dataset_name}/intermediate"
        dl_path.mkdir(parents=True, exist_ok=True)

        if workflow == "pmdbs_bulk_rnaseq":
            get_bulk_manifests(workflow, bucket_name, dl_path, team_name)
        elif workflow == "NA":
            pass
        else:
            get_sc_manifests(workflow, bucket_name, dl_path, team_name)


def make_manifest_tables(
    release_version: str, root_path: Path | str, promotion: str, major: bool = True
):
    """
    Generate the manifest tables for `unitary` (individual) datasets and `cohort` datasets
    for curated files and raw fastq files.
    promotion="dev", "prod","uat"
    """
    root_path = Path(root_path)
    release_path_base = root_path / "releases" / release_version
    artifact_path = release_path_base / "release-artifacts"
    datasets_path = release_path_base / "datasets" / promotion

    datasets_table = pd.read_csv(artifact_path / "datasets.csv")
    # unitary_datasets
    gen_unitary_manifests(datasets_table, datasets_path, promotion)

    if major:
        # cohort_datasets
        # make fastqtables for cohort datasets
        # load the cohorts.json
        with open(artifact_path / "cohorts.json", "r") as f:
            cohorts = json.load(f)

        gen_cohort_manifests(datasets_table, datasets_path, cohorts, promotion)

    # datasets_path_base = release_path_base / "datasets"
    # copy most recent artifacts to release_path_base i.e.
    # WHAT DO i actually need here...
    # just the "curated_files.csv" and "raw_files.csv" for each dataset NOT the intermediates
    # defer to prep_release2_compose_dataset_artifacts.py
    # shutil.copy2tree(datasets_path, datasets_path_base)


def make_minor_manifest_tables(
    release_version: str, root_path: Path | str, promotion: str
):
    """
    Generate the manifest tables for `unitary` (individual) datasets and `cohort` datasets
    for curated files and raw fastq files.
    promotion="prod","uat"
    """
    root_path = Path(root_path)
    release_path_base = root_path / "releases" / release_version
    artifact_path = release_path_base / "release-artifacts"
    datasets_path = release_path_base / "datasets" / promotion

    datasets_table = pd.read_csv(artifact_path / "datasets.csv")
    # unitary_datasets
    gen_unitary_manifests(datasets_table, datasets_path, promotion)


####################
## file_metadata API
####################
# if __name__ == "__main__":
#     # loads .jsons from .json path input
#     # datasets:dict,
#     # release_version:str,
#     # collection_version:dict,
#     # root_path:Path|str|None = None):
#     import argparse

#     parser = argparse.ArgumentParser(description="prep relase 0: dataset refs")

#     parser.add_argument(
#         "--release",
#         dest="release_version",
#         type=str,
#         default=LATEST_RELEASE,
#         help="release version",
#     )

#     parser.add_argument(
#         "--root",
#         dest="root_path",
#         type=str,
#         default="None",
#         help="root path of release artifacts",
#     )

#     args = parser.parse_args()

#     make_manifest_tables(args.release_version, args.root_path)
