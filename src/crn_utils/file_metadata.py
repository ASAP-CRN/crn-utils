import pandas as pd
from pathlib import Path
import os, sys
import json

from .bucket_util import gcloud_ls
from .checksums import get_md5_hashes

repo_root = Path(__file__).resolve().parents[2]
wf_common_path = repo_root.parent / "wf-common" / "util"
sys.path.insert(0, str(wf_common_path))

# from wf-common
from common import strip_team_prefix

__all__ = [
    "make_file_metadata",
    "update_data_table_with_gcp_uri",
    "update_spatial_table_with_gcp_uri",
    "gen_bucket_summary",
    "gen_spatial_bucket_summary",
    "get_artifacts_df",
    "get_fastqs_df",
    "get_spatial_df",
    "add_bucket_md5",
]

def make_file_metadata(
    ds_path: str | Path,
    dl_path: str | Path,
    data_df: pd.DataFrame,
    spatial: bool = False,
):
    """
    Generate file metadata for a dataset.

    Required fields:
        - ds_path: path to the dataset directory
        - dl_path: path to the download directory
        - data_df: DataFrame containing dataset information
        - spatial: boolean indicating if spatial data should be included

    """

    dl_path = Path(dl_path)
    ds_path = Path(ds_path)

    dataset_name = ds_path.name
    team_name = dataset_name.split("-")[0]

    print(f"Processing {dataset_name}, {team_name=}")
    # we'll get metadata from the raw bucket at "/metadata/release"

    data_df = data_df[
        [
            "ASAP_sample_id",
            "ASAP_team_id",
            "ASAP_dataset_id",
            "sample_id",  # sample_id gets clobbered.
            "replicate",
            "batch",
            "file_name",
            "file_MD5",
            "file_type",
        ]
    ]

    data_df["sample_name"] = (
        data_df["ASAP_sample_id"].astype(str) + "_" + data_df["replicate"].astype(str)
    )

    asap_dataset_id = data_df["ASAP_dataset_id"].unique()[0]
    team_id = data_df["ASAP_team_id"].unique()[0]

    # add contributed artifacts
    artifacts_df = get_artifacts_df(dl_path, asap_dataset_id, team_id)

    if artifacts_df.shape[0] > 0:
        artifacts_df.to_csv(os.path.join(dl_path, "artifacts.csv"), index=False)
    else:
        print(f"No artifact files found for {dataset_name}")

    ############################################
    ## raw files
    ############################################

    samp_df = data_df.copy()
    samp_df["project_id"] = team_name

    fastq_df = get_fastqs_df(dl_path, asap_dataset_id, team_id)
    files_df = pd.concat([fastq_df, artifacts_df])

    if spatial:
        spatial_df = get_spatial_df(dl_path, asap_dataset_id, team_id)
        spatial_df.to_csv(os.path.join(dl_path, "spatial_files.csv"), index=False)
        files_df = pd.concat([fastq_df, spatial_df])

    merge_cols = ["gcp_uri", "file_name", "bucket_md5"]

    df = samp_df.merge(files_df[merge_cols].copy(), on="file_name", how="left")
    keep_cols = [
        "ASAP_dataset_id",
        "ASAP_team_id",
        "ASAP_sample_id",
        "file_name",
        "replicate",
        "batch",
        "file_MD5",
        "file_type",
        "gcp_uri",
        "sample_name",
        "bucket_md5",
    ]
    df = df.loc[:, keep_cols]
    # # check md5s
    check = pd.Index(df.loc[:, "file_MD5"] == df.loc[:, "bucket_md5"])
    if not check.all():
        print(f"MD5s do not match for {dataset_name}")
        mismatch_counts = df.loc[~check, "file_name"].value_counts()
        for file_name, count in mismatch_counts.items():
            print(f"  {file_name} ({count} occurrence{'s' if count > 1 else ''})")

    # now export the combined_df to a csv file
    df.to_csv(os.path.join(dl_path, "raw_files.csv"), index=False)


def update_data_table_with_gcp_uri(
        data_df: pd.DataFrame, 
        ds_path: str | Path):
    """
    Add GCP URIs to DATA table.
    Handles pooled/multiplexed files where multiple samples share the same file_name.

    Required fields:
    - data_df: DataFrame containing the DATA table information, including "file_name" column
    - ds_path: path to the dataset directory, where file_metadata/raw_files.csv is located

    Returns:
    - Updated DataFrame with "gcp_uri" column added based on the mapping from raw_files.csv

    """
    ds_path = Path(ds_path)
    file_metadata_path = os.path.join(ds_path, "file_metadata")

    raw_files = pd.read_csv(os.path.join(file_metadata_path, "raw_files.csv"))
    
    # Deduplicate by file_name before merging
    # Multiple samples share the same physical files in pooled sequencing
    raw_files_unique = raw_files[["file_name", "gcp_uri"]].drop_duplicates(subset=["file_name"])
    
    # Ensure we're not creating duplicates
    initial_rows = len(data_df)
    data_df = data_df.merge(raw_files_unique, on="file_name", how="left", validate="many_to_one")
    
    if len(data_df) != initial_rows:
        print(f"WARNING: Row count changed from {initial_rows} to {len(data_df)} during merge!")
    
    print(f"Updated 'DATA.csv' with gcp_uri ({len(data_df)} rows)")

    return data_df


def update_spatial_table_with_gcp_uri(
    spatial_df: pd.DataFrame, ds_path: str | Path, spatial_subtype: str = "other"
):
    """
    Update SPATIAL table with GCP URIs and MD5 hashes.

    Required fields:
        - spatial_df: DataFrame containing the SPATIAL table information
        - ds_path: path to the dataset directory, where file_metadata/raw_files.csv is located
        - spatial_subtype: type of spatial data ("visium", "geomx", "cosmx")

    Returns:
        - Updated DataFrame with "gcp_uri" and "md5" columns added based on the mapping from raw_files.csv and spatial_files.csv
        - The specific columns updated depend on the spatial_subtype
    """

    ds_path = Path(ds_path)
    file_metadata_path = os.path.join(ds_path, "file_metadata")

    raw_files = pd.read_csv(os.path.join(file_metadata_path, "raw_files.csv"))
    spatial_files = pd.read_csv(
        os.path.join(file_metadata_path, f"{ds_path.name}-spatial_files.csv")
    )
    # make mappers for spatial files
    spatial_file_gcp_mapper = dict(
        zip(spatial_files["file_name"], spatial_files["spatial_files"])
    )
    spatial_file_md5_mapper = dict(
        zip(spatial_files["file_name"], spatial_files["bucket_md5"])
    )

    raw_files = raw_files[["file_name", "gcp_uri", "bucket_md5"]]
    raw_file_gcp_mapper = dict(zip(raw_files["file_name"], raw_files["gcp_uri"]))
    raw_file_md5_mapper = dict(zip(raw_files["file_name"], raw_files["bucket_md5"]))

    # combine mappers
    spatial_file_gcp_mapper.update(raw_file_gcp_mapper)
    spatial_file_md5_mapper.update(raw_file_md5_mapper)

    if spatial_subtype == "visium":
        left_ons = ["visium_cytassist"]
    elif spatial_subtype == "geomx":
        left_ons = ["geomx_config", "geomx_dsp_config", "geomx_annotation_file"]
    elif spatial_subtype == "cosmx":
        left_ons = [] # currently no files in SPATIAL table for CosMx datasets
    else:
        raise ValueError(f"Unsupported spatial subtype: {spatial_subtype}")

    for left_on in left_ons:
        spatial_df[f"{left_on}_md5"] = spatial_df[left_on].map(spatial_file_md5_mapper)
        spatial_df[f"{left_on}_gcp_uri"] = spatial_df[left_on].map(
            spatial_file_gcp_mapper
        )

    print(f"Updated 'SPATIAL.csv' with gcp_uris")
    return spatial_df

# Replaces gen_raw_bucket_summary and gen_dev_bucket_summary
# gen_dev_bucket_summary seemed to be a partial copy of gen_raw_bucket_summary
# and had incostiencies such as listing fastq's, that aren't expected in the dev bucket.
def gen_bucket_summary(
    dl_path: str | Path,
    dataset_id: str,
    env_type: str,
    flatten: bool = False,
):
    """
    Generate summary of raw or dev bucket contents and save to dl_path.

    Required fields in config:
        - dl_path: path to save the summary files
        - dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq")
        - env_type: environment type ("raw" or "dev")
        - flatten: whether to use a flattened prefix for gcloud_ls (True or False)
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise RuntimeError(
            f"Invalid dataset_id format: {dataset_id}. Expected format: team-<team_name>-<dataset_details>")

    if ("cohort" in dataset_name) & (env_type == "raw"):
        print(f"No raw bucket for cohort datasets: {dataset_name}")
        return

    # Get bucket name and path
    bucket_name = f"asap-{env_type}-{dataset_id}"
    dl_path = Path(dl_path)
    bucket_path = bucket_name.split("/")[-1]

    # --- Artifacts ---
    artifacts = gcloud_ls(bucket_path, "artifacts/**", project="dnastack-asap-parkinsons")
    artifact_files = [
        f for f in artifacts if f != "" and Path(f).name[0] != "." and f[-1] != "/"
    ]
    if artifact_files:
        bucket_files_md5 = get_md5_hashes(bucket_path, "artifacts/**")
        artifact_files_df = pd.DataFrame(artifact_files, columns=["artifact_files"])
        artifact_files_df["file_name"] = artifact_files_df["artifact_files"].apply(lambda x: x.split("/")[-1])
        artifact_files_df["bucket_md5"] = artifact_files_df["file_name"].map(bucket_files_md5)
        artifact_files_df.to_csv(dl_path / f"{dataset_name}-{env_type}_artifact_files.csv", index=False)
        with open(dl_path / f"{dataset_name}-{env_type}_artifacts-md5s.json", "w") as f:
            json.dump(bucket_files_md5, f)
    else:
        print(f"No artifact files found for {dataset_name}")

    # --- Raw files (only in raw env)---
    # TODO: need to list raw files beyond raw and fastq extensions
    if env_type == "raw":
        raw_types = ["raw", "fastq"]

        wrote_raw_files = 0
        for raw_type in raw_types:
            if raw_type == "fastq":
                prefix = "fastqs/*.fastq.gz" if flatten else "fastqs/**/*.fastq.gz"
            else:
                prefix = "raw/*.raw" if flatten else "raw/**/*.raw"

            raw_files = [f for f in gcloud_ls(bucket_path, prefix, project="dnastack-asap-parkinsons") if f != ""]
            if raw_files:
                print(f"Found {len(raw_files)} {raw_type} raw files for {dataset_name}")
                raw_files_df = pd.DataFrame(raw_files, columns=["raw_files"])
                raw_files_df["file_name"] = raw_files_df["raw_files"].apply(lambda x: x.split("/")[-1])
                bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
                raw_files_df["bucket_md5"] = raw_files_df["file_name"].map(bucket_files_md5)

                # write file list and md5s to file
                wrote_raw_files += 1
                if wrote_raw_files == 1:
                    raw_files_df.to_csv(dl_path / f"{dataset_name}-{env_type}_{raw_type}_files.csv", index=False)
                    with open(dl_path / f"{dataset_name}-{env_type}_{raw_type}_files-md5s.json", "w") as f:
                        json.dump(bucket_files_md5, f)
                else:
                    #append to existing file if multiple raw_types are found
                    raw_files_df.to_csv(dl_path / f"{dataset_name}-{env_type}_raw_files.csv", index=False, mode='a', header=False)
                    with open(dl_path / f"{dataset_name}-{env_type}_raw_files-md5s.json", "r") as f:
                        existing_md5s = json.load(f)
                    new_md5s = get_md5_hashes(bucket_path, prefix)
                    existing_md5s.update(new_md5s)
                    with open(dl_path / f"{dataset_name}-{env_type}_raw_files-md5s.json", "w") as f:
                        json.dump(existing_md5s, f)
            else:
                print(f"No {raw_type} raw files found for {dataset_name}")


def gen_spatial_bucket_summary(
    dl_path: str | Path, 
    dataset_id: str
):
    
    """
    Generate summary of spatial files in raw bucket and save to dl_path.

    Required fields in config:
        - dl_path: path to save the summary files
        - dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq")
    """

    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise ValueError(f"Dataset ID [{dataset_id}] does not start with expected 'team-' prefix.")


    if "cohort" in dataset_id:
        print(f"No raw bucket for cohort datasets: {dataset_id}")

    else:
        # Get bucket name and path
        raw_bucket_name = f"asap-raw-{dataset_id}"
        dl_path = Path(dl_path)
        bucket_path = raw_bucket_name.split("/")[-1]
        prefix = f"spatial/**/*"
        spatial_files_ = gcloud_ls(
            bucket_path, prefix, project="dnastack-asap-parkinsons"
        )
        spatial_files = [f for f in spatial_files_ if f != ""]

        if len(spatial_files) > 0:
            bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
            spatial_files_df = pd.DataFrame(spatial_files, columns=["spatial_files"])

            spatial_files_df["file_name"] = spatial_files_df["spatial_files"].apply(
                lambda x: x.split("/")[-1]
            )
            spatial_files_df["bucket_md5"] = spatial_files_df["file_name"].map(
                bucket_files_md5
            )
            spatial_files_df.to_csv(
                os.path.join(dl_path, f"{dataset_name}-spatial_files.csv"), index=False
            )
            # merge in md5s.
            # dump md5s to file
            with open(
                os.path.join(dl_path, f"{dataset_name}-spatial_files-md5s.json"), "w"
            ) as f:
                json.dump(bucket_files_md5, f)
        else:
            print(f"No spatial files found for {dataset_name}")


####################
def get_artifacts_df(dl_path: str | Path, 
                     asap_dataset_id: str, 
                     asap_team_id: str):
    """ 
    Looks for files matching "*-artifact_files.csv" in the given dl_path,
    reads the first one it finds, and processes it to create an artifacts DataFrame

    Required fields:
        - dl_path: path to download summary files from bucket
        - asap_dataset_id: ASAP dataset ID (e.g. DS_PMDBS_0004)
        - asap_team_id: ASAP team ID (e.g. TEAM_SMITH)

    Returns a DataFrame with columns specified in keep_cols
    """
    dl_path = Path(dl_path)

    keep_cols = [
        "ASAP_dataset_id",
        "ASAP_team_id",
        "artifact_type",
        "file_name",
        "timestamp",
        "workflow",
        "workflow_version",
        "gcp_uri",  # change to gcp_uri
        "bucket_md5",
    ]

    dl_path = Path(dl_path)
    artifacts = list(dl_path.glob("*-artifact_files.csv"))
    if len(artifacts) > 0:
        artifact = artifacts[0]
        print(f"Processing {artifact.name}")
        df = pd.read_csv(artifact)

        df["exclude"] = (
            df["artifact_files"].apply(lambda x: "cellranger_counts" in x)
            | df["artifact_files"].apply(lambda x: ".git" in x)
            | df["artifact_files"].apply(lambda x: ".DS_Store" in x)
        )
        # now concatenate the dataframes
        df = df[~df["exclude"]]
        df["ASAP_dataset_id"] = asap_dataset_id
        df["ASAP_team_id"] = asap_team_id
        df["timestamp"] = "NA"
        df["workflow"] = "NA"
        df["workflow_version"] = "NA"
        df["artifact_type"] = "contributed"
        df["gcp_uri"] = df["artifact_files"]

        return df[keep_cols]
    else:
        print(f"no artifact files found for {dl_path.parent.name}")
        df = pd.DataFrame(columns=keep_cols)
        return df


def get_fastqs_df(
        dl_path: str | Path, 
        asap_dataset_id: str, 
        asap_team_id: str
        ) -> pd.DataFrame:
    """
    Looks for raw file CSVs written by gen_bucket_summary in the given dl_path,
    reads all matches, concatenates them, and returns a DataFrame with metadata
    about the raw files (fastq or .raw).
    It adds columns for ASAP dataset and team IDs, timestamps, workflow information,
    artifact type, GCP URI, and bucket MD5 checksums.

    gen_bucket_summary writes files with these naming conventions:
      - First raw_type found: "{dataset_name}-{env_type}_{raw_type}_files.csv"
        e.g. "*-raw_fastq_files.csv" or "*-raw_raw_files.csv"
      - Subsequent raw_types (appended): "{dataset_name}-{env_type}_raw_files.csv"
    This function uses a glob of "*-raw_*_files.csv" to match all variants.

    Required fields:
        - dl_path: path to download summary files from bucket
        - asap_dataset_id: ASAP dataset ID (e.g. DS_PMDBS_0004)
        - asap_team_id: ASAP team ID (e.g. TEAM_SMITH)

    Returns a DataFrame with columns specified in keep_cols,
    or an empty DataFrame with those columns if no matching files are found.

    """

    dl_path = Path(dl_path)

    keep_cols = [
        "ASAP_dataset_id",
        "ASAP_team_id",
        "artifact_type",
        "file_name",
        "timestamp",
        "workflow",
        "workflow_version",
        "gcp_uri",
        "bucket_md5",
    ]

    # gen_bucket_summary writes raw file CSVs with one of two specific names
    # (column "raw_files"), depending on which raw_type is found first:
    #   "{dataset_name}-raw_raw_files.csv"   (when .raw files are found first)
    #   "{dataset_name}-raw_fastq_files.csv" (when fastq.gz files are found first)
    # If both types exist, the second is appended to the first file (same name, no new file).
    # We must NOT match "*-raw_artifact_files.csv" (column "artifact_files"), hence the
    # explicit patterns below rather than the broader "*-raw_*_files.csv".
    RAW_FILE_GLOBS = ["*-raw_raw_files.csv", "*-raw_fastq_files.csv"]
    raw_file_csvs = []
    for pattern in RAW_FILE_GLOBS:
        raw_file_csvs.extend(dl_path.glob(pattern))

    if len(raw_file_csvs) > 0:
        dfs = []
        for raw_file_csv in raw_file_csvs:
            print(f"Processing {raw_file_csv.name}")
            df = pd.read_csv(raw_file_csv)
            df["ASAP_dataset_id"] = asap_dataset_id
            df["ASAP_team_id"] = asap_team_id
            df["timestamp"] = "NA"
            df["workflow"] = "NA"
            df["workflow_version"] = "NA"
            df["artifact_type"] = "contributed"
            df["gcp_uri"] = df["raw_files"]
            dfs.append(df[keep_cols])

        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["file_name"])
    else:
        print(f"no raw files found for {dl_path.parent.name}")
        return pd.DataFrame(columns=keep_cols)


def get_spatial_df(
        dl_path: str | Path, 
        asap_dataset_id: str, 
        asap_team_id: str) -> pd.DataFrame:
    """
    Looks for files matching "*-spatial_files.csv" in the given dl_path,
    reads the first one it finds, and processes it to create a DataFrame
    with metadata about the spatial files.
    It adds columns for ASAP dataset and team IDs, timestamps, workflow information,
    artifact type, GCP URI, and bucket MD5 checksums.

    Required fields:
        - dl_path: path to download summary files from bucket
        - asap_dataset_id: ASAP dataset ID (e.g. DS_PMDBS_0004)
        - asap_team_id: ASAP team ID (e.g. TEAM_SMITH)

    Returns a DataFrame with columns specified in keep_cols,
    or an empty DataFrame with those columns if no matching files are found.
    
    """
    dl_path = Path(dl_path)

    keep_cols = [
        "ASAP_dataset_id",
        "ASAP_team_id",
        "artifact_type",
        "file_name",
        "timestamp",
        "workflow",
        "workflow_version",
        "gcp_uri",  # change to gcp_uri
        "bucket_md5",
    ]

    spatial_files = list(dl_path.glob("*-spatial_files.csv"))
    if len(spatial_files) > 0:
        spatial_file = spatial_files[0]  # HACK
        print(f"Processing {spatial_file.name}")
        # spatial_files,file_name,bucket_md5
        spatial_df = pd.read_csv(spatial_file)
        # now export the combined_df to a csv file
        spatial_df["ASAP_dataset_id"] = asap_dataset_id
        spatial_df["ASAP_team_id"] = asap_team_id
        spatial_df["timestamp"] = "NA"
        spatial_df["workflow"] = "NA"
        spatial_df["workflow_version"] = "NA"
        spatial_df["artifact_type"] = "contributed"
        spatial_df["gcp_uri"] = spatial_df["spatial_files"]

        spatial_df = spatial_df[keep_cols]
        # rename "spatial_files" to "raw_files"
        # spatial_df.rename(columns={"spatial_files": "raw_files"}, inplace=True)
        return spatial_df
    else:
        print(f"no images files found for {dl_path.parent.name}")
        return pd.DataFrame(columns=keep_cols)


def add_bucket_md5(
        df_file_metadata: pd.DataFrame, 
        dl_path: str | Path):
    """
    Adds a column for bucket MD5 checksums to the given DataFrame.

    Required fields
        - df_file_metadata: DataFrame containing file metadata
        - dl_path: path to download summary files from bucket

    Returns the DataFrame with an added "bucket_md5" column.

    """

    dl_path = Path(dl_path)

    md5_files = list(dl_path.glob(f"*-md5s.json"))
    if len(md5_files) == 0:
        print(f"no md5 files found for {dl_path.parent.name}")
        df_file_metadata["bucket_md5"] = "NA"
        return df_file_metadata

    md5_mapper = {}
    for file in md5_files:
        with open(file, "r") as f:
            md5s = json.load(f)
            md5_mapper.update(md5s)
    df_file_metadata["bucket_md5"] = df_file_metadata["file_name"].map(md5_mapper)
    return df_file_metadata
