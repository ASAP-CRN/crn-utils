import pandas as pd
from pathlib import Path
import os, sys
import json

# TODO: Target of folding gcloud operations into wf-common::gcloud_ops.py
from .bucket_util import gcloud_ls, gcloud_hash

repo_root = Path(__file__).resolve().parents[2]
wf_common_path = repo_root.parent / "wf-common" / "util"
sys.path.insert(0, str(wf_common_path))

# from wf-common
from common.gcloud_ops import strip_team_prefix

__all__ = [
    "make_file_metadata",
    "update_data_table_with_gcp_uri",
    "gen_bucket_summary",
    "get_artifacts_df",
    "get_fastqs_df",
    "add_bucket_md5",
]


def summarize_bucket_prefix(
    bucket: str,
    prefix: str,
    source_prefix: str,
    cache_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    List all objects under bucket/prefix and return one row per object.

    Columns: file_name, gcp_uri, bucket_md5, source_prefix

    If cache_path is given and exists and force=False, reads from cache.
    Otherwise walks the bucket, fetches MD5s, and writes the cache if cache_path is given.

    Args:
        bucket: GCS bucket name (without gs://)
        prefix: glob prefix, e.g. "artifacts/**"
        source_prefix: 'fastq' | 'raw' | 'artifact' | 'spatial' | 'curated' —
            identifies which bucket prefix a row came from. Carried through to
            the final artifacts.csv so artifact- and spatial-sourced rows stay
            distinguishable once merged.
        cache_path: optional path for the intermediate CSV cache
        force: if True, re-walk the bucket even if cache exists
    """
    cols = ["file_name", "gcp_uri", "bucket_md5", "source_prefix"]

    if cache_path is not None and cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    listing = gcloud_ls(bucket, prefix, project="dnastack-asap-parkinsons")
    files = [f for f in listing if f and Path(f).name[0] != "." and not f.endswith("/")]

    if not files:
        return pd.DataFrame(columns=cols)

    md5s = gcloud_hash(bucket, prefix)
    df = pd.DataFrame({"gcp_uri": files})
    df["file_name"] = df["gcp_uri"].apply(lambda x: x.split("/")[-1])
    df["bucket_md5"] = df["file_name"].map(md5s)
    df["source_prefix"] = source_prefix
    df = df[cols]

    if cache_path is not None:
        df.to_csv(cache_path, index=False)

    return df


def gen_bucket_summary(
    dl_path: str | Path,
    dataset_id: str,
    env_type: str,
    force: bool = False,
):
    """
    Generate summary of raw or dev bucket contents, writing one intermediate
    CSV per prefix to dl_path. These CSVs serve as the cache for subsequent
    runs (re-read unless force=True).

    For env_type="raw", walks artifacts/**, spatial/**, fastqs/**, and raw/**
    if they exist. 

    Args:
        dl_path: directory to write intermediate CSVs for caching
        dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq")
        env_type: "raw" or "dev"
        force: re-walk bucket even if cached CSVs exist
    """
    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise RuntimeError(
            f"Invalid dataset_id format: {dataset_id}. Expected: team-<team>-<details>"
        )

    if "cohort" in dataset_id and env_type == "raw":
        print(f"No raw bucket file metadata summary required for cohort datasets: {dataset_id}")
        return

    bucket = f"asap-{env_type}-{dataset_id}"
    dl_path = Path(dl_path)

    artifact_cache = dl_path / f"{dataset_name}-{env_type}_artifact_files.csv"
    summarize_bucket_prefix(bucket, "artifacts/**", "artifact", artifact_cache, force=force)

    if env_type == "raw":
        # Create spatial summary only if it exists
        spatial_cache = dl_path / f"{dataset_name}-{env_type}_spatial_files.csv"
        summarize_bucket_prefix(bucket, "spatial/**/*", "spatial", spatial_cache, force=force)

        # NOTE: Assumes the raw bucket only contains fastq.gz or .raw files. Must be updated if other files are encountered.
        for raw_type in ["fastq", "raw"]:
            prefix = "fastqs/**/*.fastq.gz" if raw_type == "fastq" else "raw/**/*.raw"
            raw_cache = dl_path / f"{dataset_name}-{env_type}_{raw_type}_files.csv"
            summarize_bucket_prefix(bucket, prefix, raw_type, raw_cache, force=force)


def make_file_metadata(
    ds_path: str | Path,
    dl_path: str | Path,
    data_df: pd.DataFrame,
):
    """
    Generate file metadata for a dataset.

    Required fields:
        - ds_path: path to the dataset directory
        - dl_path: path to the download directory
        - data_df: DataFrame containing dataset information
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
        ds_path: str | Path,
        release_version: str,
     ) -> pd.DataFrame:
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
    file_metadata_path = os.path.join(ds_path, "file_metadata", "release", release_version)

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
    artifacts = list(dl_path.glob("*artifact_files.csv"))
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
