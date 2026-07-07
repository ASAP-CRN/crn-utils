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
    "get_raw_df",
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
    
    Assumptions (revisit if they change):
    - A dataset's raw bucket contains either fastqs/ OR raw/, never both.
    - The raw directory (raw/ or fastqs/) only contains fastq.gz or .raw files.
    
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

    if env_type == "raw":
        
        artifact_cache = dl_path / f"{dataset_name}_artifact_files.csv"
        summarize_bucket_prefix(bucket, "artifacts/**", "artifact", artifact_cache, force=force)
        
        # Spatial intermediate will be folded into artifacts.csv if it exists
        spatial_cache = dl_path / f"{dataset_name}_spatial_files.csv"
        summarize_bucket_prefix(bucket, "spatial/**/*", "spatial", spatial_cache, force=force)

        raw_cache = dl_path / f"{dataset_name}_raw_files.csv"
        for raw_type, prefix in [("fastq", "fastqs/**/*.fastq.gz"), ("raw", "raw/**/*.raw")]:
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

    raw_df = get_raw_df(dl_path)
    files_df = pd.concat([raw_df, artifacts_df])

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


def update_data_table_with_bucket_metadata(
    data_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add gcp_uri and file_MD5 to the DATA table from the raw-files inventory
    DataFrame. file_MD5 is computed from the bucket (bucket_md5), not
    contributor-submitted — any pre-existing file_MD5 column is dropped and
    replaced. Handles pooled/multiplexed files where multiple samples share
    the same file_name.
    """
    uri_map = (
        inventory_df[["file_name", "gcp_uri", "bucket_md5"]]
        .drop_duplicates(subset=["file_name"])
        .rename(columns={"bucket_md5": "file_MD5"})
    )

    data_df = data_df.drop(columns=["file_MD5"], errors="ignore")

    initial_rows = len(data_df)
    data_df = data_df.merge(uri_map, on="file_name", how="left", validate="many_to_one")

    if len(data_df) != initial_rows:
        print(f"WARNING: Row count changed from {initial_rows} to {len(data_df)} during merge!")

    print(f"Updated 'DATA.csv' with gcp_uri and file_MD5 ({len(data_df)} rows)")
    return data_df


def get_artifacts_df(
    dl_path: str | Path,
    asap_dataset_id: str,
    asap_team_id: str,
) -> pd.DataFrame:
    """
    Read the artifact (and spatial, if it exists) intermediate CSVs written by 
    gen_bucket_summary and return a single combined DataFrame with ASAP identifiers.
    
    Args:
        dl_path: path to download summary files from bucket (e.g., file_metadata/release/v5.0.0)
        asap_dataset_id: ASAP dataset ID (e.g., DS_PMDBS_0004)
        asap_team_id: ASAP team ID (e.g., TEAM_SMITH)
    """
    dl_path = Path(dl_path)
    
    keep_cols = [
        "ASAP_dataset_id", "ASAP_team_id", "file_name", "gcp_uri", "bucket_md5", "source_prefix",
    ]
    
    ARTIFACT_GLOBS = ["*_artifact_files.csv", "*_spatial_files.csv"]
    
    csvs = []
    for pattern in ARTIFACT_GLOBS:
        csvs.extend(dl_path.glob(pattern))

    if not csvs:
        print(f"no artifact or spatial files found for {dl_path}")
        return pd.DataFrame(columns=keep_cols)

    dfs = []
    for csv_path in csvs:
        df = pd.read_csv(csv_path)
        
        # NOTE/TODO: These files are excluded from the promote_raw_data script for
        # storage reasons and to encourage users to use DNAstack's processed data.
        # This exclusion list is rather fragile and should be revisited: see ticket BIOS-2421
        exclude = (
            df["gcp_uri"].str.contains("cellranger_counts", na=False)
            | df["gcp_uri"].str.contains("bam_files", na=False)
            | df["gcp_uri"].str.contains(r"\.git", na=False)
            | df["gcp_uri"].str.contains(r"\.DS_Store", na=False)
        )
        df = df[~exclude].copy()
        df["ASAP_dataset_id"] = asap_dataset_id
        df["ASAP_team_id"] = asap_team_id
        dfs.append(df[keep_cols])

    return pd.concat(dfs, ignore_index=True)


def get_raw_df(dl_path: str | Path) -> pd.DataFrame:
    """
    Read the raw-files intermediate CSV written by gen_bucket_summary()
    
    Args:
        dl_path: path to download summary files from bucket (e.g., file_metadata/release/v5.0.0)
    """
    dl_path = Path(dl_path)
    keep_cols = ["file_name", "gcp_uri", "bucket_md5"]
    raw_file_csvs = list(dl_path.glob("*_raw_files.csv"))
    if not raw_file_csvs:
        print(f"no raw files found for {dl_path}")
        return pd.DataFrame(columns=keep_cols)

    df = pd.read_csv(raw_file_csvs[0])
    return df[keep_cols].drop_duplicates(subset=["file_name"])

