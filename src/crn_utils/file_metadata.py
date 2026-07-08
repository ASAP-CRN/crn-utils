import sys
import logging
import pandas as pd
from pathlib import Path

# TODO: Target of folding gcloud operations into wf-common::gcloud_ops.py
from .bucket_util import gcloud_ls, gcloud_hash, gcloud_rsync

repo_root = Path(__file__).resolve().parents[2]
wf_common_path = repo_root.parent / "wf-common" / "util"
sys.path.insert(0, str(wf_common_path))

# from wf-common
from common.gcloud_ops import strip_team_prefix

__all__ = [
    "update_data_table_with_bucket_metadata",
    "gen_raw_bucket_summary",
    "gen_dev_bucket_summary",
    "process_curated_files",
    "get_artifacts_df",
    "get_raw_df",
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


def gen_raw_bucket_summary(
    dl_path: str | Path,
    dataset_id: str,
    force: bool = False,
):
    """
    Generate summary of raw bucket contents, writing one intermediate CSV per
    prefix to dl_path. These CSVs serve as the cache for subsequent runs
    (re-read unless force=True).

    Walks artifacts/**, spatial/**, fastqs/**, and raw/** if they exist.

    Assumptions (revisit if they change):
    - A dataset's raw bucket contains either fastqs/ OR raw/, never both.
    - The raw directory (raw/ or fastqs/) only contains fastq.gz or .raw files.

    Args:
        dl_path: directory to write intermediate CSVs for caching
        dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq")
        force: re-walk bucket even if cached CSVs exist
    """
    if dataset_id.startswith("team-"):
        dataset_name = strip_team_prefix(dataset_id)
    else:
        raise RuntimeError(
            f"Invalid dataset_id format: {dataset_id}. Expected: team-<team>-<details>"
        )

    if "cohort" in dataset_id:
        logging.info(f"No raw bucket file metadata summary required for cohort datasets: {dataset_id}")
        return

    bucket = f"asap-raw-{dataset_id}"
    dl_path = Path(dl_path)

    artifact_cache = dl_path / f"{dataset_name}_artifact_files.csv"
    summarize_bucket_prefix(bucket, "artifacts/**", "artifact", artifact_cache, force=force)

    # Spatial intermediate will be folded into artifacts.csv if it exists
    spatial_cache = dl_path / f"{dataset_name}_spatial_files.csv"
    summarize_bucket_prefix(bucket, "spatial/**/*", "spatial", spatial_cache, force=force)

    raw_cache = dl_path / f"{dataset_name}_raw_files.csv"
    for raw_type, prefix in [("fastq", "fastqs/**/*.fastq.gz"), ("raw", "raw/**/*.raw")]:
        summarize_bucket_prefix(bucket, prefix, raw_type, raw_cache, force=force)


def gen_dev_bucket_summary(
    dl_path: str | Path,
    dataset_id: str,
    workflow_name: str,
    curated_workflows: list[str],
    force: bool = False,
) -> pd.DataFrame:
    """
    Generate summary of dev bucket contents for a dataset's curated workflow
    outputs, writing an intermediate CSV cache to dl_path. Works for both
    team- and cohort- dataset_ids.

    Unlike the raw bucket (fixed artifacts/spatial/fastq/raw prefixes), the dev
    bucket has exactly one relevant prefix per dataset: the workflow that
    produced its curated outputs. workflow_name is looked up per-dataset by the
    caller from the dataset release table; curated_workflows is the allowlist of
    workflow names this release pipeline knows how to catalogue.

    Columns: file_name, gcp_uri, bucket_md5, source_prefix (source_prefix is
    always "curated" here).

    Args:
        dl_path: directory to write the intermediate CSV for caching
        dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq" or "cohort-pmdbs-bulk-rnaseq")
        workflow_name: name of the workflow whose outputs to catalogue (e.g., pmdbs_sc_rnaseq), or "NA"
        curated_workflows: allowlist of workflow names this pipeline supports
        force: re-walk bucket even if a cached CSV exists
    """
    cols = ["file_name", "gcp_uri", "bucket_md5", "source_prefix"]

    dataset_name = strip_team_prefix(dataset_id)

    if workflow_name == "NA":
        logging.info(f"Skipping {dataset_id} as workflow is NA")
        return pd.DataFrame(columns=cols)
    if workflow_name not in curated_workflows:
        logging.info(f"Skipping {dataset_id} as workflow {workflow_name} is not implemented")
        return pd.DataFrame(columns=cols)

    bucket = f"asap-dev-{dataset_id}"
    dl_path = Path(dl_path)
    cache_path = dl_path / f"{dataset_name}_curated_files.csv"

    return summarize_bucket_prefix(bucket, f"{workflow_name}/**", "curated", cache_path, force=force)


def process_curated_files(
    dataset_dir: str | Path,
    dataset_id: str,
    workflow_name: str,
    curated_df: pd.DataFrame,
    release_version: str,
) -> None:
    """
    Merges the dev-bucket curated-file inventory (file_name, gcp_uri, bucket_md5,
    source_prefix — from gen_dev_bucket_summary) with downloaded MANIFEST.tsv
    metadata (workflow_version, timestamp), rebases gcp_uri from the dev bucket
    to the curated (production) bucket, and writes curated_files.csv.

    NOTE on the gcp_uri rebase: MANIFEST.tsv and the workflow outputs it
    describes are downloaded/hashed from the DEV bucket (gs://asap-dev-...),
    since that's where they live at release-prep time — promotion to the
    curated bucket (gs://asap-curated-...) is a separate, later step. gcp_uri
    in the final output is therefore a *prospective* path: where the file will
    live once promoted, not where it currently lives.

    NOTE on ASAP_dataset_id/ASAP_team_id for cohorts: cohort dev/curated
    buckets contain files exclusive to the cohort (not attributable to any
    single constituent dataset), so there is no single-constituent STUDY.csv
    identity to source from. For cohort dataset_ids, use the cohort dataset_id
    itself as ASAP_dataset_id and the literal string "Cohort" as ASAP_team_id,
    rather than reading STUDY.csv (which for a cohort is itself a
    concatenation of multiple constituents' rows and has no single answer).

    Args:
        dataset_dir: path to the dataset (or cohort) directory, e.g.
            {dss_meta_root}/datasets/{dataset_name}
        dataset_id: dataset identifier (e.g., "team-smith-pmdbs-sn-rnaseq" or
            "cohort-pmdbs-bulk-rnaseq")
        workflow_name: name of the workflow whose outputs were catalogued
            (e.g., pmdbs_sc_rnaseq)
        curated_df: DataFrame from gen_dev_bucket_summary, columns
            [file_name, gcp_uri, bucket_md5, source_prefix]
        release_version: Release version string (e.g., "v4.0.0")
    """
    dataset_dir = Path(dataset_dir)
    logging.info(f"Start: generate curated_files.csv for {dataset_id}")

    if curated_df.empty:
        logging.info(f"Skipping {dataset_id} as no curated files were found")
        return

    logging.info(f"Dataset: {dataset_id} has {len(curated_df)} curated files")
    curated_df = curated_df.copy()

    file_metadata_path = dataset_dir / "file_metadata" / "release" / release_version
    file_metadata_path.mkdir(parents=True, exist_ok=True)

    # gcp_uri is the full dev-bucket object path (bucket + workflow + subdirs +
    # filename) — strip the known "gs://asap-dev-{dataset_id}/{workflow_name}/"
    # prefix to get the remainder (subdirs/filename), used both for the
    # archive filter and to rebase onto the curated bucket.
    dev_prefix = f"gs://asap-dev-{dataset_id}/{workflow_name}/"
    curated_prefix = f"gs://asap-curated-{dataset_id}/{workflow_name}/"
    curated_df["_remainder"] = curated_df["gcp_uri"].str.slice(len(dev_prefix))

    # Archive filter: only the *first* subdirectory under the workflow root is
    # checked — a file sitting directly under the workflow root is never
    # treated as archived, regardless of its own name.
    curated_df["_first_dir"] = curated_df["_remainder"].apply(
        lambda r: r.split("/")[0] if "/" in r else ""
    )
    curated_df = curated_df[~curated_df["_first_dir"].str.startswith("archive")]

    # Get manifest files (needed to source workflow_version/timestamp)
    manifest_rows = curated_df[curated_df["file_name"] == "MANIFEST.tsv"]

    manifests_df = pd.DataFrame()
    for _, manifest_row in manifest_rows.iterrows():
        remote = manifest_row["gcp_uri"]  # dev-bucket path -- not yet promoted
        remainder_cleaned = manifest_row["_remainder"].replace("/", "-")
        local = file_metadata_path / remainder_cleaned

        gcloud_rsync(remote, local, directory=False)
        logging.info(f"Downloaded {remote} to {local}")

        local_df = pd.read_csv(local, sep="\t")
        local_df = local_df.dropna(subset=["workflow_version"], how="all")
        local_df = local_df.rename(columns={"filename": "file_name"})
        manifests_df = pd.concat([manifests_df, local_df])

        local.unlink()
        logging.info(f"Deleted temporary file {local}")

    # Merge manifest data. No suffix collision here -- curated_df has no
    # "workflow" column at this point, so the manifest's own "workflow" column
    # passes through unrenamed and is used directly for the check below.
    curated_df = curated_df.merge(manifests_df, on="file_name", how="left")

    # Validate workflow consistency (manifest-declared workflow vs. the one we
    # walked the dev bucket for)
    if "workflow" in curated_df.columns:
        checked = curated_df.loc[curated_df["workflow"].notna(), "workflow"]
        assert (checked == workflow_name).all(), (
            f"MANIFEST workflow does not match expected workflow "
            f"'{workflow_name}' for {dataset_id}"
        )

    # Rebase gcp_uri from dev bucket to curated (production) bucket
    curated_df["gcp_uri"] = curated_prefix + curated_df["_remainder"]

    # Add dataset identifiers.
    if dataset_id.startswith("cohort-"):
        asap_dataset_id = dataset_id
        asap_team_id = "Cohort"
    else:
        study_file = dataset_dir / "metadata" / "release" / release_version / "STUDY.csv"
        study_df = pd.read_csv(study_file)
        asap_dataset_id = study_df["ASAP_dataset_id"].unique()[0]
        asap_team_id = study_df["ASAP_team_id"].unique()[0]

    curated_df["ASAP_dataset_id"] = asap_dataset_id
    curated_df["ASAP_team_id"] = asap_team_id
    curated_df["workflow"] = workflow_name  # overwrite/normalize for rows with no manifest match

    # Column set aligned with artifacts.csv (ASAP_dataset_id, ASAP_team_id,
    # file_name, gcp_uri, bucket_md5, source_prefix), plus workflow/
    # workflow_version/timestamp which carry real signal here (sourced from
    # MANIFEST.tsv), per the file_metadata refactor's D8 decision.
    output_cols = [
        "ASAP_dataset_id", "ASAP_team_id", "file_name", "gcp_uri", "bucket_md5",
        "source_prefix", "workflow", "workflow_version", "timestamp",
    ]
    curated_df = curated_df[output_cols]

    # Save curated files metadata
    outfile_curated = file_metadata_path / "curated_files.csv"
    curated_df.to_csv(outfile_curated, index=False)

    logging.info(f"End: generate curated_files.csv for {dataset_id}")


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
        logging.warning(f"Row count changed from {initial_rows} to {len(data_df)} during merge!")

    logging.info(f"Updated 'DATA.csv' with gcp_uri and file_MD5 ({len(data_df)} rows)")
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
        logging.warning(f"No artifact or spatial files found for {dl_path}")
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
        logging.warning(f"No raw files found for {dl_path}")
        return pd.DataFrame(columns=keep_cols)

    df = pd.read_csv(raw_file_csvs[0])
    return df[keep_cols].drop_duplicates(subset=["file_name"])

