import pandas as pd
from pathlib import Path
import os, sys
import json

from .bucket_util import gcloud_ls
from .checksums import get_md5_hashes

__all__ = [
    "make_file_metadata",
    "update_data_table_with_gcp_uri",
    "update_spatial_table_with_gcp_uri",
    "gen_raw_bucket_summary",
    "gen_dev_bucket_summary",
    "gen_spatial_bucket_summary",
    "get_artifacts_df",
    "get_fastqs_df",
    "get_spatial_df",
    "add_bucket_md5",
]

# define collections, collection names and datasets


def make_file_metadata(
    ds_path: str | Path,
    dl_path: str | Path,
    data_df: pd.DataFrame,
    spatial: bool = False,
):

    dl_path = Path(dl_path)
    ds_path = Path(ds_path)

    dataset_name = ds_path.name
    team_name = dataset_name.split("-")[0]
    short_dataset_name = "-".join(dataset_name.split("-")[1:])

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

    dataset_id = data_df["ASAP_dataset_id"].unique()[0]
    team_id = data_df["ASAP_team_id"].unique()[0]

    # add contributed artifacts
    artifacts_df = get_artifacts_df(dl_path, dataset_id, team_id)

    if artifacts_df.shape[0] > 0:
        artifacts_df.to_csv(os.path.join(dl_path, "artifacts.csv"), index=False)
    else:
        print(f"no artifact files found for {dataset_name}")

    ############################################
    ## raw files
    ############################################

    samp_df = data_df.copy()
    samp_df["project_id"] = team_name

    fastq_df = get_fastqs_df(dl_path, dataset_id, team_id)
    # fastq_df["file_name"] = fastq_df["raw_files"].apply(lambda x: x.split("/")[-1])
    files_df = pd.concat([fastq_df, artifacts_df])

    if spatial:
        spatial_df = get_spatial_df(dl_path, dataset_id, team_id)
        spatial_df.to_csv(os.path.join(dl_path, "spatial_files.csv"), index=False)
        # print(f"spatial_df: {spatial_df.columns}")
        # print(f"fastq_df: {fastq_df.columns}")
        files_df = pd.concat([fastq_df, spatial_df])

    merge_cols = ["gcp_uri", "file_name", "bucket_md5"]

    df = samp_df.merge(files_df[merge_cols].copy(), on="file_name", how="left")
    # df = add_bucket_md5(df, dl_path)
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
        print(f"{df.loc[~check,"file_name"].to_list()}")

    # now export the combined_df to a csv file
    df.to_csv(os.path.join(dl_path, "raw_files.csv"), index=False)


def update_data_table_with_gcp_uri(data_df: pd.DataFrame, ds_path: str | Path):
    """
    Add GCP URIs to DATA table.
    Handles pooled/multiplexed files where multiple samples share the same file_name.
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
    spatial_df: pd.DataFrame, ds_path: str | Path, visium: bool = True
):
    """ """
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

    if visium:
        left_ons = ["visium_cytassist"]
    else:
        left_ons = ["geomx_config", "geomx_dsp_config", "geomx_annotation_file"]

    for left_on in left_ons:
        spatial_df[f"{left_on}_md5"] = spatial_df[left_on].map(spatial_file_md5_mapper)
        spatial_df[f"{left_on}_gcp_uri"] = spatial_df[left_on].map(
            spatial_file_gcp_mapper
        )

    print(f"Updated 'SPATIAL.csv' with gcp_uris")
    return spatial_df


def gen_raw_bucket_summary(
    raw_bucket_name: str,
    dl_path: str | Path,
    dataset_name: str,
    flatten: bool = False,
    raw_type: str = "fastq",
):
    if "cohort" in dataset_name:
        print(f"No raw bucket for cohort datasets: {dataset_name}")
        # need to join with the cohort-pmdbs-sn-rnaseq raw files

    else:

        ## OTHER and everything else...
        # create a list of the curated files in /artifacts
        dl_path = Path(dl_path)

        prefix = f"artifacts/**"
        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
        artifacts = gcloud_ls(bucket_path, prefix, project="dnastack-asap-parkinsons")
        # drop empty strings, files that start with ".", and folders
        artifact_files = [
            f for f in artifacts if f != "" and Path(f).name[0] != "." and f[-1] != "/"
        ]

        if len(artifact_files) > 0:
            bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
            artifact_files_df = pd.DataFrame(artifact_files, columns=["artifact_files"])

            artifact_files_df["file_name"] = artifact_files_df["artifact_files"].apply(
                lambda x: x.split("/")[-1]
            )
            artifact_files_df["bucket_md5"] = artifact_files_df["file_name"].map(
                bucket_files_md5
            )

            artifact_files_df.to_csv(
                os.path.join(dl_path, f"{dataset_name}-artifact_files.csv"), index=False
            )

            # merge in md5s.
            # dump md5s to file
            with open(
                os.path.join(dl_path, f"{dataset_name}-raw_fastqs-md5s.json"), "w"
            ) as f:
                json.dump(bucket_files_md5, f)

        else:
            print(f"No artifact files found for {dataset_name}")

        # create a list of the files in the raw_bucket/fastqs
        if raw_type == "raw":
            prefix = "raw/*.raw" if flatten else "raw/**/*.raw"
        else:  # raw_type == "fastq":
            prefix = "fastqs/*.fastq.gz" if flatten else "fastqs/**/*.fastq.gz"

        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
        fastqs = gcloud_ls(bucket_path, prefix, project="dnastack-asap-parkinsons")

        raw_files = [f for f in fastqs if f != ""]
        print(f"Found {len(raw_files)} raw files for {dataset_name}")
        print(raw_files[:10])
        if len(raw_files) > 0:
            # should just use the list of raw_files?
            bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
            raw_files_df = pd.DataFrame(raw_files, columns=["raw_files"])

            raw_files_df["file_name"] = raw_files_df["raw_files"].apply(
                lambda x: x.split("/")[-1]
            )
            raw_files_df["bucket_md5"] = raw_files_df["file_name"].map(bucket_files_md5)

            # TODO: fix this so the file manifest isn't always fastqs.  There are downstream implications (file name is assumed in release utils)
            raw_files_df.to_csv(
                os.path.join(dl_path, f"{dataset_name}-raw_fastqs.csv"), index=False
            )

            # TODO: merge in md5s.
            # dump md5s to file
            with open(
                os.path.join(dl_path, f"{dataset_name}-raw_fastqs-md5s.json"), "w"
            ) as f:
                json.dump(bucket_files_md5, f)

        else:
            print(f"No spatial files found for {dataset_name}")


def gen_dev_bucket_summary(
    raw_bucket_name: str, dl_path: str | Path, dataset_name: str, flatten: bool = False
):

    dl_path = Path(dl_path)
    ## OTHER and everything else...
    # create a list of the curated files in /artifacts

    prefix = f"artifacts/**"
    bucket_path = (
        f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
    )
    artifacts = gcloud_ls(bucket_path, prefix, project="dnastack-asap-parkinsons")
    # drop empty strings, files that start with ".", and folders
    artifact_files = [
        f for f in artifacts if f != "" and Path(f).name[0] != "." and f[-1] != "/"
    ]

    if len(artifact_files) > 0:
        bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
        artifact_files_df = pd.DataFrame(artifact_files, columns=["artifact_files"])

        artifact_files_df["file_name"] = artifact_files_df["artifact_files"].apply(
            lambda x: x.split("/")[-1]
        )
        artifact_files_df["bucket_md5"] = artifact_files_df["file_name"].map(
            bucket_files_md5
        )

        artifact_files_df.to_csv(
            os.path.join(dl_path, f"{dataset_name}-artifact_files.csv"), index=False
        )

        # merge in md5s.
        # dump md5s to file
        with open(
            os.path.join(dl_path, f"{dataset_name}-raw_fastqs-md5s.json"), "w"
        ) as f:
            json.dump(bucket_files_md5, f)

    else:
        print(f"No artifact files found for {dataset_name}")

    # create a list of the files in the raw_bucket/fastqs
    prefix = "fastqs/*.fastq.gz" if flatten else "fastqs/**/*.fastq.gz"
    bucket_path = (
        f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
    )
    fastqs = gcloud_ls(bucket_path, prefix, project="dnastack-asap-parkinsons")
    raw_files = [f for f in fastqs if f != ""]

    if len(raw_files) > 0:
        # should just use the list of raw_files?
        bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
        raw_files_df = pd.DataFrame(raw_files, columns=["raw_files"])

        raw_files_df["file_name"] = raw_files_df["raw_files"].apply(
            lambda x: x.split("/")[-1]
        )
        raw_files_df["bucket_md5"] = raw_files_df["file_name"].map(bucket_files_md5)
        raw_files_df.to_csv(
            os.path.join(dl_path, f"{dataset_name}-raw_fastqs.csv"), index=False
        )

        # merge in md5s.
        # dump md5s to file
        with open(
            os.path.join(dl_path, f"{dataset_name}-raw_fastqs-md5s.json"), "w"
        ) as f:
            json.dump(bucket_files_md5, f)

    else:
        print(f"No spatial files found for {dataset_name}")


def gen_spatial_bucket_summary(
    raw_bucket_name: str, dl_path: str | Path, dataset_name: str
):
    if "cohort" in dataset_name:
        print(f"No raw bucket for cohort datasets: {dataset_name}")
        # need to join with the cohort-pmdbs-sn-rnaseq raw files

    else:
        dl_path = Path(dl_path)
        # create a list of the files in the raw_bucket/fastqs
        prefix = f"spatial/**/*"
        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
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
def get_artifacts_df(dl_path: str | Path, dataset_id: str, team_id: str):
    """ """
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
        df["ASAP_dataset_id"] = dataset_id
        df["ASAP_team_id"] = team_id
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


def get_fastqs_df(dl_path: str | Path, dataset_id: str, team_id: str) -> pd.DataFrame:
    """ """
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

    dl_path = Path(dl_path)  #
    fastqs = list(dl_path.glob("*-raw_fastqs.csv"))
    if len(fastqs) > 0:

        fastq = fastqs[0]
        print(f"Processing {fastq.name}")
        df = pd.read_csv(fastq)
        df["ASAP_dataset_id"] = dataset_id
        df["ASAP_team_id"] = team_id
        df["timestamp"] = "NA"
        df["workflow"] = "NA"
        df["workflow_version"] = "NA"
        df["artifact_type"] = "contributed"
        df["gcp_uri"] = df["raw_files"]

        df = df[keep_cols]
        return df
    else:
        print(f"no fastq files found for {dl_path.parent.name}")
        return pd.DataFrame(columns=keep_cols)


def get_spatial_df(dl_path: str | Path, dataset_id: str, team_id: str) -> pd.DataFrame:
    """ """
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
        spatial_df["ASAP_dataset_id"] = dataset_id
        spatial_df["ASAP_team_id"] = team_id
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


def add_bucket_md5(df: pd.DataFrame, dl_path: str | Path):
    """ """

    dl_path = Path(dl_path)

    md5_files = list(dl_path.glob(f"*-md5s.json"))
    if len(md5_files) == 0:
        print(f"no md5 files found for {dl_path.parent.name}")
        df["bucket_md5"] = "NA"
        return df

    md5_mapper = {}
    for file in md5_files:
        with open(file, "r") as f:
            md5s = json.load(f)
            md5_mapper.update(md5s)
    df["bucket_md5"] = df["file_name"].map(md5_mapper)
    return df
