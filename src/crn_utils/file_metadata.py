# %%
# #### create the key dataset manifest tables for each dataset
# re datasets for a release
#  dataset_name, bucket_name, collection_name
#
#


# %%
import pandas as pd
from pathlib import Path
import os, sys
import json
import shutil

from .bucket_util import (
    authenticate_with_service_account,
    gsutil_ls,
    gsutil_cp,
    gsutil_ls2,
    gsutil_cp2,
)

from .util import read_meta_table
from .checksums import extract_md5_from_details2, get_md5_hashes

__all__ = [
    "make_manifest_tables",
    "get_bulk_manifests",
    "get_sc_manifests",
    "make_cohort_fastqtables",
    "gen_raw_bucket_summary",
    "gen_spatial_bucket_summary",
    "make_file_metadata",
    "get_fastqs_df",
    "get_artifacts_df",
    "get_spatial_df",
]

# define collections, collection names and datasets

BULK_MODES = ["alignment_mode", "mapping_mode"]


def make_file_metadata(
    ds_path: Path,
    dl_path: Path,
    data_df: pd.DataFrame,
    spatial: bool = False,
):

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
        artifacts_df.to_csv(dl_path / "artifacts.csv", index=False)
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
        spatial_df.to_csv(dl_path / "spatial_files.csv", index=False)
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
    df.to_csv(dl_path / "raw_files.csv", index=False)


def update_data_table_with_gcp_uri(data_df: pd.DataFrame, ds_path: str | Path):
    """ """
    ds_path = Path(ds_path)
    file_metadata_path = ds_path / "file_metadata"

    raw_files = pd.read_csv(file_metadata_path / "raw_files.csv")

    raw_files = raw_files[["file_name", "gcp_uri"]]
    data_df = data_df.merge(raw_files, on="file_name", how="left")

    print(f"Updated 'DATA.csv' with gcp_uri")

    return data_df


def update_spatial_table_with_gcp_uri(
    spatial_df: pd.DataFrame, ds_path: str | Path, visium: bool = True
):
    """ """
    ds_path = Path(ds_path)
    file_metadata_path = ds_path / "file_metadata"

    raw_files = pd.read_csv(file_metadata_path / "raw_files.csv")
    spatial_files = pd.read_csv(
        file_metadata_path / f"{ds_path.name}-spatial_files.csv"
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
    raw_bucket_name: str, dl_path: Path, dataset_name: str, flatten: bool = False
):
    if "cohort" in dataset_name:
        print(f"No raw bucket for cohort datasets: {dataset_name}")
        # need to join with the cohort-pmdbs-sn-rnaseq raw files

    else:

        ## OTHER and everything else...
        # create a list of the curated files in /artifacts

        prefix = f"artifacts/**"
        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
        artifacts = gsutil_ls2(bucket_path, prefix, project="dnastack-asap-parkinsons")
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
                dl_path / f"{dataset_name}-artifact_files.csv", index=False
            )

            # merge in md5s.
            # dump md5s to file
            with open(dl_path / f"{dataset_name}-raw_fastqs-md5s.json", "w") as f:
                json.dump(bucket_files_md5, f)

        else:
            print(f"No artifact files found for {dataset_name}")

        # create a list of the files in the raw_bucket/fastqs
        prefix = "fastqs/*.fastq.gz" if flatten else "fastqs/**/*.fastq.gz"
        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
        fastqs = gsutil_ls2(bucket_path, prefix, project="dnastack-asap-parkinsons")
        raw_files = [f for f in fastqs if f != ""]

        if len(raw_files) > 0:
            # should just use the list of raw_files?
            bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
            raw_files_df = pd.DataFrame(raw_files, columns=["raw_files"])

            raw_files_df["file_name"] = raw_files_df["raw_files"].apply(
                lambda x: x.split("/")[-1]
            )
            raw_files_df["bucket_md5"] = raw_files_df["file_name"].map(bucket_files_md5)
            raw_files_df.to_csv(dl_path / f"{dataset_name}-raw_fastqs.csv", index=False)

            # merge in md5s.
            # dump md5s to file
            with open(dl_path / f"{dataset_name}-raw_fastqs-md5s.json", "w") as f:
                json.dump(bucket_files_md5, f)

        else:
            print(f"No spatial files found for {dataset_name}")


def gen_dev_bucket_summary(
    raw_bucket_name: str, dl_path: Path, dataset_name: str, flatten: bool = False
):

    ## OTHER and everything else...
    # create a list of the curated files in /artifacts

    prefix = f"artifacts/**"
    bucket_path = (
        f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
    )
    artifacts = gsutil_ls2(bucket_path, prefix, project="dnastack-asap-parkinsons")
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
            dl_path / f"{dataset_name}-artifact_files.csv", index=False
        )

        # merge in md5s.
        # dump md5s to file
        with open(dl_path / f"{dataset_name}-raw_fastqs-md5s.json", "w") as f:
            json.dump(bucket_files_md5, f)

    else:
        print(f"No artifact files found for {dataset_name}")

    # create a list of the files in the raw_bucket/fastqs
    prefix = "fastqs/*.fastq.gz" if flatten else "fastqs/**/*.fastq.gz"
    bucket_path = (
        f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
    )
    fastqs = gsutil_ls2(bucket_path, prefix, project="dnastack-asap-parkinsons")
    raw_files = [f for f in fastqs if f != ""]

    if len(raw_files) > 0:
        # should just use the list of raw_files?
        bucket_files_md5 = get_md5_hashes(bucket_path, prefix)
        raw_files_df = pd.DataFrame(raw_files, columns=["raw_files"])

        raw_files_df["file_name"] = raw_files_df["raw_files"].apply(
            lambda x: x.split("/")[-1]
        )
        raw_files_df["bucket_md5"] = raw_files_df["file_name"].map(bucket_files_md5)
        raw_files_df.to_csv(dl_path / f"{dataset_name}-raw_fastqs.csv", index=False)

        # merge in md5s.
        # dump md5s to file
        with open(dl_path / f"{dataset_name}-raw_fastqs-md5s.json", "w") as f:
            json.dump(bucket_files_md5, f)

    else:
        print(f"No spatial files found for {dataset_name}")


def gen_spatial_bucket_summary(raw_bucket_name: str, dl_path: Path, dataset_name: str):
    if "cohort" in dataset_name:
        print(f"No raw bucket for cohort datasets: {dataset_name}")
        # need to join with the cohort-pmdbs-sn-rnaseq raw files

    else:

        # create a list of the files in the raw_bucket/fastqs
        prefix = f"spatial/**/*"
        bucket_path = (
            f"{raw_bucket_name.split('/')[-1]}"  # dev_bucket_name has gs:// prefix
        )
        s_files = gsutil_ls2(bucket_path, prefix, project="dnastack-asap-parkinsons")
        spatial_files = [f for f in s_files if f != ""]

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
                dl_path / f"{dataset_name}-spatial_files.csv", index=False
            )
            # merge in md5s.
            # dump md5s to file
            with open(dl_path / f"{dataset_name}-spatial_files-md5s.json", "w") as f:
                json.dump(bucket_files_md5, f)
        else:
            print(f"No spatial files found for {dataset_name}")


####################


def get_artifacts_df(dl_path: Path, dataset_id: str, team_id: str):
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


def get_fastqs_df(dl_path: Path, dataset_id: str, team_id: str) -> pd.DataFrame:
    """ """

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


def get_spatial_df(dl_path: Path, dataset_id: str, team_id: str) -> pd.DataFrame:

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


def add_bucket_md5(df: pd.DataFrame, dl_path: Path):
    """ """
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


####################
## raw bucket contributions
####################

####################
## helpers
####################


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
## API
####################


if __name__ == "__main__":
    # loads .jsons from .json path input
    # datasets:dict,
    # release_version:str,
    # collection_version:dict,
    # root_path:Path|str|None = None):
    import argparse

    parser = argparse.ArgumentParser(description="prep relase 0: dataset refs")

    parser.add_argument(
        "--release",
        dest="release_version",
        type=str,
        default=LATEST_RELEASE,
        help="release version",
    )

    parser.add_argument(
        "--root",
        dest="root_path",
        type=str,
        default="None",
        help="root path of release artifacts",
    )

    args = parser.parse_args()

    make_manifest_tables(args.release_version, args.root_path)
