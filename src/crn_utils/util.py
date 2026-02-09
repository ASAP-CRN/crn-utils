import os, sys
import pandas as pd
from pathlib import Path
import datetime
import shutil
import numpy as np

NULL = "NA"

crn_utils_root = str(os.path.join(Path(__file__).resolve().parents[2], "crn-utils"))

__all__ = [
    "read_CDE",
    "read_CDE_asap_ids",
    "compare_CDEs",
    "export_tables_versioned",
    "export_table",
    "read_meta_table",
    "capitalize_first_letter",
    "prep_table",
    "load_tables",
    "export_meta_tables",
    "get_dataset_version",
    "get_release_version",
    "get_cde_version",
    "write_version",
    "archive_CDE",
    "list_expected_metadata_tables",
]

SUPPORTED_CDE_VERSIONS = [
    "v1",
    "v2",
    "v2.1",
    "v3",
    "v3.0",
    "v3.0-beta",
    "v3.1",
    "v3.2",
    "v3.2-beta",
    "v3.3",
]



# TODO: This will be deprecated in favor of call to list tables by source/species/assay
def list_expected_metadata_tables() -> list[str]:
    """
    This returns a list of all CRN metadata tables
    """
    tables = [
        "STUDY",
        "PROTOCOL",
        "SUBJECT",
        "SAMPLE",
        "ASSAY",
        "ASSAY_RNAseq",
        "DATA",
        "PMDBS",
        "CLINPATH",
        "CONDITION",
        "MOUSE",
        "SPATIAL",
        "CELL",
        "PROTEOMICS",
        "SDRF"
    ]
        
    return tables



def sanitize_validation_string(validation_str):
    """Sanitize validation strings by replacing smart quotes with straight quotes."""
    if not isinstance(validation_str, str):
        return validation_str
    return (
        validation_str.replace('"', '"')
        .replace('"', '"')
        .replace(""", "'").replace(""", "'")
        .replace("…", "...")
    )

def read_CDE(
    cde_version: str = "v3.2",
    local_path: str | bool | Path = False,
    include_asap_ids: bool = False,
    include_aliases: bool = False,
):
    """
    Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes
    """
    # Construct the path to CSD.csv
    GOOGLE_SHEET_ID = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"

    column_list = [
        "Table",
        "Field",
        "Description",
        "DataType",
        "Required",
        "Validation",
    ]

    # set up fallback
    if cde_version == "v1":
        resource_fname = "ASAP_CDE_v1"
    elif cde_version == "v2":
        resource_fname = "ASAP_CDE_v2"
    elif cde_version == "v2.1":
        resource_fname = "ASAP_CDE_v2.1"
    elif cde_version == "v3.0-beta":
        resource_fname = "ASAP_CDE_v3.0-beta"
    elif cde_version in ["v3", "v3.0", "v3.0.0"]:
        resource_fname = "ASAP_CDE_v3.0"
    elif cde_version in ["v3.1"]:
        resource_fname = "ASAP_CDE_v3.1"
    elif cde_version in ["v3.2", "v3.2-beta"]:
        resource_fname = "ASAP_CDE_v3.2"
    elif cde_version == "v3.3":
        resource_fname = "ASAP_CDE_v3.3"
    else:
        sys.exit(f"Unsupported cde_version: {cde_version}")

    # add the Shared_key column for v3
    if cde_version in [
        "v3.3",
        "v3.2",
        "v3.2-beta",
        "v3.1",
        "v3",
        "v3.0",
        "v3.0-beta",
    ]:
        column_list.append("Shared_key")

    # insert "DisplayName" after "Field"
    if cde_version in ["v3.2", "v3.2-beta", "v3.3"]:
        column_list.insert(2, "DisplayName")

    if cde_version in SUPPORTED_CDE_VERSIONS:
        print(f"cde_version: {resource_fname}")
    else:
        print(f"Unsupported cde_version: {resource_fname}")

    cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={cde_version}"
    print(cde_url)

    if local_path:
        cde_url = os.path.join(local_path, f"{resource_fname}.csv")
        print(f"reading from local file: {cde_url}")
    else:
        print(f"reading from googledoc {cde_url}")

    try:
        # if cde_version == "v3.2":
        #     cde_version = "CDE_final"
        # cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={cde_version}"

        CDE_df = pd.read_csv(cde_url)
        print(f"read CDE")

        # read_source = "url" if not local_path else "local file"
    except:
        # import requests

        # # download the cde_url to a temp file
        # print(f"downloading {cde_url} to {resource_fname}.csv")
        # response = requests.get(cde_url)
        # response.raise_for_status()  # Check if the request was successful
        # # save to ../../resource/CDE/
        # new_resource_fname = f"../../resource/CDE/_{resource_fname}.csv"
        # with open(new_resource_fname, "wb") as file:
        #     file.write(response.content)

        # CDE_df = pd.read_csv(new_resource_fname)
        # print(f"exception:read local file: {new_resource_fname}")
        CDE_df = pd.read_csv(os.path.join(crn_utils_root, "/resource/CDE/", f"{resource_fname}.csv"))
        print(f"exception:read fallback file: ../../resource/CDE/{resource_fname}.csv")

    # drop ASAP_ids if not requested
    if not include_asap_ids:
        CDE_df = CDE_df[CDE_df["Required"] != "Assigned"]
        CDE_df = CDE_df.reset_index(drop=True)
        print(f"dropped ASAP_ids")

    # drop Alias if not requested
    if not include_aliases:
        CDE_df = CDE_df[CDE_df["Required"] != "Alias"]
        CDE_df = CDE_df.reset_index(drop=True)
        print(f"dropped Alias")

    # drop rows with no table name (i.e. ASAP_ids)
    CDE_df = CDE_df.loc[:, column_list]
    CDE_df = CDE_df.dropna(subset=["Table"])
    CDE_df = CDE_df.reset_index(drop=True)
    CDE_df = CDE_df.drop_duplicates()

    # force Shared_key to be int
    if cde_version in [
        "v3.3",
        "v3.2",
        "v3.2-beta",
        "v3.1",
        "v3",
        "v3.0",
        "v3.0-beta",
    ]:
        CDE_df["Shared_key"] = CDE_df["Shared_key"].fillna(0).astype(int)
    # force extraneous columns to be dropped.
    # CDE_df["Validation"] = CDE_df["Validation"].apply(sanitize_validation_string)

    return clean_cde_schema(CDE_df)

def archive_CDE(
    cde_version, resource_path: str | Path, CDE_df: pd.DataFrame | None = None
):
    """
    Archive CDE data to a CSV file
    """
    if CDE_df is None:
        CDE_df = read_CDE(
            cde_version=cde_version,
            include_asap_ids=True,
            include_aliases=True,
        )

    resource_path = Path(resource_path)
    if not resource_path.exists():
        resource_path = os.path.join(crn_utils_root, "resource/CDE")
        print(f"exporting to default: {resource_path}")

    export_path = os.path.join(resource_path, f"ASAP_CDE_{cde_version}.csv")

    CDE_df.to_csv(export_path, index=False)
    print(f"wrote CDE to: {export_path}")

def sanitize_string(s):
    """Replace smart quotes with straight quotes and other problematic characters."""
    if not isinstance(s, str):
        return s
    return (
        s.replace('"', '"')
        .replace('"', '"')
        .replace(
            """, "'")
             .replace(""",
            "'",
        )
        .replace("…", "...")
    )

def clean_cde_schema(cde_schema):
    """
    Clean the CDE schema by sanitizing validation strings.

    Args:
        cde_schema (pd.DataFrame): The CDE schema dataframe

    Returns:
        pd.DataFrame: A cleaned copy of the CDE schema
    """
    # Make a copy to avoid modifying the original
    cleaned_schema = cde_schema.copy()

    # Sanitize the Validation column
    if "Validation" in cleaned_schema.columns:
        cleaned_schema["Validation"] = cleaned_schema["Validation"].apply(
            sanitize_string
        )

    # Also sanitize any other columns that might contain validation expressions
    for col in ["Description", "Notes", "Example"]:
        if col in cleaned_schema.columns:
            cleaned_schema[col] = cleaned_schema[col].apply(sanitize_string)

    return cleaned_schema


# new function
def read_CDE_asap_ids(
    schema_version: str = "v3.3", local_path: str | bool | Path = False
) -> pd.DataFrame:
    """
    Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes
    """

    df = read_CDE(schema_version, local_path, include_asap_ids=True)

    df = df[df["Required"] == "Assigned"]
    # drop rows with no table name (i.e. ASAP_ids)
    df = df[["Table", "Field", "Description", "DataType", "Required", "Validation"]]
    df = df.dropna(subset=["Table"])
    df = df.reset_index(drop=True)

    return df

# original function
def _read_CDE_asap_ids(
    schema_version: str = "v3.3", local_path: str | bool | Path = False
) -> pd.DataFrame:
    """
    Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes
    """
    # Construct the path to CSD.csv
    GOOGLE_SHEET_ID = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"

    resource_fname = "ASAP_assigned_keys"

    # strip the .0 from the schema version if it exists
    schema_version = schema_version.rstrip(".0")

    cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={resource_fname}"
    print(cde_url)
    if local_path:
        # ASAP_assigned_keys only in >v3.0
        cde_url = os.path.join(local_path, f"ASAP_CDE_{schema_version}_{resource_fname}.csv")
        print(f"local_path: {cde_url}")

    try:
        # add version to cde_url
        cde_url = f"{cde_url}_{schema_version}"
        print(f"cde_url: {cde_url}")
        df = pd.read_csv(cde_url)
        read_source = "url" if not local_path else "local file"
        print(f"read {read_source}")
    except:
        df = pd.read_csv(os.path.join(f"{crn_utils_root}/resource/CDE/ASAP_CDE_{schema_version}_{resource_fname}.csv"))
        print("read local file")

    # drop rows with no table name (i.e. ASAP_ids)
    df = df[["Table", "Field", "Description", "DataType", "Required", "Validation"]]
    df = df.dropna(subset=["Table"])
    df = df.reset_index(drop=True)

    return df


def compare_CDEs(df1: pd.DataFrame, df2: pd.DataFrame) -> str | list[str]:
    """
    Compares two pandas dataframes and returns if they are identical or not.
    If they are not identical, it returns the differences between the two dataframes.
    """
    # Sort both dataframes by 'Table' and 'Field' before comparison
    df1 = df1.sort_values(by=["Table", "Field"]).reset_index(drop=True)
    df2 = df2.sort_values(by=["Table", "Field"]).reset_index(drop=True)

    if df1.equals(df2):
        return "The dataframes are identical."

    differences = []

    # Check for differences in values
    df1_diff = df1 != df2
    diff_rows, diff_cols = (
        df1_diff.stack()[df1_diff.stack()].index.to_list(),
        df1_diff.columns[df1_diff.any()],
    )

    for row, col in diff_rows:
        differences.append(
            f"Difference at row {row}, column '{col}': "
            f"df1 value = {df1.loc[row, col]}, df2 value = {df2.loc[row, col]}"
        )

    return differences if differences else "No differences found in data."

def export_tables_versioned(tables_path: str, out_dir: str, tables: dict):
    """ """
    # Prepare output directory
    current_date = datetime.now()

    date_str = current_date.strftime("%Y%m%d")
    export_root = os.path.join(tables_path, f"{out_dir}_{date_str}")

    for name, table in tables.items():
        export_table(name, table, export_root)

def export_table(table_name: str, df: pd.DataFrame, out_dir: str):
    """
    Export a table to a csv file, with nulls/empty entries replaced by "NA"
    """
    # make sure the output directory exists
    export_root = Path(out_dir).parent
    os.makedirs(export_root, exist_ok=True)

    df = df.astype(str).replace({
        "": NULL,
        "<NA>": NULL,
        pd.NA: NULL,
        "none": NULL,
        "nan": NULL,
        "Nan": NULL
    })
    df.to_csv(os.path.join(out_dir, f"{table_name}.csv"), index=False)

def read_meta_table(table_path: str | Path) -> pd.DataFrame:
    # read the whole table
    try:
        table_df = pd.read_csv(table_path, encoding="utf-8", dtype=str)
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError: {table_path}")
        table_df = pd.read_csv(table_path, encoding="latin1", dtype=str)

    for col in table_df.select_dtypes(include="object").columns:
        table_df[col] = (
            table_df[col]
            .str.encode("latin1", errors="replace")
            .str.decode("utf-8", errors="replace")
        )

    for col in table_df.columns:
        table_df[col] = table_df[col].apply(sanitize_validation_string)

    # drop the first column if it is just the index incase it was saved with index = True
    if table_df.columns[0] == "Unnamed: 0":
        table_df = table_df.drop(columns=["Unnamed: 0"])

    # drop rows with all null values
    table_df.dropna(how="all", inplace=True)
    table_df.fillna(np.nan, inplace=True)
    table_df = table_df.astype(str)
    table_df.replace({
        "": NULL,
        "<NA>": NULL,
        pd.NA: NULL,
        "none": NULL,
        "nan": NULL,
        "Nan": NULL
    }, inplace=True)
    return table_df.reset_index(drop=True)

######## HELPERS ########
# Define a function to only capitalize the first letter of a string
def capitalize_first_letter(s) -> str:
    if (
        not isinstance(s, str) or len(s) == 0
    ):  # Check if the value is a string and non-empty
        return s
    return s[0].upper() + s[1:]


def prep_table(df_in: pd.DataFrame, CDE: pd.DataFrame) -> pd.DataFrame:
    """helper to force capitalization of first letters for string and Enum fields"""
    df = df_in.copy()
    string_enum_fields = CDE[CDE["DataType"].isin(["Enum", "String"])]["Field"].tolist()
    # Convert the specified columns to string data type using astype() without a loop
    columns_to_convert = {col: "str" for col in string_enum_fields if col in df.columns}
    df = df.astype(columns_to_convert)
    for col in string_enum_fields:
        if col in df.columns and col not in [
            "sample_id",
            "source_subject_id",
            "subject_id",
            "source_sample_id",
            "assay",
            "file_type",
            "file_name",
            "file_MD5",
            "replicate",
            "batch",
            "path_brak",
        ]:
            df[col] = df[col].apply(capitalize_first_letter)
    return df


def load_tables(table_dir: Path, table_names: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load the specified metadata tables from table_path into a dictionary of DataFrames.
    """
    table_dir = Path(table_dir)
    tables = {}
    for tab in table_names:
        table_path = table_dir / f"{tab}.csv"
        if not table_path.exists():
            raise FileNotFoundError(f"Table file not found: {table_path}")
        # TODO: read_meta_table() was propagating latin1 encoding errors
        # tables[tab] = read_meta_table(table_path)
        tables[tab] = pd.read_csv(table_path, encoding="utf-8")
    
    return tables


def export_meta_tables(dfs: dict[str, pd.DataFrame], export_path: Path) -> None:
    """
    Save the dictionary of metadata tables (dfs) to CSV files in export_path.
    """
    export_path = Path(export_path)
    if not export_path.exists():
        raise ValueError(f"export_path {export_path} does not exist")
    
    for tab in dfs.keys():
        table_path = export_path / f"{tab}.csv"
        dfs[tab].to_csv(table_path, index=False)


# depricate (there is a create_metadata_package in release_util.py) and this is no longer used
# TODO: remove
def _create_metadata_package(metadata_source: Path, package_destination: Path):
    """
    Move the metadata folders in the metadata_source to the package_destination

    Do it folder by folder to avoid copying empty folders
    use Path tools to copy since these are local files.  We will upload with gsutil later

    return list of folders copied
    """

    os.makedirs(package_destination, exist_ok=True)
    # make metadata subdir
    package_destination = os.path.join(package_destination, "metadata")
    os.makedirs(package_destination, exist_ok=True)

    copied = []
    for folder in metadata_source.iterdir():
        # check that the folder is not empty
        if os.path.isdir(folder):
            # check that the folder is not empty
            if not list(folder.iterdir()):
                print(f"Skipping empty folder {folder}")
                continue
            else:
                dest = os.path.join(package_destination, folder.name)
                # os.makedirs(dest, exist_ok=True)
                shutil.copytree(folder, dest, dirs_exist_ok=True)

                print(f"Copied {folder} to {dest}")
                copied.append(folder)
    return copied


def get_dataset_version(dataset_name: str, datasets_path: Path) -> str:
    """
    Get the version of the dataset from the dataset name
    """
    dataset_path = os.path.join(datasets_path, dataset_name)
    with open(os.path.join(dataset_path, "version"), "r") as f:
        ds_ver = f.read().strip()
    # ds_ver = "v2.0"

    return ds_ver


def get_release_version(release_path: Path) -> str:
    """
    Get the version of the release from the release_path
    """

    with open(os.path.join(release_path, "version"), "r") as f:
        release_ver = f.read().strip()

    return release_ver


def get_cde_version(cde_path: Path):
    """
    Get the version of the CDE from the cde_path
    """
    with open(os.path.join(cde_path, "cde_version"), "r") as f:
        cde_ver = f.read().strip()
    return cde_ver


def write_version(version: str, version_path: Path):
    """
    Write the version to the version_path
    """
    # check if the version has a trailing .?
    if len(version.split(".")) > 2:
        version = f"{version}"
    else:
        version = f"{version}.0"
    with open(version_path, "w") as f:
        f.write(version)
