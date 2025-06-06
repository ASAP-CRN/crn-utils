# imports
import pandas as pd
from pathlib import Path
import datetime
import shutil

NULL = "NA"

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
    "create_metadata_package",
    "get_dataset_version",
    "get_release_version",
    "get_cde_version",
    "write_version",
]

SUPPORTED_METADATA_VERSIONS = [
    "v1",
    "v2",
    "v2.1",
    "v3",
    "v3.0",
    "v3.0-beta",
    "v3.1",
    "v3.2",
    "v3.2-beta",
]


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


def read_CDE(metadata_version: str = "v3.0", local_path: str | bool | Path = False):
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
    if metadata_version == "v1":
        resource_fname = "ASAP_CDE_v1"
    elif metadata_version == "v2":
        resource_fname = "ASAP_CDE_v2"
    elif metadata_version == "v2.1":
        resource_fname = "ASAP_CDE_v2.1"
    elif metadata_version == "v3.0-beta":
        resource_fname = "ASAP_CDE_v3.0-beta"
    elif metadata_version in ["v3", "v3.0", "v3.0.0"]:
        resource_fname = "ASAP_CDE_v3.0"
    elif metadata_version in ["v3.1"]:
        resource_fname = "ASAP_CDE_v3.1"
    elif metadata_version in ["v3.2"]:
        resource_fname = "ASAP_CDE_v3.2"
    else:
        resource_fname = "ASAP_CDE_v3.1"

    # add the Shared_key column for v3
    if metadata_version in ["v3.2", "v3.2-beta", "v3.1", "v3", "v3.0", "v3.0-beta"]:
        column_list += ["Shared_key"]

    if metadata_version in SUPPORTED_METADATA_VERSIONS:
        print(f"metadata_version: {resource_fname}")
    else:
        print(f"Unsupported metadata_version: {resource_fname}")

    cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={metadata_version}"
    print(cde_url)

    if local_path:
        cde_url = Path(local_path) / f"{resource_fname}.csv"
        print(cde_url)

    try:
        GOOGLE_SHEET_ID = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"

        if metadata_version == "v3.1":
            metadata_version = "CDE_final"
        cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={metadata_version}"

        print(f"reading from googledoc {cde_url}")

        CDE_df = pd.read_csv(cde_url)
        print(f"read url")

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
        root = Path(__file__).parent.parent.parent
        CDE_df = pd.read_csv(f"{root}/resource/CDE/{resource_fname}.csv")
        print(f"exception:read local file: ../../resource/CDE/{resource_fname}.csv")

    # drop rows with no table name (i.e. ASAP_ids)
    CDE_df = CDE_df[column_list]
    CDE_df = CDE_df.dropna(subset=["Table"])
    CDE_df = CDE_df.reset_index(drop=True)
    CDE_df = CDE_df.drop_duplicates()
    if metadata_version in ["v3.2", "v3.2-beta", "v3.1", "v3", "v3.0", "v3.0-beta"]:
        CDE_df["Shared_key"] = CDE_df["Shared_key"].fillna(0).astype(int)
    # force extraneous columns to be dropped.

    # CDE_df["Validation"] = CDE_df["Validation"].apply(sanitize_validation_string)

    return clean_cde_schema(CDE_df)


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


def read_CDE_asap_ids(
    schema_version: str = "v3.1", local_path: str | bool | Path = False
):
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
        cde_url = Path(local_path) / f"ASAP_CDE_{schema_version}_{resource_fname}.csv"
        print(f"local_path: {cde_url}")

    try:
        # add version to cde_url
        cde_url = f"{cde_url}_{schema_version}"
        print(f"cde_url: {cde_url}")
        df = pd.read_csv(cde_url)
        read_source = "url" if not local_path else "local file"
        print(f"read {read_source}")
    except:

        root = Path(__file__).parent.parent.parent
        df = pd.read_csv(
            f"{root}/resource/CDE/ASAP_CDE_{schema_version}_{resource_fname}.csv"
        )

        # df = pd.read_csv(f"ASAP_CDE_{schema_version}_{resource_fname}.csv")
        print("read local file")

    # drop rows with no table name (i.e. ASAP_ids)
    df = df[["Table", "Field", "Description", "DataType", "Required", "Validation"]]
    df = df.dropna(subset=["Table"])
    df = df.reset_index(drop=True)

    return df


def compare_CDEs(df1, df2):
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
    # # Prepare output directory
    current_date = datetime.now()

    date_str = current_date.strftime("%Y%m%d")
    export_root = Path(tables_path) / f"{out_dir}_{date_str}"

    for name, table in tables.items():
        export_table(name, table, export_root)
        # table.to_csv(export_root / f"{name}.csv", index=False)


def export_table(table_name: str, df: pd.DataFrame, out_dir: str):
    """
    Export a table to a csv file, with nulls/empty entries replaced by "NA"
    """
    # make sure the output directory exists
    export_root = Path(out_dir).parent
    export_root.mkdir(parents=True, exist_ok=True)

    df = df.replace({"": NULL, pd.NA: NULL, "none": NULL, "nan": NULL, "Nan": NULL})
    df.to_csv(out_dir / f"{table_name}.csv", index=False)


def read_meta_table(table_path):
    # read the whole table
    try:
        table_df = pd.read_csv(table_path, dtype=str)
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
    table_df.fillna(NULL, inplace=True)
    table_df.replace(
        {"": NULL, pd.NA: NULL, "none": NULL, "nan": NULL, "Nan": NULL}, inplace=True
    )

    return table_df.reset_index(drop=True)


######## HELPERS ########
# Define a function to only capitalize the first letter of a string
def capitalize_first_letter(s):
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


def load_tables(table_path, tables):
    dfs = {}
    for tab in tables:
        # print(f"loading {tab}")
        dfs[tab] = read_meta_table(table_path / f"{tab}.csv")
    return dfs


def export_meta_tables(dfs, export_path):
    for tab in dfs.keys():
        if tab not in dfs:  # BUG:?  can this ever be true
            print(f"Table {tab} not found in dataset tables")
            continue
        dfs[tab].to_csv(export_path / f"{tab}.csv")
    return 0


def create_metadata_package(metadata_source: Path, package_destination: Path):
    """
    Move the metadata folders in the metadata_source to the package_destination

    Do it folder by folder to avoid copying empty folders
    use Path tools to copy since these are local files.  We will upload with gsutil later

    return list of folders copied
    """

    package_destination.mkdir(exist_ok=True)
    # make metadata subdir
    package_destination = package_destination / "metadata"
    package_destination.mkdir(exist_ok=True)

    copied = []
    for folder in metadata_source.iterdir():
        # check that the folder is not empty
        if folder.is_dir():
            # check that the folder is not empty
            if not list(folder.iterdir()):
                print(f"Skipping empty folder {folder}")
                continue
            else:
                dest = package_destination / folder.name
                # dest.mkdir(exist_ok=True)
                shutil.copy2tree(folder, dest, dirs_exist_ok=True)

                print(f"Copied {folder} to {dest}")
                copied.append(folder)
    return copied


def get_dataset_version(dataset_name: str, datasets_path: Path):
    """
    Get the version of the dataset from the dataset name
    """
    dataset_path = datasets_path / dataset_name
    with open(dataset_path / "version", "r") as f:
        ds_ver = f.read().strip()
    # ds_ver = "v2.0"

    return ds_ver


def get_release_version(release_path: Path):
    """
    Get the version of the release from the release_path
    """

    with open(release_path / "version", "r") as f:
        release_ver = f.read().strip()

    return release_ver


def get_cde_version(cde_path: Path):
    """
    Get the version of the CDE from the cde_path
    """
    with open(cde_path / "cde_version", "r") as f:
        cde_ver = f.read().strip()
    return cde_ver


def write_version(version: str, version_path: Path):
    """
    Write the version to the version_path
    """
    # check if the version has a trailing .?
    if len(version.split(".")) > 2:
        version = f"{version}.0"
    else:
        version = f"{version}.0"
    with open(version_path, "w") as f:
        f.write(version)
