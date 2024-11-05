# imports
import pandas as pd
from pathlib import Path
import datetime
import shutil

NULL = "NA"

def read_CDE(metadata_version:str="v3.0", local_path:str|bool|Path=False):
    """
    Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes
    """
    # Construct the path to CSD.csv
    GOOGLE_SHEET_ID = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"

    if metadata_version == "v1":
        sheet_name = "ASAP_CDE_v1"
    elif metadata_version == "v2":
        sheet_name = "ASAP_CDE_v2"
    elif metadata_version == "v2.1":
        sheet_name = "ASAP_CDE_v2.1"
    elif metadata_version == "v3.0-beta":
        sheet_name = "ASAP_CDE_v3.0-beta"
    elif metadata_version in ["v3","v3.0", "v3.0.0"]:
        sheet_name = "ASAP_CDE_v3.0"
    else:
        sheet_name = "ASAP_CDE_v3.0"


    if metadata_version in ["v1","v2","v2.1","v3","v3.0","v3.0-beta"]:
        print(f"metadata_version: {sheet_name}")
    else:
        print(f"Unsupported metadata_version: {sheet_name}")
        return 0,0
    
    cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={metadata_version}"
    print(cde_url)
    if local_path:
        cde_url = Path(local_path) / f"{sheet_name}.csv"
        print(cde_url)

    
    try:
        CDE_df = pd.read_csv(cde_url)
        read_source = "url" if not local_path else "local file"
        print(f"read {read_source}")
    except:
        CDE_df = pd.read_csv(f"{sheet_name}.csv")
        print("read local file")

    # drop rows with no table name (i.e. ASAP_ids)
    CDE_df = CDE_df[["Table", "Field", "Description", "DataType", "Required", "Validation", "Shared_key"]]
    CDE_df = CDE_df.dropna(subset=['Table'])
    CDE_df = CDE_df.reset_index(drop=True)
    CDE_df = CDE_df.drop_duplicates()
    # force extraneous columns to be dropped.

    return CDE_df


def read_CDE_asap_ids( local_path:str|bool|Path=False):
    """
    Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes
    """
    # Construct the path to CSD.csv
    GOOGLE_SHEET_ID = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"

    sheet_name = "ASAP_assigned_keys"

    cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    print(cde_url)
    if local_path:
        # ASAP_assigned_keys only in v3.0
        cde_url = Path(local_path) / f"ASAP_CDE_v3.0_{sheet_name}.csv"
        print(cde_url)
   
    try:
        df = pd.read_csv(cde_url)
        read_source = "url" if not local_path else "local file"
        print(f"read {read_source}")
    except:
        df = pd.read_csv(f"ASAP_CDE_v3.0_{sheet_name}.csv")
        print("read local file")

    # drop rows with no table name (i.e. ASAP_ids)
    df = df[["Table", "Field", "Description", "DataType", "Required", "Validation"]]
    df = df.dropna(subset=['Table'])
    df = df.reset_index(drop=True)

    return df


def compare_CDEs(df1, df2):
    """
    Compares two pandas dataframes and returns if they are identical or not.
    If they are not identical, it returns the differences between the two dataframes.
    """
    # Sort both dataframes by 'Table' and 'Field' before comparison
    df1 = df1.sort_values(by=['Table', 'Field']).reset_index(drop=True)
    df2 = df2.sort_values(by=['Table', 'Field']).reset_index(drop=True)
    
    if df1.equals(df2):
        return "The dataframes are identical."
    
    differences = []
    
    # Check for differences in values
    df1_diff = df1 != df2
    diff_rows, diff_cols = df1_diff.stack()[df1_diff.stack()].index.to_list(), df1_diff.columns[df1_diff.any()]
    
    for row, col in diff_rows:
        differences.append(
            f"Difference at row {row}, column '{col}': "
            f"df1 value = {df1.loc[row, col]}, df2 value = {df2.loc[row, col]}"
        )
    
    return differences if differences else "No differences found in data."


def export_tables_versioned(tables_path:str, out_dir:str, tables:dict):
    """
    """
    # # Prepare output directory
    current_date = datetime.now()

    date_str = current_date.strftime('%Y%m%d')
    export_root = Path(tables_path) / f"{out_dir}_{date_str}"
    
    for name, table in tables.items():
        export_table(name, table, export_root)
        # table.to_csv(export_root / f"{name}.csv", index=False)


def export_table(table_name:str, df:pd.DataFrame, out_dir:str):
    """
    Export a table to a csv file, with nulls/empty entries replaced by "NA"
    """
    # make sure the output directory exists
    export_root = Path(out_dir).parent
    export_root.mkdir(parents=True, exist_ok=True)

    df = df.replace({"":NULL, pd.NA:NULL, "none":NULL, "nan":NULL, "Nan":NULL})
    df.to_csv(out_dir / f"{table_name}.csv", index=False)



def read_meta_table(table_path):
    # read the whole table
    try:
        table_df = pd.read_csv(table_path,dtype=str)
    except UnicodeDecodeError:
        table_df = pd.read_csv(table_path, encoding='latin1',dtype=str)


    for col in table_df.select_dtypes(include='object').columns:
        table_df[col] = table_df[col].str.encode('latin1', errors='replace').str.decode('utf-8', errors='replace')


    # drop the first column if it is just the index incase it was saved with index = True
    if table_df.columns[0] == "Unnamed: 0":
        table_df = table_df.drop(columns=["Unnamed: 0"])

    # drop rows with all null values
    table_df.dropna(how='all', inplace=True)

    table_df.replace({"":NULL, pd.NA:NULL, "none":NULL, "nan":NULL, "Nan":NULL}, inplace=True)

    return table_df.reset_index(drop=True)


######## HELPERS ########
# Define a function to only capitalize the first letter of a string
def capitalize_first_letter(s):
    if not isinstance(s, str) or len(s) == 0:  # Check if the value is a string and non-empty
        return s
    return s[0].upper() + s[1:]

def prep_table(df_in:pd.DataFrame, CDE:pd.DataFrame) -> pd.DataFrame:
    """helper to force capitalization of first letters for string and Enum fields"""
    df = df_in.copy()
    string_enum_fields = CDE[CDE["DataType"].isin(["Enum", "String"])]["Field"].tolist()
    # Convert the specified columns to string data type using astype() without a loop
    columns_to_convert = {col: 'str' for col in string_enum_fields if col in df.columns}
    df = df.astype(columns_to_convert)
    for col in string_enum_fields:
        if col in df.columns and col not in ["sample_id", "source_subject_id", "subject_id", "source_sample_id","assay", "file_type", "file_name", "file_MD5", 'replicate', 'batch']:
            df[col] = df[col].apply(capitalize_first_letter) 
    return df




def create_metadata_package(metadata_source:Path, package_destination:Path):
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
                shutil.copytree(folder, dest, dirs_exist_ok=True)

                print(f"Copied {folder} to {dest}")
                copied.append(folder)
    return copied
    
    

