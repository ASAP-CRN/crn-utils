
"""
Utilities to check if a Google Spreadsheet and its tabs are available and read them into Pandas DataFrames.

Authors: Javier Diaz
"""

import requests
import pandas as pd
from urllib.parse import quote
from io import StringIO

def read_google_sheet(spreadsheet_id: str, tab_name: str) -> pd.DataFrame:
    """
    Checks if Google Spreadsheet and tab are available, then reads the tab into a Pandas DataFrame.
    
    Parameters
    ----------
    google_datasets_sheet_id : str
        Google Sheets ID for the datasets document
    tab : str
        Tab name (i.e release version string (e.g., "v4.0.0")
        
    Returns
    -------
    pd.DataFrame
        datasets_df dataframe

    Raises
    ------
    SystemExit
        If unable to read the Google Sheets document
        
    """
    check_spreadsheet_available(spreadsheet_id)
    check_tab_exists(spreadsheet_id, tab_name)
    return read_tab_as_pd(spreadsheet_id, tab_name)


def check_spreadsheet_available(spreadsheet_id: str, timeout: int = 10) -> None:
    spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    print( f"Checking spreadsheet URL" )
    response = requests.get(spreadsheet_url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Spreadsheet not accessible (HTTP {response.status_code})"
        )

def check_tab_exists(spreadsheet_id: str, tab_name: str, timeout: int = 10) -> None:
    print( f"Checking tab available" )
    encoded_tab_name = quote(tab_name)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={encoded_tab_name}"
    )
    response = requests.get(csv_url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Tab '{tab_name}' not accessible (HTTP {response.status_code})"
        )
    content_type = response.headers.get("Content-Type", "")
    if "text/csv" not in content_type:
        raise RuntimeError(
            f"Tab '{tab_name}' does not exist or is not exportable as CSV"
        )

def read_tab_as_pd(spreadsheet_id: str, tab_name: str, timeout: int = 10) -> pd.DataFrame:
    print( f"Reading {tab_name} into DataFrame" )
    encoded_tab_name = quote(tab_name)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={encoded_tab_name}"
    )
    response = requests.get(csv_url, timeout=timeout)
    response.raise_for_status()
    csv_data = StringIO(response.text)
    dataframe = pd.read_csv(csv_data)
    return dataframe

