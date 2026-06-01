
"""
Utilities to check if a Google Spreadsheet and its tabs are available and read them into Pandas DataFrames.

Authors: Javier Diaz
"""

import time
import requests
import pandas as pd
from urllib.parse import quote
from io import StringIO

_DEFAULT_TIMEOUT      = 30
_DEFAULT_MAX_RETRIES  = 3
_DEFAULT_BACKOFF      = 2.0   # seconds; doubles each attempt: 2, 4, 8


def _get_with_retry(
    url: str,
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff_factor: float = _DEFAULT_BACKOFF,
) -> requests.Response:
    """
    GET url with exponential-backoff retries on ReadTimeout.

    Parameters
    ----------
    url : str
        URL to fetch.
    timeout : int
        Per-attempt read timeout in seconds.
    max_retries : int
        Total number of attempts (1 = no retry).
    backoff_factor : float
        Base wait time in seconds; wait on attempt n is `backoff_factor ** (n-1)`.

    Returns
    -------
    requests.Response
        The HTTP response from the first successful attempt.

    Raises
    ------
    requests.exceptions.ReadTimeout
        If all attempts time out.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0:
            wait = backoff_factor ** (attempt - 1)
            print(f"  Retry {attempt}/{max_retries - 1} after {wait:.0f}s (previous attempt timed out)")
            time.sleep(wait)
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.ReadTimeout as exc:
            last_exc = exc
    raise last_exc


def read_google_sheet(spreadsheet_id: str, tab_name: str) -> pd.DataFrame:
    """
    Checks if Google Spreadsheet and tab are available, then reads the tab into a Pandas DataFrame.

    Parameters
    ----------
    spreadsheet_id : str
        Google Sheets ID for the datasets document
    tab_name : str
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


def check_spreadsheet_available(spreadsheet_id: str, timeout: int = _DEFAULT_TIMEOUT) -> None:
    spreadsheet_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    print( f"Checking spreadsheet URL" )
    response = _get_with_retry(spreadsheet_url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Spreadsheet not accessible (HTTP {response.status_code})"
        )

def check_tab_exists(spreadsheet_id: str, tab_name: str, timeout: int = _DEFAULT_TIMEOUT) -> None:
    print( f"Checking tab available" )
    encoded_tab_name = quote(tab_name)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={encoded_tab_name}"
    )
    response = _get_with_retry(csv_url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"Tab '{tab_name}' not accessible (HTTP {response.status_code})"
        )
    content_type = response.headers.get("Content-Type", "")
    if "text/csv" not in content_type:
        raise RuntimeError(
            f"Tab '{tab_name}' does not exist or is not exportable as CSV"
        )

def read_tab_as_pd(spreadsheet_id: str, tab_name: str, timeout: int = _DEFAULT_TIMEOUT) -> pd.DataFrame:
    print( f"Reading {tab_name} into DataFrame" )
    encoded_tab_name = quote(tab_name)
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/gviz/tq"
        f"?tqx=out:csv&sheet={encoded_tab_name}"
    )
    response = _get_with_retry(csv_url, timeout=timeout)
    response.raise_for_status()
    csv_data = StringIO(response.text)
    dataframe = pd.read_csv(csv_data)
    return dataframe

