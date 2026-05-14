"""
Encoding detection and repair utilities for CRN metadata tables.

Handles Windows-1252 and Mac OS Roman byte artefacts that arise when CSV files
authored in Excel are read as UTF-8 or Latin-1.  Also provides free-text column
sanitization used by dataset QC hooks.
"""

import unicodedata
import pandas as pd

__all__ = [
    "audit_encoding_columns",
    "remove_special_characters_ascii_printable",
]

# Bytes 0x80–0x9F in Windows-1252 carry typographic characters that are undefined
# in ISO-8859-1/latin-1. pandas read_csv (UTF-8 default) or a latin1→utf-8
# re-encode pass will leave these as raw byte escapes rather than their intended
# Unicode equivalents.
_WIN1252_ARTEFACTS: dict[str, str] = {
    "\x91": "'",  # '  left single quotation mark
    "\x92": "'",  # '  right single quotation mark / apostrophe
    "\x93": """,  # "  left double quotation mark
    "\x94": """,  # "  right double quotation mark
    "\x96": "–",  # –  en dash
    "\x97": "—",  # —  em dash
}

# Mac OS Roman uses a different byte range (0xD0–0xD5) for the same typographic
# characters. Rare in practice since OS X and modern Mac Excel default to UTF-8,
# but included for completeness.
_MACROMAN_ARTEFACTS: dict[str, str] = {
    "\xd4": "'",  # '  left single quotation mark
    "\xd5": "'",  # '  right single quotation mark / apostrophe
    "\xd2": """,  # "  left double quotation mark
    "\xd3": """,  # "  right double quotation mark
    "\xd0": "–",  # –  en dash
    "\xd1": "—",  # —  em dash
}

_ALL_ENCODING_ARTEFACTS: dict[str, str] = {**_WIN1252_ARTEFACTS, **_MACROMAN_ARTEFACTS}


def _char_repr(ch: str) -> str:
    """Return a printable escape-sequence label for a single character."""
    cp = ord(ch)
    if cp <= 0xFF:
        return f"\\x{cp:02x}"
    return f"\\u{cp:04x}"


def _sanitize_series(series: pd.Series) -> pd.Series:
    """
    Replace Win1252/MacRoman artefact bytes and strip U+FFFD in a single Series.

    Parameters
    ----------
    series : pd.Series
        String series to clean.

    Returns
    -------
    pd.Series
        Cleaned series with known artefact bytes replaced and U+FFFD stripped.
    """
    result = series.copy()
    for artefact, replacement in _ALL_ENCODING_ARTEFACTS.items():
        result = result.str.replace(artefact, replacement, regex=False)
    result = result.str.replace("�", "", regex=False)
    return result


def _detect_encoding_issues(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, list[tuple[str, str, int]]]:
    """
    Scan columns for characters that `_sanitize_series` would modify.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to scan.
    columns : list of str
        Column names to check. Non-existent columns are silently skipped.

    Returns
    -------
    dict of str → list of tuple
        Maps each column name to a list of ``(artefact_char, replacement_char,
        count)`` tuples — one entry per artefact type found in that column.
        Only columns with at least one match are included.
    """
    candidates: dict[str, str] = {**_ALL_ENCODING_ARTEFACTS, "�": ""}
    findings: dict[str, list[tuple[str, str, int]]] = {}

    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        col_hits: list[tuple[str, str, int]] = []
        for artefact, replacement in candidates.items():
            count = int(series.apply(lambda cell, a=artefact: cell.count(a)).sum())
            if count:
                col_hits.append((artefact, replacement, count))
        if col_hits:
            findings[col] = col_hits

    return findings


def _print_audit_summary(
    scanned: list[str],
    findings: dict[str, list[tuple[str, str, int]]],
    promote: bool,
) -> None:
    """
    Print a per-column table of encoding artefacts that would be (or were) changed.

    Parameters
    ----------
    scanned : list of str
        Column names that were scanned.
    findings : dict
        Output of `_detect_encoding_issues`.
    promote : bool
        Whether changes will be applied after this summary.
    """
    if not findings:
        action = "No artefacts found"
    elif promote:
        action = "applying changes"
    else:
        total_issues = sum(c for hits in findings.values() for _, _, c in hits)
        action = f"dry-run, {total_issues} issue(s) found but not promoted"
    print(
        f"\naudit_encoding_columns [{action}]"
        f" — {len(scanned)} column(s) scanned, {len(findings)} with artefacts"
    )

    hdr = f"  {'Original char':<14}  {'Unicode':<10}  {'Transformed to':<45}  Count"
    sep = "  " + "-" * (len(hdr) - 2)

    for col in scanned:
        hits = findings.get(col)
        if hits:
            total = sum(c for _, _, c in hits)
            print(f"\n  Column: {col}  ({total} substitution(s))")
            print(hdr)
            print(sep)
            for artefact, replacement, count in hits:
                orig = _char_repr(artefact)
                uni  = f"U+{ord(artefact):04X}"
                if replacement:
                    try:
                        name = unicodedata.name(replacement)
                    except ValueError:
                        name = f"U+{ord(replacement):04X}"
                    repl = f"{replacement}  ({name})"
                else:
                    repl = "(removed)"
                print(f"  {orig:<14}  {uni:<10}  {repl:<45}  {count}")
        else:
            print(f"\n  Column: {col}  (no artefacts found)")


def audit_encoding_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    promote: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Detect and optionally apply Unicode-safe sanitization to free-text prose columns.

    Scans for Windows-1252/Mac OS Roman byte artefacts and U+FFFD replacement
    characters. When `verbose=True`, prints a per-column summary of findings.
    Changes are applied only when `promote=True`.

    Does not apply any encode/decode pass, so characters outside the Latin-1
    range that survive upstream processing (μ, –, curly single quotes) are
    preserved. Note that curly double quotes (U+201C/U+201D) are converted to
    straight double quotes by `sanitize_validation_string` in `read_meta_table`
    before this function runs in a QC hook.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to clean.
    columns : list of str or None
        Column names to scan. If None, all object-dtype columns are checked.
    promote : bool
        If False (default), return `df` unmodified. If True, apply fixes and
        return the cleaned copy.
    verbose : bool
        If True (default), print a per-column summary of findings. Set to
        False when calling from `read_meta_table` to suppress per-table noise.

    Returns
    -------
    pd.DataFrame
        `promote=True`: copy of `df` with the specified columns sanitized.
        `promote=False`: `df` unchanged.
    """
    cols = columns if columns is not None else df.select_dtypes("object").columns.tolist()
    scanned = [col for col in cols if col in df.columns]

    findings = _detect_encoding_issues(df, scanned)
    if verbose:
        _print_audit_summary(scanned, findings, promote)

    if not promote or not findings:
        return df

    out = df.copy()
    for col in findings:
        out[col] = _sanitize_series(out[col])
    return out


def remove_special_characters_ascii_printable(value: object) -> object:
    """
    For URL/DOI-like fields, keep only ASCII printable characters (0x20..0x7E),
    drop control chars, and drop U+FFFD.

    Note: even after .str.encode("latin1", errors="replace").str.decode("utf-8", errors="replace")
    some non-ASCII characters can remain as "?" (U+FFFD) which can cause hyperlinking issues.

    Parameters
    ----------
    value : object
        Cell value to sanitize. None and pd.NA are returned unchanged.

    Returns
    -------
    object
        String with only ASCII printable characters (0x20–0x7E), stripped of
        leading/trailing whitespace. Non-string inputs are converted via `str`.
    """
    if value is None or value is pd.NA:
        return value

    value_str = str(value)

    # Remove Unicode replacement char explicitly.
    value_str = value_str.replace("�", "")

    # Keep only ASCII printable characters (space through ~).
    # This removes things like Ê (U+00CA) and any other non-ASCII artifacts.
    value_str = "".join(
        character
        for character in value_str
        if 32 <= ord(character) <= 126
    )
    return value_str.strip()
