"""
Biobank source_subject_id pattern detection and correction.

Provides BiobankSubjectIdFixer to detect and fix source_subject_id values
in SUBJECT tables that don't match the expected pattern for a given biobank,
using algorithmic derivation rules or sibling-dataset cross-references.
Sibling-datasets mean datasets from the same team (e.g. "scherzer-*/")
that have already been QC'd and contain metadata/latest/SUBJECT.csv files.

The BIOBANK_NAME_NORMALIZATION, BIOBANK_PATTERNS, and BIOBANK_DERIVATION_RULES
tables live in crn_utils.cde_vocab — update them there when onboarding new
biobanks.

Use example in a qc_hook:
    from crn_utils.biobank_subject_id import BiobankSubjectIdFixer
    def _fix_subject(meta_tables, **_):
        fixer = BiobankSubjectIdFixer(dataset_name=dataset_name, caller_path=__file__)
        meta_tables["SUBJECT"] = fixer.fix(meta_tables["SUBJECT"])
        return meta_tables
"""
import re
import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

from crn_utils.cde_vocab import (
    BIOBANK_DERIVATION_RULES,
    BIOBANK_NAME_NORMALIZATION,
    BIOBANK_PATTERNS,
)

__all__ = ["BiobankSubjectIdFixer"]


def _find_repo_root(start: Path) -> Path:
    """
    Walk up from start until a directory containing `.git` is found.

    Parameters
    ----------
    start : Path
        Starting path (typically `Path(__file__)` of the calling module).

    Returns
    -------
    Path
        The nearest ancestor directory that contains a `.git` entry.

    Raises
    ------
    RuntimeError
        If no `.git` directory is found between start and the filesystem root.
    """
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(
        f"Could not locate a git repository root above '{start}'. "
        "Pass repo_root explicitly if running outside the repository."
    )


class BiobankSubjectIdFixer:
    """
    Detect and fix source_subject_id values that don't match the expected
    biobank pattern, and normalize biobank_name to its CDE Enum value.

    Parameters
    ----------
    dataset_name : str
        Hyphenated dataset name without the `team-` prefix
        (e.g. `"scherzer-pmdbs-lr-wgs"`).
        The first token is used to glob sibling datasets for cross-reference.
    repo_root : Path, optional
        Root of the asap-crn-cloud-dataset-metadata repository.
        When omitted, auto-detected via `.git` traversal from caller_path.
    caller_path : Path, optional
        Path to the calling file (pass `__file__` from the qc_hook).
        Used as the starting point for `.git` traversal when repo_root is
        omitted, so the walk lands in asap-crn-cloud-dataset-metadata rather
        than in crn-utils.

    Notes
    -----
    Fix strategy precedence (per biobank):

    1. Algorithmic derivation — if a rule exists in BIOBANK_DERIVATION_RULES,
       apply it and validate against sibling data before committing.
    2. Sibling lookup — join on subject_id across same-team datasets whose
       source_subject_id values already match the expected pattern.
    3. ValueError — raised if neither strategy fully resolves all mismatches.
    """

    def __init__(
        self,
        dataset_name: str,
        repo_root: Path | None = None,
        caller_path: Path | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.team = dataset_name.split("-")[0]
        if repo_root is not None:
            self.repo_root = Path(repo_root)
        else:
            # Walk up from the calling file (qc_hook) rather than this library
            # file, so we land in asap-crn-cloud-dataset-metadata, not crn-utils.
            start = Path(caller_path) if caller_path is not None else Path(__file__)
            self.repo_root = _find_repo_root(start)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_biobank_name(self, subject_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise biobank_name to its CDE Enum value.

        Parameters
        ----------
        subject_df : pd.DataFrame
            SUBJECT table containing a `biobank_name` column.

        Returns
        -------
        pd.DataFrame
            Copy of subject_df with biobank_name values mapped through
            BIOBANK_NAME_NORMALIZATION; unrecognised values are left unchanged.
        """
        df = subject_df.copy()
        df["biobank_name"] = df["biobank_name"].map(
            lambda v: BIOBANK_NAME_NORMALIZATION.get(str(v), v)
        )
        return df

    def detect_mismatches(self, subject_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return rows where source_subject_id does not match the expected biobank pattern.

        Parameters
        ----------
        subject_df : pd.DataFrame
            SUBJECT table with `biobank_name` and `source_subject_id` columns.
            Normalise biobank_name first with `normalize_biobank_name`.

        Returns
        -------
        pd.DataFrame
            Subset of rows where source_subject_id fails the expected regex or
            where biobank_name is not in BIOBANK_PATTERNS (flagged for review).
            Empty DataFrame if all rows match.
        """
        bad_rows = []
        for _, row in subject_df.iterrows():
            biobank = row.get("biobank_name")
            if pd.isna(biobank) or str(biobank).strip() in ("", "NA"):
                continue
            pattern = BIOBANK_PATTERNS.get(str(biobank))
            if pattern is None:
                bad_rows.append(row)  # unknown biobank — flag for review
                continue
            if not re.match(pattern, str(row.get("source_subject_id", ""))):
                bad_rows.append(row)
        return pd.DataFrame(bad_rows, columns=subject_df.columns)

    def fix(self, subject_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise biobank_name and fix any source_subject_id values that fail
        the expected pattern.

        Parameters
        ----------
        subject_df : pd.DataFrame
            SUBJECT table with `subject_id`, `biobank_name`, and
            `source_subject_id` columns.

        Returns
        -------
        pd.DataFrame
            Copy with biobank_name normalized and source_subject_id corrected.

        Raises
        ------
        ValueError
            If mismatches remain after both fix strategies are exhausted for
            any biobank present in the table.
        """
        df = self.normalize_biobank_name(subject_df)
        mismatches = self.detect_mismatches(df)
        if mismatches.empty:
            return df

        for biobank_name in _unique_biobanks(mismatches):
            result = self._apply_derivation(df, biobank_name)
            if result is not None:
                df = result
                continue

            result = self._apply_sibling_lookup(df, biobank_name)
            if result is not None:
                df = result
                continue

            unresolved = mismatches[mismatches["biobank_name"] == biobank_name]
            raise ValueError(
                f"Could not fix {len(unresolved)} source_subject_id value(s) for "
                f"biobank '{biobank_name}'. No derivation rule exists and no sibling "
                f"data was found. Affected subject_ids: "
                f"{unresolved['subject_id'].tolist()}"
            )

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_derivation(
        self, subject_df: pd.DataFrame, biobank_name: str
    ) -> pd.DataFrame | None:
        """
        Apply the algorithmic derivation rule for biobank_name, if one exists.
        Validates derived values against sibling data before committing.

        Parameters
        ----------
        subject_df : pd.DataFrame
            SUBJECT table with biobank_name already normalized.
        biobank_name : str
            CDE Enum biobank name.

        Returns
        -------
        pd.DataFrame or None
            Fixed DataFrame if derivation succeeded and validated; None otherwise.
        """
        rule = BIOBANK_DERIVATION_RULES.get(biobank_name)
        if rule is None:
            return None

        df = subject_df.copy()
        derived = df["subject_id"].map(rule)

        reference = self._load_sibling_reference(biobank_name)
        if not reference.empty:
            ref_map = reference.set_index("subject_id")["source_subject_id"]
            overlap = df[df["subject_id"].isin(reference["subject_id"])]
            conflicts = overlap[
                overlap["subject_id"].map(ref_map) != overlap["subject_id"].map(rule)
            ]
            if not conflicts.empty:
                warnings.warn(
                    f"Derivation rule for '{biobank_name}' conflicts with sibling "
                    f"reference on {len(conflicts)} subject_id(s): "
                    f"{conflicts['subject_id'].tolist()}. Falling back to lookup.",
                    stacklevel=3,
                )
                return None

        df["source_subject_id"] = derived
        return df

    def _apply_sibling_lookup(
        self, subject_df: pd.DataFrame, biobank_name: str
    ) -> pd.DataFrame | None:
        """
        Fix source_subject_id using a lookup built from sibling SUBJECT files.

        Parameters
        ----------
        subject_df : pd.DataFrame
            SUBJECT table with biobank_name already normalized.
        biobank_name : str
            CDE Enum biobank name used to filter sibling rows by expected pattern.

        Returns
        -------
        pd.DataFrame or None
            Fixed DataFrame if all mismatches were resolved; None if the reference
            is empty or unresolved mismatches remain.
        """
        reference = self._load_sibling_reference(biobank_name)
        if reference.empty:
            return None

        df = subject_df.copy()
        lookup = reference.set_index("subject_id")["source_subject_id"]
        df["source_subject_id"] = (
            df["subject_id"].map(lookup).fillna(df["source_subject_id"])
        )

        still_bad = self.detect_mismatches(df)
        if not still_bad.empty:
            warnings.warn(
                f"Sibling lookup could not resolve {len(still_bad)} "
                f"source_subject_id value(s): {still_bad['subject_id'].tolist()}",
                stacklevel=3,
            )
            return None

        return df

    def _load_sibling_reference(self, biobank_name: str) -> pd.DataFrame:
        """
        Build a subject_id → source_subject_id reference from sibling datasets.

        Searches `metadata/latest/SUBJECT.csv` in all datasets matching
        `datasets/{team}-*/` (current dataset excluded). Uses `latest/`
        because `original/` is not retained post-QC.

        Parameters
        ----------
        biobank_name : str
            CDE Enum biobank name used to filter rows by expected pattern.

        Returns
        -------
        pd.DataFrame
            Deduplicated reference with columns [subject_id, source_subject_id].
            Empty if no usable sibling data found.
        """
        pattern = BIOBANK_PATTERNS.get(biobank_name)
        frames: list[pd.DataFrame] = []

        for path in sorted(
            self.repo_root.glob(f"datasets/{self.team}-*/metadata/latest/SUBJECT.csv")
        ):
            if self.dataset_name in path.parts:
                continue
            try:
                df = pd.read_csv(path, usecols=["subject_id", "source_subject_id"])
            except (FileNotFoundError, ValueError):
                continue
            if pattern:
                df = df[df["source_subject_id"].str.match(pattern, na=False)]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["subject_id", "source_subject_id"])
        return (
            pd.concat(frames)
            .drop_duplicates("subject_id")
            .reset_index(drop=True)
        )


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _unique_biobanks(mismatches: pd.DataFrame) -> list[str]:
    """Return unique non-NA biobank names from a mismatches DataFrame."""
    return (
        mismatches["biobank_name"]
        .dropna()
        .pipe(lambda s: s[~s.isin(["", "NA"])])
        .unique()
        .tolist()
    )
