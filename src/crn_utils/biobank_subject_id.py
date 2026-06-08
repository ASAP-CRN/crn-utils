"""
Biobank source ID pattern detection and correction.

Provides BiobankSubjectIdFixer to detect and fix source_subject_id and
source_sample_id values in SUBJECT and SAMPLE tables that don't match the
expected pattern for a given biobank, using algorithmic derivation rules or
sibling-dataset cross-references.
Sibling-datasets mean datasets from the same team (e.g. "smith-*/")
that have already been QC'd and contain metadata/latest/ CSV files.

The BIOBANK_NAME_NORMALIZATION, BIOBANK_PATTERNS, and BIOBANK_DERIVATION_RULES
tables live in crn_utils.cde_vocab — update them there when onboarding new
biobanks.

Use example in a qc_hook:
    from crn_utils.biobank_subject_id import BiobankSubjectIdFixer

    # fix SUBJECT first to ensure SAMPLE can join in biobank_name context, then fix SAMPLE

    def _fix_subject(meta_tables, **_):
        fixer = BiobankSubjectIdFixer(dataset_id=dataset_id, caller_path=__file__)
        meta_tables["SUBJECT"] = fixer.fix_subject(meta_tables["SUBJECT"])
        ...
    def _fix_sample(meta_tables, **_):
        fixer = BiobankSubjectIdFixer(dataset_id=dataset_id, caller_path=__file__)
        meta_tables["SAMPLE"]  = fixer.fix_sample(meta_tables["SAMPLE"], meta_tables["SUBJECT"])
        return meta_tables
"""
import re
import warnings
from pathlib import Path

import pandas as pd

from crn_utils.cde_vocab import (
    BIOBANK_DERIVATION_RULES,
    BIOBANK_NAME_NORMALIZATION,
    BIOBANK_PATTERNS,
    normalize_vocab_key,
)

__all__ = ["BiobankSubjectIdFixer"]

# Matches Excel-mangled date values such as "29-Mar", "Mar-39", "03-29".
# These arise when Excel auto-formats XX-YY numeric strings as month abbreviations.
_DATE_LIKE = re.compile(
    r"^\d{1,2}-[A-Za-z]{3}$|^[A-Za-z]{3}-\d{1,2}$|^\d{2}-\d{2}$",
    re.IGNORECASE,
)


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
    Detect and fix source_subject_id / source_sample_id values that don't
    match the expected biobank pattern, and normalize biobank_name to its
    CDE Enum value.

    Parameters
    ----------
    dataset_id : str
        Full dataset identifier including team prefix
        (e.g. "team-smith-pmdbs-sn-rnaseq"). Used to derive the team slug
        for sibling-dataset globbing and to exclude the current dataset
        from cross-reference lookups.
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
    2. Sibling lookup — join on the id column across same-team datasets whose
       source ID values already match the expected pattern.
    3. ValueError — raised if neither strategy fully resolves all mismatches.
    """

    def __init__(
        self,
        dataset_id: str,
        repo_root: Path | None = None,
        caller_path: Path | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.team = dataset_id.split("-")[1]  # assumes format "team-smith-sn-rnaseq"
        self.dataset_name = "-".join(dataset_id.split("-")[1:])  # strips "team-" prefix
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
        Normalize biobank_name to its CDE Enum value.

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
            lambda v: BIOBANK_NAME_NORMALIZATION.get(normalize_vocab_key(str(v)), v)
        )
        return df

    def detect_mismatches(
        self, df: pd.DataFrame, source_col: str = "source_subject_id"
    ) -> pd.DataFrame:
        """
        Return rows where the source ID column does not match the expected biobank pattern.

        Parameters
        ----------
        df : pd.DataFrame
            Table with `biobank_name` and `source_col` columns.
            For SUBJECT, normalize biobank_name first with `normalize_biobank_name`.
            For SAMPLE, biobank_name must be joined in from SUBJECT beforehand.
        source_col : str
            Name of the source ID column to validate. Default `"source_subject_id"`.

        Returns
        -------
        pd.DataFrame
            Subset of rows where source_col fails the expected regex or where
            biobank_name is not in BIOBANK_PATTERNS (flagged for review).
            Empty DataFrame if all rows match.
        """
        bad_rows = []
        for _, row in df.iterrows():
            biobank = row.get("biobank_name")
            if pd.isna(biobank) or str(biobank).strip() in ("", "NA"):
                continue
            pattern = BIOBANK_PATTERNS.get(normalize_vocab_key(str(biobank)))
            if pattern is None:
                bad_rows.append(row)  # unknown biobank — flag for review
                continue
            try:
                if not re.match(pattern, str(row.get(source_col, ""))):
                    bad_rows.append(row)
            except re.error:
                bad_rows.append(row)
        return pd.DataFrame(bad_rows, columns=df.columns)

    def fix_subject(self, subject_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize biobank_name and fix any source_subject_id values that fail
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
        return self._fix_id_column(
            df,
            id_col="subject_id",
            source_col="source_subject_id",
            csv_name="SUBJECT.csv",
        )

    def fix_sample(
        self, sample_df: pd.DataFrame, subject_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fix source_sample_id values that are date-like Excel artefacts.

        When a source_sample_id matches a date-like pattern (e.g. "29-Mar"
        from a string that should have been 29-03) — a sign that Excel 
        auto-formatted an original XX-YY numeric string — the value is replaced 
        with the already-fixed source_subject_id from subject_df, joined on subject_id.

        This approach is used instead of applying BIOBANK_DERIVATION_RULES to
        sample_id, because: i) those rules are designed for subject_id format
        (e.g. "BN0009") in Scherzer datasets, ii) not always available in SAMPLE, 
        and hence is better to fix on a case-by-case basis.

        Parameters
        ----------
        sample_df : pd.DataFrame
            SAMPLE table with `sample_id`, `subject_id`, and
            `source_sample_id` columns.
        subject_df : pd.DataFrame
            SUBJECT table already processed by `fix_subject`, providing
            corrected `source_subject_id` values keyed on `subject_id`.

        Returns
        -------
        pd.DataFrame
            Copy of sample_df with date-like source_sample_id values replaced
            by the corresponding source_subject_id from subject_df.
        """
        df = sample_df.copy()

        sid_to_ssid = (
            subject_df
            .drop_duplicates(subset=["subject_id"])
            .set_index("subject_id")["source_subject_id"]
        )

        mask = df["source_sample_id"].apply(
            lambda v: bool(_DATE_LIKE.match(str(v)))
            if pd.notna(v) and str(v) not in ("", "NA") else False
        )

        if mask.any():
            df.loc[mask, "source_sample_id"] = (
                df.loc[mask, "subject_id"].map(sid_to_ssid)
            )

        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_id_column(
        self,
        df: pd.DataFrame,
        id_col: str,
        source_col: str,
        csv_name: str,
    ) -> pd.DataFrame:
        """
        Detect and fix mismatches in source_col, scoped to rows by biobank.

        Parameters
        ----------
        df : pd.DataFrame
            Table with `biobank_name`, `id_col`, and `source_col` columns.
        id_col : str
            Primary ID column (e.g. `"subject_id"` or `"sample_id"`).
        source_col : str
            Source ID column to fix (e.g. `"source_subject_id"` or
            `"source_sample_id"`).
        csv_name : str
            Filename to load from sibling `metadata/latest/` directories
            (e.g. `"SUBJECT.csv"` or `"SAMPLE.csv"`).

        Returns
        -------
        pd.DataFrame
            Fixed copy of df.

        Raises
        ------
        ValueError
            If mismatches remain after both fix strategies are exhausted.
        """
        mismatches = self.detect_mismatches(df, source_col=source_col)
        if mismatches.empty:
            return df

        for biobank_name in _unique_biobanks(mismatches):
            result = self._apply_derivation(
                df, biobank_name, id_col=id_col, source_col=source_col, csv_name=csv_name
            )
            if result is not None:
                df = result
                continue

            result = self._apply_sibling_lookup(
                df, biobank_name, id_col=id_col, source_col=source_col, csv_name=csv_name
            )
            if result is not None:
                df = result
                continue

            unresolved = mismatches[mismatches["biobank_name"] == biobank_name]
            raise ValueError(
                f"Could not fix {len(unresolved)} {source_col} value(s) for "
                f"biobank '{biobank_name}'. No derivation rule exists and no sibling "
                f"data was found. Affected {id_col}s: "
                f"{unresolved[id_col].tolist()}"
            )

        return df

    def _apply_derivation(
        self,
        subject_df: pd.DataFrame,
        biobank_name: str,
        id_col: str,
        source_col: str,
        csv_name: str,
    ) -> pd.DataFrame | None:
        """
        Apply the algorithmic derivation rule for biobank_name, if one exists.
        Validates derived values against sibling data before committing.

        Parameters
        ----------
        subject_df : pd.DataFrame
            Table with biobank_name already normalized.
        biobank_name : str
            CDE Enum biobank name.
        id_col : str
            Primary ID column (e.g. `"subject_id"` or `"sample_id"`).
        source_col : str
            Source ID column to fix.
        csv_name : str
            Sibling CSV filename to load for validation.

        Returns
        -------
        pd.DataFrame or None
            Fixed DataFrame if derivation succeeded and validated; None otherwise.
        """
        rule = BIOBANK_DERIVATION_RULES.get(normalize_vocab_key(biobank_name))
        if rule is None:
            return None

        df = subject_df.copy()
        mask = df["biobank_name"] == biobank_name

        reference = self._load_sibling_reference(
            biobank_name, id_col=id_col, source_col=source_col, csv_name=csv_name
        )
        if not reference.empty:
            ref_map = reference.set_index(id_col)[source_col]
            overlap = df[mask & df[id_col].isin(reference[id_col])]
            conflicts = overlap[
                overlap[id_col].map(ref_map) != overlap[id_col].map(rule)
            ]
            if not conflicts.empty:
                warnings.warn(
                    f"Derivation rule for '{biobank_name}' conflicts with sibling "
                    f"reference on {len(conflicts)} {id_col}(s): "
                    f"{conflicts[id_col].tolist()}. Falling back to lookup.",
                    stacklevel=3,
                )
                return None

        df.loc[mask, source_col] = df.loc[mask, id_col].map(rule)
        return df

    def _apply_sibling_lookup(
        self,
        subject_df: pd.DataFrame,
        biobank_name: str,
        id_col: str,
        source_col: str,
        csv_name: str,
    ) -> pd.DataFrame | None:
        """
        Fix source ID using a lookup built from sibling CSV files.

        Parameters
        ----------
        subject_df : pd.DataFrame
            Table with biobank_name already normalized.
        biobank_name : str
            CDE Enum biobank name used to filter sibling rows by expected pattern.
        id_col : str
            Primary ID column (e.g. `"subject_id"` or `"sample_id"`).
        source_col : str
            Source ID column to fix.
        csv_name : str
            Sibling CSV filename to load (e.g. `"SUBJECT.csv"` or `"SAMPLE.csv"`).

        Returns
        -------
        pd.DataFrame or None
            Fixed DataFrame if all mismatches were resolved; None if the reference
            is empty or unresolved mismatches remain.
        """
        reference = self._load_sibling_reference(
            biobank_name, id_col=id_col, source_col=source_col, csv_name=csv_name
        )
        if reference.empty:
            return None

        df = subject_df.copy()
        mask = df["biobank_name"] == biobank_name
        lookup = reference.set_index(id_col)[source_col]
        df.loc[mask, source_col] = (
            df.loc[mask, id_col].map(lookup).fillna(df.loc[mask, source_col])
        )

        still_bad = self.detect_mismatches(df, source_col=source_col)
        if not still_bad.empty:
            warnings.warn(
                f"Sibling lookup could not resolve {len(still_bad)} "
                f"{source_col} value(s): {still_bad[id_col].tolist()}",
                stacklevel=3,
            )
            return None

        return df

    def _load_sibling_reference(
        self,
        biobank_name: str,
        id_col: str,
        source_col: str,
        csv_name: str,
    ) -> pd.DataFrame:
        """
        Build an id_col → source_col reference from sibling datasets.

        Searches `metadata/latest/{csv_name}` in all datasets matching
        `datasets/{team}-*/` (current dataset excluded). Uses `latest/`
        because `original/` is not retained post-QC.

        Parameters
        ----------
        biobank_name : str
            CDE Enum biobank name used to filter rows by expected pattern.
        id_col : str
            Primary ID column to load (e.g. `"subject_id"` or `"sample_id"`).
        source_col : str
            Source ID column to load (e.g. `"source_subject_id"` or
            `"source_sample_id"`).
        csv_name : str
            Filename to glob within sibling `metadata/latest/` directories.

        Returns
        -------
        pd.DataFrame
            Deduplicated reference with columns [id_col, source_col].
            Empty if no usable sibling data found.
        """
        pattern = BIOBANK_PATTERNS.get(normalize_vocab_key(biobank_name))
        frames: list[pd.DataFrame] = []

        for path in sorted(
            self.repo_root.glob(f"datasets/{self.team}-*/metadata/latest/{csv_name}")
        ):
            if self.dataset_name in path.parts:
                continue
            try:
                df = pd.read_csv(path, usecols=[id_col, source_col])
            except (FileNotFoundError, ValueError):
                continue
            if pattern:
                df = df[df[source_col].str.match(pattern, na=False)]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=[id_col, source_col])
        return (
            pd.concat(frames)
            .drop_duplicates(id_col)
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
