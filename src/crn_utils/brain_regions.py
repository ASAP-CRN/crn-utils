"""
Brain region string normalization and code lookup for ASAP CRN metadata.

Provides a lightweight normalization step before looking up raw submitted
brain_region strings against the controlled vocabulary tables in cde_vocab.

Normalization collapses the most common surface-form variants — different
cases, underscores, hyphens, and runs of spaces — so that cde_vocab.py
only needs one canonical lowercase key per brain region.

Note: releases <= v4.0.0 used `brain_region` values that have been mapped
      to the CDE v4.4+ `region_level_2` controlled vocabularies as specified in cde_vocab.py

Brain mapping chain for region_level_2 (final CDE >= v4.4 compliant) for qc_hooks output:
Step  Function                     Input                       Output
----  ---------------------------  --------------------------  -----------------------------------------------
1     normalize_brain_region_key   "Frontal_Cortex"            "frontal cortex"
2     get_region_code              "frontal cortex"            "F_CTX"
3     get_region_level2            "F_CTX"                     "Frontal cortex (F_CTX, UBERON:0001870)"
Note: different dataset qc_hooks may enter this chain at different steps,
      that's why we expose the individual functions.

Use examples:

1) Normalize fields in a qc_hook to populate SAMPLE.region_level_2:
    from crn_utils.brain_regions import get_region_level2
    df["region_level_2"] = df["brain_region"].map(get_region_level2)

2) Normalize fields in aggregation by region for summary_stats:
    from crn_utils.brain_regions import get_region_code, get_region_title
    brain_code   = sw_df["brain_region"].map(get_region_code).value_counts().to_dict()
    brain_region = sw_df["brain_region"].map(get_region_title).value_counts().to_dict()
"""
from crn_utils.cde_vocab import (
    BRAIN_L2_UBERON,
    BRAIN_REGION_CODES,
    BRAIN_REGION_TITLES,
    normalize_vocab_key,
)

__all__ = [
    "normalize_brain_region_key",
    "get_region_code",
    "get_region_title",
    "get_region_level2",
]


def normalize_brain_region_key(raw: str) -> str:
    """
    Normalize a raw `brain_region` string for dict lookup.

    Strips surrounding whitespace, lowercases, replaces `_` and `-` with
    spaces, then collapses runs of spaces to a single space.
    Delegates to `normalize_vocab_key` from `crn_utils.cde_vocab`.

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").

    Returns
    -------
    str
        Normalized string (e.g. "frontal cortex").
    """
    return normalize_vocab_key(raw)


def get_region_code(raw: str) -> str | None:
    """
    Map a raw `brain_region` string to its short code
    (e.g. Frontal Cortex → F_CTX).

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").

    Returns
    -------
    str or None
        Short code (e.g. "F_CTX"), or `None` if not recognized.
    """
    return BRAIN_REGION_CODES.get(normalize_brain_region_key(str(raw)))


def get_region_title(raw: str) -> str | None:
    """
    Map a raw `brain_region` string to its canonical BRAIN_REGION_TITLES title.
    (e.g. Frontal_Cortex → Frontal Cortex).

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").

    Returns
    -------
    str or None
        Canonical BRAIN_REGION_TITLES title (e.g. "Frontal Cortex"), or `None` if not
        recognized.
    """
    code = get_region_code(raw)
    if code is None:
        return None
    return BRAIN_REGION_TITLES.get(normalize_vocab_key(code))


def get_region_level2(raw: str) -> str | None:
    """
    Map a raw `brain_region` string to its CDE >= v4.4 `region_level_2` string.
    (e.g. Frontal_Cortex → Frontal cortex (F_CTX, UBERON:0001876)).

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Substantia_Nigra").

    Returns
    -------
    str or None
        CDE >= v4.4 compliant `region_level_2` string (e.g.
        "Substantia nigra (SN, UBERON:0002038)"), or `None` if not recognized.
    """
    code = get_region_code(raw)
    if code is None:
        return None
    return BRAIN_L2_UBERON.get(normalize_vocab_key(code))
