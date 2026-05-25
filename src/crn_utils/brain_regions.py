"""
Brain region string normalization and code lookup for ASAP CRN metadata.

Provides a lightweight normalization step before looking up raw submitted
brain_region strings against the controlled vocabulary tables in cde_vocab.

Normalization collapses the most common surface-form variants — different
cases, underscores, hyphens, and runs of spaces — so that cde_vocab.py
only needs one canonical lowercase key per brain region.

Note: releases <= v4.0.0 used `brain_region` values that have been mapped
      to the CDE v4.4+ `region_level_2` controlled vocabularies as specified in cde_vocab.py

Brain mapping chain for region_level_2 (CDE >= v4.5 compliant) for qc_hooks output:
Step  Function                              Input             Output
----  ------------------------------------  ----------------  -------------------------------------------
1     normalize_brain_region_key            "Frontal_Cortex"  "frontal cortex"
2     get_region_code                       "frontal cortex"  "F_CTX"
3     get_region_level                      "F_CTX"           "Frontal cortex (F_CTX, UBERON:0001870)"
Note: different dataset qc_hooks may enter this chain at different steps,
      that's why we expose the individual functions.

Use examples:

1) Normalize submitted legacy brain_region strings to SAMPLE.region_level_2:
    from crn_utils.brain_regions import brain_dicts

    df["region_level_2"] = df["brain_region"].map(
        brain_dicts("legacy_region_name", "remap_cde"))

2) Fix wrong UBERON IDs from a prior release (correct display name, wrong UBERON):
    from crn_utils.brain_regions import brain_dicts

    df["region_level_2"] = df["region_level_2"].map(
        brain_dicts("remap_cde", "remap_cde"))

3) Strip UBERON suffix, keeping only the display name:
    from crn_utils.brain_regions import brain_dicts

    df["region_level_2"] = df["region_level_2"].map(
        brain_dicts("remap_cde", "region_name"))

4) Normalize fields in aggregation by region for summary_stats:
    from crn_utils.brain_regions import get_region_code, get_region_title
    brain_code   = sw_df["brain_region"].map(get_region_code).value_counts().to_dict()
    brain_region = sw_df["brain_region"].map(get_region_title).value_counts().to_dict()
"""
from typing import Callable

from crn_utils.cde_vocab import (
    LEGACY_BRAIN_REGION_CODE_TO_REGION_CODE,
    LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE,
    BRAIN_LEVELS_CDE,
    BRAIN_REGION_CODE_TO_CDE,
    normalize_vocab_key,
)

__all__ = [
    "normalize_brain_region_key",
    "get_region_code",
    "get_region_title",
    "get_region_level",
    "brain_dicts",
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
    Map a raw `brain_region` string to its CDE short code
    (e.g. "Frontal Cortex" → "F_CTX").

    Checks `LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE` first (submitted region
    names such as "frontal cortex"), then falls back to
    `LEGACY_BRAIN_REGION_CODE_TO_REGION_CODE` (legacy team-specific short
    codes such as "amy" or "snd").

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").

    Returns
    -------
    str or None
        CDE short code (e.g. "F_CTX"), or `None` if not recognized.
    """
    key  = normalize_brain_region_key(str(raw))
    code = LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE.get(key)
    if code is None:
        code = LEGACY_BRAIN_REGION_CODE_TO_REGION_CODE.get(key)
    return code


def get_region_title(raw: str) -> str | None:
    """
    Map a raw `brain_region` string to its legacy display title
    (e.g. "Frontal_Cortex" → "Frontal Cortex").

    Uses `BRAIN_REGION_CODE_TO_CDE` for the title lookup. For releases
    > v4.0.0, prefer `get_region_level` which returns the full CDE
    `region_level_<N>` Validation string including UBERON ID.

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").

    Returns
    -------
    str or None
        Legacy display title (e.g. "Frontal Cortex"), or `None` if not
        recognized.
    """
    code = get_region_code(raw)
    if code is None:
        return None
    return BRAIN_REGION_CODE_TO_CDE.get(normalize_vocab_key(code))


def get_region_level(raw: str, level: int) -> str | None:
    """
    Map a raw `brain_region` string to its CDE >= v4.5 `region_level_<level>` string.
    (e.g. "Frontal_Cortex" → "Frontal cortex (F_CTX, UBERON:0001870)").

    Parameters
    ----------
    raw : str
        Raw `brain_region` string (e.g. "Frontal_Cortex").
    level : int
        Level of the region (e.g. 2 for `region_level_2`).

    Returns
    -------
    str or None
        CDE >= v4.5 compliant `region_level_<level>` string (e.g.
        "Frontal cortex (F_CTX, UBERON:0001870)"), or `None` if not recognized.
    """
    code = get_region_code(raw)
    if code is None:
        return None
    return BRAIN_LEVELS_CDE.get(f"region_level_{level}", {}).get(normalize_vocab_key(code))


def brain_dicts(
        type_of_input: str,
        type_of_output: str,
        region_level: str = "region_level_2") -> Callable[[str], str | None]:
    """
    Build a mapping callable for use with `df[col].map()` in qc_hooks.

    Converts between any two brain-region representations: submitted legacy
    names, CDE short codes, display names, and CDE Validation strings.

    Parameters
    ----------
    type_of_input : str
        Input representation. One of:

        "region_code"
            CDE short code, canonical or legacy form (e.g. "F_CTX", "f ctx",
            "amy", "snd"). Normalized before lookup; legacy codes such as
            "snd" are resolved via `LEGACY_BRAIN_REGION_CODE_TO_REGION_CODE`.
        "legacy_region_name"
            Submitted region name key from
            `LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE` (e.g. "frontal cortex",
            "frontal lobe"). Normalized before lookup.
        "region_name"
            CDE display name from `brain_region_levels.display_name`
            (e.g. "Frontal cortex"). Case-insensitive match.
        "remap_cde"
            Full CDE Validation string
            (e.g. "Frontal cortex (F_CTX, UBERON:0001870)"). The display-name
            prefix is parsed and matched case-insensitively, so entries with
            incorrect UBERON IDs still resolve correctly.

    type_of_output : str
        Output representation. Same values as `type_of_input`.

    region_level : str, optional
        Key into `BRAIN_LEVELS_CDE` to select the level.
        Default "region_level_2".

    Returns
    -------
    Callable[[str], str | None]
        A function suitable for `df[col].map()`. Returns `None` for
        unrecognized inputs (rendered as NaN by pandas).

    Examples
    --------
    Submitted legacy name → full CDE region_level_2 string:

        df["region_level_2"] = df["brain_region"].map(
            brain_dicts("legacy_region_name", "remap_cde"))

    Fix wrong UBERON IDs from a prior release (correct display name preserved):

        df["region_level_2"] = df["region_level_2"].map(
            brain_dicts("remap_cde", "remap_cde"))

    Strip UBERON suffix, keeping only the display name:

        df["region_level_2"] = df["region_level_2"].map(
            brain_dicts("remap_cde", "region_name"))
    """
    level_map = BRAIN_LEVELS_CDE.get(region_level, {})

    def _extract_raw_code(full_cde: str) -> str:
        if " (" not in full_cde:
            return ""
        inner = full_cde.split(" (", 1)[1]
        return inner.split(",")[0].strip()

    def _extract_display_name(s: str) -> str:
        idx = s.rfind(" (")
        return s[:idx].strip() if idx != -1 else s.strip()

    # display_name (lowercase) → normalized_code, built once at call time
    display_lower_to_norm_code: dict[str, str] = {
        _extract_display_name(full_cde).lower(): norm_code
        for norm_code, full_cde in level_map.items()
    }

    def _get_output(norm_code: str) -> str | None:
        full_cde = level_map.get(norm_code)
        if full_cde is None:
            return None
        if type_of_output == "remap_cde":
            return full_cde
        if type_of_output == "region_name":
            return _extract_display_name(full_cde)
        if type_of_output == "region_code":
            return _extract_raw_code(full_cde)
        if type_of_output == "legacy_region_name":
            raw_code   = _extract_raw_code(full_cde)
            norm_target = normalize_vocab_key(raw_code)
            return next(
                (k for k, v in LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE.items()
                 if normalize_vocab_key(v) == norm_target),
                None,
            )
        raise ValueError(f"Unknown type_of_output: {type_of_output!r}")

    def _norm_code_from_region_code(raw: str) -> str | None:
        norm = normalize_brain_region_key(raw)
        if norm in level_map:
            return norm
        canonical = LEGACY_BRAIN_REGION_CODE_TO_REGION_CODE.get(norm)
        if canonical is not None:
            return normalize_vocab_key(canonical)
        return None

    def _norm_code_from_legacy_name(raw: str) -> str | None:
        code = LEGACY_BRAIN_REGION_NAME_TO_REGION_CODE.get(normalize_brain_region_key(raw))
        return normalize_vocab_key(code) if code is not None else None

    def _norm_code_from_display_name(display: str) -> str | None:
        return display_lower_to_norm_code.get(display.strip().lower())

    if type_of_input == "region_code":
        def _map(raw: str) -> str | None:
            if not isinstance(raw, str):
                return None
            norm_code = _norm_code_from_region_code(raw)
            return _get_output(norm_code) if norm_code else None

    elif type_of_input == "legacy_region_name":
        def _map(raw: str) -> str | None:
            if not isinstance(raw, str):
                return None
            norm_code = _norm_code_from_legacy_name(raw)
            return _get_output(norm_code) if norm_code else None

    elif type_of_input == "region_name":
        def _map(raw: str) -> str | None:
            if not isinstance(raw, str):
                return None
            norm_code = _norm_code_from_display_name(raw)
            return _get_output(norm_code) if norm_code else None

    elif type_of_input == "remap_cde":
        def _map(raw: str) -> str | None:
            if not isinstance(raw, str):
                return None
            norm_code = _norm_code_from_display_name(_extract_display_name(raw))
            return _get_output(norm_code) if norm_code else None

    else:
        raise ValueError(f"Unknown type_of_input: {type_of_input!r}")

    return _map
