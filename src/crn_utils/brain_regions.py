"""
Brain region string normalization and CDE Validation string lookup for ASAP CRN metadata.

Function `map_brain_region` is a wrapper to use dictionaries and functions from cde_vocab
to CDE-compliant values as follows:

Step  Function / Dict          Input             Intermediate       Output
----  -----------------------  ----------------  -----------------  -------------------------------------------
1     normalize_vocab_key      "Frontal_Cortex"  None               "frontal cortex"
2a    BRAIN_REGION_NAME_TO_CDE "frontal cortex"  None               "Frontal cortex (F_CTX, UBERON:0001870)"
2b    BRAIN_REGION_NAME_TO_CDE "frontal ctx"     "frontal cortex"   "Frontal cortex (F_CTX, UBERON:0001870)"
2c    BRAIN_REGION_CODE_TO_CDE "f ctx"           "frontal cortex"   "Frontal cortex (F_CTX, UBERON:0001870)"
2d    BRAIN_REGION_CODE_TO_CDE "snd"             "sn"               "Substantia nigra (SN, UBERON:0002038)"

1  Normalizes common variants — different cases, underscores, hyphens, and runs of spaces.
2a covers canonical brain region name lookups.
2b covers legacy brain region name lookups (typos, partial names).
2c covers canonical brain region short code lookups.
2d covers legacy brain region short code lookups (team-specific or aliased).

Semicolon-separated inputs (AllowMultiEnum fields) are split, each element mapped
individually, and the results deduplicated before being re-joined with ";".

Use examples:

1. Legacy or canonical region names mapped to CDE compliant vocabulary string, like:
    "frontal ctx" → "Frontal cortex (F_CTX, UBERON:0001870)" or
    "frontal cortex" → "Frontal cortex (F_CTX, UBERON:0001870)"

    df["region_level_X"] = df["brain_region"].map(
        map_brain_region("brain_region_name", "brain_cde_vocab"))

2. Legacy or canonical region codes mapped to CDE compliant vocabulary string, like:
    "SND" →  "SN" → "Substantia nigra (SN, UBERON:0002038)" or
    "SN" → "Substantia nigra (SN, UBERON:0002038)"

    df["region_level_X"] = df["region_level_Y"].map(
        map_brain_region("brain_short_code", "brain_cde_vocab"))

3. Fix wrong UBERON IDs in a CDE-like string (i.e. use region name as mapping key), like:
    "Frontal cortex (F_CTX, UBERON:0000000)" → "Frontal cortex (F_CTX, UBERON:0001870)"

4. Legacy region names that are split into two regions, like
    "Grey and white matter" → "Grey matter (GM, UBERON:0002020);White matter (WM, UBERON:0002316)"

    df["region_level_X"] = df["region_level_Y"].map(
        map_brain_region("brain_region_name", "brain_cde_vocab"))
"""
from typing import Callable

from crn_utils.cde_vocab import (
    BRAIN_REGION_NAME_TO_CDE,
    BRAIN_REGION_CODE_TO_CDE,
    normalize_vocab_key,
)

__all__ = [
    "map_brain_region",
]

_UNMAPPED_PLACEHOLDER = "Unmapped_brain_annotation"


def _extract_region_name(full_cde: str) -> str:
    idx = full_cde.rfind(" (")
    return full_cde[:idx].strip() if idx != -1 else full_cde.strip()


def _extract_short_code(full_cde: str) -> str:
    if " (" not in full_cde:
        return ""
    inner = full_cde.split(" (", 1)[1]
    return inner.split(",")[0].strip()


def map_brain_region(
        type_of_input: str,
        type_of_output: str) -> Callable[[str], str]:
    """
    Build a mapping callable for use with `df[col].map()` in qc_hooks.

    Converts between brain region representations including legacy and canonical forms.
    Raw inputs are normalized internally with normalize_vocab_key before lookup.

    Semicolon-separated values (AllowMultiEnum fields) are split, each element
    mapped individually, and the results deduplicated before being re-joined
    with ";". Returns `_UNMAPPED_PLACEHOLDER` if no element resolves successfully.

    Parameters
    ----------
    type_of_input : str
        Input representation. One of:

        "brain_region_name"
            Any plain region name without UBERON annotation. Covers canonical
            CDE display names (e.g. "Frontal cortex") and legacy submitted
            names that differ from CDE display names (e.g. "frontal ctx",
            "caudate").
        "brain_short_code"
            CDE short code, canonical or legacy form (e.g. "F_CTX", "SN",
            "SND", "para"). Canonical codes are resolved directly; non-standard
            codes are resolved via their mapped CDE display name.
        "brain_cde_vocab"
            Full CDE Validation string (e.g. "Frontal cortex (F_CTX,
            UBERON:0001870)"). The display-name prefix is extracted and looked
            up, so entries with incorrect UBERON IDs still resolve correctly.

    type_of_output : str
        Output representation. One of "brain_region_name", "brain_short_code", "brain_cde_vocab".
        See type_of_input for examples of each representation.


    Returns
    -------
    Callable[[str], str]
        A function suitable for `df[col].map()`. Returns `_UNMAPPED_PLACEHOLDER`
        for unrecognized inputs.

    Examples
    --------
    Legacy or canonical region names mapped to CDE compliant vocabulary string:

        df["region_level_2"] = df["brain_region"].map(
            map_brain_region("brain_region_name", "brain_cde_vocab"))

    Legacy or canonical region codes mapped to CDE compliant vocabulary string:

        df["region_level_2"] = df["region_level_2"].map(
            map_brain_region("brain_short_code", "brain_cde_vocab"))

    Fix wrong UBERON IDs in a CDE-like string (display name used as mapping key):

        df["region_level_2"] = df["region_level_2"].map(
            map_brain_region("brain_cde_vocab", "brain_cde_vocab"))
    """
    def _get_output(full_cde: str) -> str | None:
        if type_of_output == "brain_cde_vocab":
            return full_cde
        if type_of_output == "brain_region_name":
            return _extract_region_name(full_cde)
        if type_of_output == "brain_short_code":
            return _extract_short_code(full_cde)
        raise ValueError(f"Unknown type_of_output: {type_of_output!r}")

    if type_of_input == "brain_region_name":
        def _map_one(raw: str) -> str | None:
            cde = BRAIN_REGION_NAME_TO_CDE.get(normalize_vocab_key(raw))
            return _get_output(cde) if cde else None

    elif type_of_input == "brain_short_code":
        def _map_one(raw: str) -> str | None:
            cde = BRAIN_REGION_CODE_TO_CDE.get(normalize_vocab_key(raw))
            return _get_output(cde) if cde else None

    elif type_of_input == "brain_cde_vocab":
        def _map_one(raw: str) -> str | None:
            cde = BRAIN_REGION_NAME_TO_CDE.get(normalize_vocab_key(_extract_region_name(raw)))
            return _get_output(cde) if cde else None

    else:
        raise ValueError(f"Unknown type_of_input: {type_of_input!r}")

    def _map(raw: str) -> str:
        if not isinstance(raw, str):
            return _UNMAPPED_PLACEHOLDER
        if ";" not in raw:
            result = _map_one(raw)
            return result if result is not None else _UNMAPPED_PLACEHOLDER
        seen: set[str] = set()
        unique: list[str] = []
        for part in raw.split(";"):
            mapped = _map_one(part.strip())
            if mapped is not None and mapped not in seen:
                seen.add(mapped)
                unique.append(mapped)
        return ";".join(unique) if unique else _UNMAPPED_PLACEHOLDER

    return _map
