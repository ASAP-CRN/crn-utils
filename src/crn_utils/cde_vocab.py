"""
CDE controlled-vocabulary tables for the ASAP CRN metadata pipeline.

Centralizes fixed mappings between submitted/raw values and CDE-compliant
Enum values for fields that appear across multiple datasets. Import these
tables in qc_hook scripts or auxiliary modules like biobank_subject_id.py
to apply consistent normalization logic.

Keys   = normalized submitted values (strip, lowercase, `_`/`-` → space,
         collapse spaces; see `normalize_vocab_key`).
Values = CDE-compliant Enum values (exactly as they appear in ValidCategories).

Callers must apply `normalize_vocab_key(raw)` to the raw string before a
dict lookup, so that case and separator variants resolve to the correct entry.
"""
import re
from typing import Callable

from crn_utils.google_spreadsheets import read_google_sheet
_CDE_SPREADSHEET_ID      = "1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc"
_BRAIN_REGION_LEVELS_TAB = "brain_region_levels"


__all__ = [

    # Generic normalization
    "normalize_vocab_key",

    # Biobank
    "BIOBANK_NAME_NORMALIZATION",
    "BIOBANK_PATTERNS",
    "BIOBANK_DERIVATION_RULES",

    # Brain region
    "BRAIN_REGION_NAME_TO_CDE",
    "BRAIN_REGION_CODE_TO_CDE",

    # Ontology term IDs
    "SEX_ONTOLOGY",

    # Neuropathology / CLINPATH
    "MCKEITH_LB_NORMALIZATION",
    "BRAAK_NFT_NORMALIZATION",
    "PATH_THAL_NORMALIZATION",
    "NIA_AA_A_NORMALIZATION",
    "NIA_AA_B_NORMALIZATION",
    "NIA_AA_C_NORMALIZATION",
    "PATH_NIA_RI_NORMALIZATION",
    "PATH_AD_LEVEL_NORMALIZATION",
    "AMYLOID_ANGIOPATHY_NORMALIZATION",
    "PATH_AUTOPSY_DX_MAIN_NORMALIZATION",
]


def normalize_vocab_key(raw: str) -> str:
    """
    Normalize a raw string for case- and separator-insensitive dict lookup.

    Strips surrounding whitespace, lowercases, replaces underscores and
    hyphens with spaces, then collapses runs of spaces to a single space.

    All dict keys in this module that represent free-text submitted values
    are pre-normalized using this transformation.  Callers must apply the
    same normalization to the raw string before a dict lookup.

    Parameters
    ----------
    raw : str
        Raw submitted string.

    Returns
    -------
    str
        Normalized string ready for dict lookup.
    """
    s = raw.strip().lower()
    s = re.sub(r"[_\-]", " ", s)
    s = re.sub(r" +", " ", s)
    return s


# ==============================================================================
# MAINTENANCE
# The dictionaries below are manually curated to map CDE-compliant Enum values.
#
# Keys   = normalized submitted values (see normalize_vocab_key above).
# Values = CDE-compliant Enum values (exactly as they appear in ValidCategories).
#
# Initial sources:
#   - summary_stats.py (_brain_region_coder, _region_titles) — brain region tables
#   - biobank_subject_id.py for release v4.1.1+ — biobank name normalization, patterns, derivation rules
#   - qc_hooks for releases >= v4.0.1 (built by DNAstack curators): sex ontology table
#   - QC_hooks for releases <= v4.0.0 (built by DTi): McKeith, NIA-RI, amyloid, AD level tables, Thal, autopsy diagnosis tables
# ==============================================================================

# ------------------------------------------------------------------------------
# Biobank
# biobank_subject_id.BiobankSubjectIdFixer consumes all three BIOBANK* tables below.
# Add an entry here whenever a new biobank is onboarded.
# ------------------------------------------------------------------------------

# Non-CDE submitted value → CDE Enum value.
# Add an entry whenever a contributor uses a non-standard biobank_name spelling.
BIOBANK_NAME_NORMALIZATION: dict[str, str] = {
    "banner sun health usa": "Banner Sun Health Research Institute",
    "cambridge brain bank":  "Cambridge Brain Bank",
    "qsbb uk":               "QSBB UK",
    "imperial uk":           "Imperial UK",
    "edinburgh uk":          "Edinburgh UK",
}

# CDE Enum value → expected regex for source_subject_id.
BIOBANK_PATTERNS: dict[str, str] = {
    "banner sun health research institute": r"^\d{2}-\d{2}$",
    "qsbb uk":                              r"^P\d+/\d+$",
    "cambridge brain bank":                 r"^(BB\d{2}\.\d{4}|NP\d{2}-\d{5})$",
    "new york brain bank":                  r"^T-\d+$",
    "imperial uk":                          r"^(C\d+|PD\d+|PDC\d+)$",
    "sbb":                                  r"^SBB_Case_\d+$",
    "nki/nyugsom":                          r"^hSDG\d+$",
    "bmc":                                  r"^BMC_Case_\d+$",
    "edinburgh uk":                         r"^SD\d+/\d+$",
}

# CDE biobank name → callable(subject_id str) → source_subject_id str.
# Add a rule when a deterministic derivation exists (preferred over sibling lookup).
# Banner Sun: BN<MMDD> → MM-DD; Excel reformats month prefixes (MM ≤ 12) as dates.
BIOBANK_DERIVATION_RULES: dict[str, Callable[[str], str]] = {
    "banner sun health research institute": lambda sid: f"{sid[2:4]}-{sid[4:6]}",
}


# ------------------------------------------------------------------------------
# Brain region
# ------------------------------------------------------------------------------

# - Legacy region names (typo, partial key) mapped to canonical CDE region names.
# - Used internally to build BRAIN_REGION_NAME_TO_CDE.
# - Split multiple CDE entries with ";" (e.g. "grey and white matter")
_LEGACY_BRAIN_REGION_NAME_TO_CDE_NAME: dict[str, str] = {
    "antaerior cortex":          "anterior cingulate cortex", #typo, partial key
    "antaerior cingulate":       "anterior cingulate cortex", #typo, partial key
    "caudate":                   "caudate nucleus",           #partial key
    "frontal ctx":               "frontal cortex",            #partial key
    "inferior parietal lobe":    "inferior parietal lobule",  #partial key
    "parietal":                  "parietal cortex",           #partial key
    "parietal ctx":              "parietal cortex",           #partial key
    "temporal ctx":              "temporal cortex",           #partial key
    "midbrain (mesencephalon)":  "midbrain mesencephalon",    #parentheses aren't handled by normalize_vocab_key
    "midbrain":                  "midbrain mesencephalon",    #partial key
    "grey and white matter":     "grey matter;white matter",  #no single UBERON entry; expand to both
    "transentorhinal region":    "parahippocampal gyrus",     #no single UBERON entry; mapped to closest term
    "ca1 ca4":                   "hippocampus",               #unespecific multiple entries with hippocampus common term
    "basal gaglia":              "basal ganglia",             #typo
}

# Legacy short codes mapped to canonical CDE region names.
# Used internally to build BRAIN_REGION_CODE_TO_CDE.
_LEGACY_BRAIN_REGION_CODE_TO_CDE_NAME: dict[str, str] = {
    "para": "parahippocampal gyrus", # alias for PHG
    "snd":  "substantia nigra",      # team Edwards SN sub-nucleus codes
    "snv":  "substantia nigra",      # team Edwards SN sub-nucleus codes
    "snm":  "substantia nigra",      # team Edwards SN sub-nucleus codes
}


def _build_brain_levels_uberon(
        short_code_column: str,
        region_level_column: str,
        uberon_column: str,
        region_name_column: str) -> dict[str, dict[str, str]]:
    """
    Read the CDE brain_region_levels tab and build a nested dict keyed by region_level.

    Parameters
    ----------
    short_code_column : str
        Column name containing the region short code (e.g. "short_code").
    region_level_column : str
        Column name whose values become the outer dict keys
        (e.g. "region_level", containing values like "region_level_2_intermediate").
    uberon_column : str
        Column name containing the UBERON ID (e.g. "CDE_v4.5_UBERON").
        Rows where this column equals "DELETE" are excluded.
    region_name_column : str
        Column name containing the canonical region name (e.g. "region_name").

    Returns
    -------
    dict[str, dict[str, str]]
        Nested mapping: region_level → normalized short code → CDE Validation string.
        For example: "region_level_2_intermediate" → "f ctx" → "Frontal cortex (F_CTX, UBERON:0001870)".
    """
    df = read_google_sheet(_CDE_SPREADSHEET_ID, _BRAIN_REGION_LEVELS_TAB)
    df = df[df[uberon_column] != "DELETE"]
    result: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        level = row[region_level_column]
        key   = normalize_vocab_key(row[short_code_column])
        value = f"{row[region_name_column]} ({row[short_code_column]}, {row[uberon_column]})"
        result.setdefault(level, {})[key] = value
    return result


# Private nested dict used only to build BRAIN_REGION_NAME_TO_CDE and BRAIN_REGION_CODE_TO_CDE.
_BRAIN_LEVELS_CDE: dict[str, dict[str, str]] = _build_brain_levels_uberon(
    "short_code",
    "region_level",
    "CDE_v4.5_UBERON",
    "region_name"
)


def _build_brain_region_name_to_cde() -> dict[str, str]:
    """
    Build BRAIN_REGION_NAME_TO_CDE from canonical region names and legacy name mappings,
    with a safe check to prevent key collisions.

    Covers two cases:
    - Canonical names: all region names from `_BRAIN_LEVELS_CDE` (case-insensitive).
    - Legacy names: entries in `_LEGACY_BRAIN_REGION_NAME_TO_CDE_NAME` that differ
      from canonical CDE region names (e.g. "frontal ctx" → "frontal cortex").

    Returns
    -------
    dict[str, str]
        Normalized region name (lowercase, spaces) → full CDE Validation string.
        Example: "frontal ctx" → "Frontal cortex (F_CTX, UBERON:0001870)".
    """
    region_name_lower_to_cde: dict[str, str] = {}
    for level, level_dict in _BRAIN_LEVELS_CDE.items():
        for full_cde in level_dict.values():
            name_key = normalize_vocab_key(full_cde[:full_cde.rfind(" (")])
            if name_key in region_name_lower_to_cde and region_name_lower_to_cde[name_key] != full_cde:
                raise ValueError(
                    f"Duplicate region name {name_key!r} across levels: "
                    f"{region_name_lower_to_cde[name_key]!r} vs {full_cde!r} (in {level!r})"
                )
            region_name_lower_to_cde[name_key] = full_cde
    result = dict(region_name_lower_to_cde)
    for name, cde_region_name in _LEGACY_BRAIN_REGION_NAME_TO_CDE_NAME.items():
        if ";" in cde_region_name:
            seen_parts: set[str] = set()
            parts: list[str] = []
            for part in cde_region_name.split(";"):
                cde_part = region_name_lower_to_cde.get(normalize_vocab_key(part.strip()))
                if cde_part is not None and cde_part not in seen_parts:
                    seen_parts.add(cde_part)
                    parts.append(cde_part)
            cde = ";".join(parts) if parts else None
        else:
            cde = region_name_lower_to_cde.get(normalize_vocab_key(cde_region_name))
        if cde is not None:
            if name in result:
                raise ValueError(
                    f"Legacy name {name!r} conflicts with existing entry: {result[name]!r}"
                )
            result[name] = cde
    return result


# Submitted region name → full CDE Validation string, level-independent.
# Covers canonical region names (2a) and legacy submitted names (2b).
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.
BRAIN_REGION_NAME_TO_CDE: dict[str, str] = _build_brain_region_name_to_cde()


def _build_brain_region_code_to_cde() -> dict[str, str]:
    """
    Build BRAIN_REGION_CODE_TO_CDE from canonical short codes and legacy code mappings,
    with a safe check to prevent key collisions.

    Covers two cases:
    - Canonical codes: all normalized short codes from `_BRAIN_LEVELS_CDE`.
    - Legacy codes: entries in `_LEGACY_BRAIN_REGION_CODE_TO_CDE_NAME` that have
      no CDE equivalent (e.g. "snd" → "sn" → "Substantia nigra (SN, …)").

    Returns
    -------
    dict[str, str]
        Normalized short code (lowercase, spaces) → full CDE Validation string.
        Example: "snd" → "Substantia nigra (SN, UBERON:0002038)".
    """
    result: dict[str, str] = {}
    for level, level_dict in _BRAIN_LEVELS_CDE.items():
        for norm_code, full_cde in level_dict.items():
            if norm_code in result:
                raise ValueError(
                    f"Duplicate short code {norm_code!r} across levels: "
                    f"{result[norm_code]!r} vs {full_cde!r} (in {level!r})"
                )
            result[norm_code] = full_cde
    region_name_lower_to_cde: dict[str, str] = {
        normalize_vocab_key(full_cde[:full_cde.rfind(" (")]): full_cde
        for full_cde in result.values()
    }
    for code, cde_region_name in _LEGACY_BRAIN_REGION_CODE_TO_CDE_NAME.items():
        cde = region_name_lower_to_cde.get(normalize_vocab_key(cde_region_name))
        if cde is not None:
            if code in result:
                raise ValueError(
                    f"Legacy code {code!r} conflicts with canonical code: {result[code]!r}"
                )
            result[code] = cde
    return result


# Submitted short code → full CDE Validation string, level-independent.
# Covers canonical CDE short codes (2c) and legacy/aliased codes (2d).
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.
BRAIN_REGION_CODE_TO_CDE: dict[str, str] = _build_brain_region_code_to_cde()


# ------------------------------------------------------------------------------
# Ontology term IDs
# ------------------------------------------------------------------------------

# Maps plain-text sex value → CDE sex_ontology_term_id.
# Keys are pre-normalized. Use normalize_vocab_key() before lookup.
# Source: QC_biederer_pmdbs_spatial_geomx_lamda.py

SEX_ONTOLOGY: dict[str, str] = {
    "male":   "PATO:0000384 (male)",
    "female": "PATO:0000383 (female)",
}


# ------------------------------------------------------------------------------
# Neuropathology / CLINPATH field normalization
# All tables below were extracted from legacy QC scripts (lee and hardy datasets)
# for re-use in future updated releases.
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.
# Sources: QC_lee_sn_rnaseq.py, hardy_sn_rnaseq.py
# ------------------------------------------------------------------------------

# path_mckeith — Lewy body pathology staging → CDE Validation.
# Case variants (e.g. "L." vs "l.") collapsed after key normalization.

MCKEITH_LB_NORMALIZATION: dict[str, str] = {
    "l. olfactory bulb only":     "Olfactory bulb only",
    "lla. brainstem predominant": "Brainstem",
    "llb. limbic predominant":    "Limbic (transitional)",
    "lv. neocortical":            "Neocortical",
    "lll. brainstem/limbic":      "Amygdala Predominant",
    "0. no lewy bodies":          "Absent",
    "diffuse neocortical":        "Diffuse, neocortical (brainstem, limbic and neocortical involvement)",
    "limbic transitional":        "Limbic (transitional)",
    "present, but extent unknown": "Present but extent unknown",
    "limbic transitional (brainstem and limbic involvement)": "Limbic, transitional (brainstem and limbic involvement)",
}

# path_braak_nft — Braak neurofibrillary tangle stage.
# Numeric strings → Roman numerals as required by CDE Validation.
# Source: hardy_sn_rnaseq.py

BRAAK_NFT_NORMALIZATION: dict[str, str] = {
    "0": "0",
    "1": "I",
    "2": "II",
    "3": "III",
    "4": "IV",
    "5": "V",
    "6": "VI",
}

# path_thal — Thal amyloid phase.
# Source: hardy_sn_rnaseq.py

PATH_THAL_NORMALIZATION: dict[str, str] = {
    "at least 4": "4/5",
}

# path_nia_aa_a/b/c — NIA-AA ABC scoring components.
# Numeric string → letter-prefixed score.
# Sources: hardy_sn_rnaseq.py

NIA_AA_A_NORMALIZATION: dict[str, str] = {
    "0": "A0",
    "1": "A1",
    "2": "A2",
    "3": "A3",
}

NIA_AA_B_NORMALIZATION: dict[str, str] = {
    "0": "B0",
    "1": "B1",
    "2": "B2",
    "3": "B3",
}

NIA_AA_C_NORMALIZATION: dict[str, str] = {
    "0": "C0",
    "1": "C1",
    "2": "C2",
    "3": "C3",
}

# path_nia_ri — NIA Reagan Institute AD diagnosis criterion.
# Case variants collapsed after key normalization.
# Source: QC_lee_sn_rnaseq.py

PATH_NIA_RI_NORMALIZATION: dict[str, str] = {
    "criteria not met": "None",
    "not ad":           "None",
}

# path_ad_level — Alzheimer's disease neuropathological change level.
# Merges entries from lee and hardy.

PATH_AD_LEVEL_NORMALIZATION: dict[str, str] = {
    "no evidence": (
        "No evidence of Alzheimer's disease neuropathological change"
    ),
    "microscopic changes of alzheimer's disease, insufficient for diagnosis": (
        "Low level Alzheimer's disease neuropathological change"
    ),
    "microscopic lesions of alzheimer's disease, insufficient for diagnosis": (
        "Unknown"
    ),
}

# amyloid_angiopathy_severity_scale — cerebral amyloid angiopathy severity.
# Source: QC_lee_sn_rnaseq.py

AMYLOID_ANGIOPATHY_NORMALIZATION: dict[str, str] = {
    "cerebral amyloid angiopathy, temporal and occipital lobe": "Severe",
    "cerebral amyloid angiopathy, frontal lobe":                "Severe",
}

# path_autopsy_dx_main — primary autopsy diagnosis → CDE-compliant label.
# Source: hardy_sn_rnaseq.py

PATH_AUTOPSY_DX_MAIN_NORMALIZATION: dict[str, str] = {
    "parkinson's disease with dementia":    "Parkinson's disease with dementia",
    "parkinson's disease":                  "Parkinson's disease",
    "control brain":                        "Control, no misfolded protein or significant vascular pathology",
    "pathological ageing":                  "Control, no misfolded protein or significant vascular pathology",
    "control brain / path ageing":          "Control, no misfolded protein or significant vascular pathology",
    "argyrophilic grain disease":           "Control, Argyrophilic grain disease",
    "control brain, cerebrovascular disease (small vessel)": (
        "Control, Cerebrovascular disease (atherosclerosis)"
    ),
    "cerebrovascular disease (small vessel)": (
        "Control, Cerebrovascular disease (atherosclerosis)"
    ),
    "control brain, alzheimer`s disease (intermediate level ad pathological change)": (
        "Alzheimer's disease (intermediate level neuropathological change)"
    ),
    "control brain / path ageing, caa": (
        "Control, Cerebrovascular disease (cerebral amyloid angiopathy)"
    ),
}
