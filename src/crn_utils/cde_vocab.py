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

Use examples:

1) Map brain_region to region_level_2 in a qc_hook:
    from crn_utils.brain_regions import get_region_level2

    df["region_level_2"] = df["brain_region"].map(get_region_level2)

2) Normalize biobank name and fix source_subject_id in a qc_hook:
      from crn_utils.biobank_subject_id import BiobankSubjectIdFixer

      fixer = BiobankSubjectIdFixer(dataset_name=dataset_name, caller_path=__file__)
      meta_tables["SUBJECT"] = fixer.fix(meta_tables["SUBJECT"])

      # Note: users don't need to import the BIOBANK_* tables in the hook.
      # BiobankSubjectIdFixer picks them up automatically.

"""
import re
from typing import Callable

__all__ = [

    # Generic normalization
    "normalize_vocab_key",

    # Biobank
    "BIOBANK_NAME_NORMALIZATION",
    "BIOBANK_PATTERNS",
    "BIOBANK_DERIVATION_RULES",

    # Brain region
    "BRAIN_REGION_CODES",
    "BRAIN_REGION_TITLES",
    "BRAIN_L2_UBERON",

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
# MAINTENANCE — update these tables when new datasets are onboarded or CDE
# ValidCategories change.
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
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() to
# normalize the raw biobank_name before lookup.
# Add an entry whenever a contributor uses a non-standard biobank_name spelling.

BIOBANK_NAME_NORMALIZATION: dict[str, str] = {
    "banner sun health usa": "Banner Sun Health Research Institute",
    "qsbb uk":               "QSBB UK",
    "imperial uk":           "Imperial UK",
    "edinburgh uk":          "Edinburgh UK",
}

# CDE Enum value → expected regex for source_subject_id.
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.

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
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.

BIOBANK_DERIVATION_RULES: dict[str, Callable[[str], str]] = {
    "banner sun health research institute": lambda sid: f"{sid[2:4]}-{sid[4:6]}",
}


# ------------------------------------------------------------------------------
# Brain region
#
# BRAIN_REGION_CODES keys are normalized: lowercase, spaces (no underscores or
# hyphens). Do NOT add case or separator variants here — they are handled at
# lookup time by brain_regions.normalize_brain_region_key().  Only add a new
# entry when the submitted string maps to a region not covered by normalized keys.
#
# Brain mapping chain for region_level_2 (final CDE >= v4.4 compliant) for qc_hooks output:
# Step  Transform                    Input                       Output
# ----  ---------------------------  --------------------------  -----------------------------------------------
# 1     normalize_brain_region_key   "Frontal_Cortex"            "frontal cortex"
# 2     BRAIN_REGION_CODES           "frontal cortex"            "F_CTX"
# 3     BRAIN_L2_UBERON              "F_CTX"                     "Frontal cortex (F_CTX, UBERON:0001870)"
# Note: different dataset qc_hooks may enter this chain at different steps,
#       that's why we expose the individual functions.
# ------------------------------------------------------------------------------

BRAIN_REGION_CODES: dict[str, str] = {
    # canonical regions (already lowercase after normalization)
    "anterior cingulate gyrus":   "ACG",
    "caudate":                    "CAU",
    "putamen":                    "PUT",
    "hippocampus":                "HC",
    "substantia nigra":           "SN",
    "amygdala":                   "AMY",
    "parietal":                   "P_CTX",
    "prefrontal cortex":          "PFC",
    "inferior parietal lobe":     "IPL",
    "anterior cingulate cortex":  "ACC",
    "antaerior cortex":           "ACC",   # typo: "Antaerior" → "Anterior"
    "antaerior cingulate":        "ACC",   # typo: "Antaerior" → "Anterior"
    "frontal cortex":             "F_CTX",
    "frontal ctx":                "F_CTX",
    "frontal lobe":               "F_CTX",
    "parietal cortex":            "P_CTX",
    "parietal ctx":               "P_CTX",
    "parietal lobe":              "P_CTX",
    "cingulate cortex":           "C_CTX",
    "cingulate gyrus":            "C_CTX",
    "temporal cortex":            "T_CTX",
    "temporal ctx":               "T_CTX",
    "middle frontal gyrus":       "MFG",
    "middle temporal gyrus":      "MTG",
    "parahippocampal gyrus":      "PARA",
    "posterior cingulate gyrus": "PCG",

    # team-specific short codes (already lowercase after normalization)
    "amy":  "AMY",  # team Jakobsson
    "snd":  "SN",   # team Edwards: SN sub-nucleus
    "snv":  "SN",   # team Edwards: SN sub-nucleus
    "vta":  "SN",   # team Edwards: SN-adjacent
    "snm":  "SN",   # team Edwards: SN sub-nucleus
    "snl":  "SN",   # team Edwards: SN sub-nucleus

}

# Maps short code → canonical display name (inverse of BRAIN_REGION_CODES).
# Used to label plots and reports with human-readable region names.
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.
# Note: BRAIN_REGION_TITLES is kept for legacy purposes (releases <= v4.0.0).
#       Releases > v4.0.0 should use the BRAIN_L2_UBERON mapping

BRAIN_REGION_TITLES: dict[str, str] = {
    "acg":   "Anterior Cingulate Gyrus",
    "acc":   "Anterior Cingulate Cortex",
    "amy":   "Amygdala",
    "cau":   "Caudate",
    "c ctx": "Cingulate Cortex",
    "f ctx": "Frontal Cortex",
    "hc":    "Hippocampus",
    "ipl":   "Inferior Parietal Lobe",
    "mfg":   "Middle Frontal Gyrus",
    "mtg":   "Middle Temporal Gyrus",
    "para":  "Para-Hippocampal Gyrus",
    "pfc":   "Prefrontal Cortex",
    "put":   "Putamen",
    "p ctx": "Parietal Cortex",
    "sn":    "Substantia Nigra",
    "t ctx": "Temporal Cortex",
}

# Maps short code → full CDE >= v4.4 region_level_2 ValidCategory string.
# Used by qc_hooks to populate SAMPLE.region_level_2 from a normalized short code.
# Chain with BRAIN_REGION_CODES: raw value → short code → region_level_2 string.
# Keys are pre-normalized (lowercase, spaces). Use normalize_vocab_key() before lookup.

BRAIN_L2_UBERON: dict[str, str] = {
    "acc":   "Anterior cingulate cortex (ACC, UBERON:0009835)",
    "acg":   "Anterior cingulate gyrus (ACG, UBERON:0009838)",
    "amy":   "Amygdala (AMY, UBERON:0001876)",
    "cau":   "Caudate nucleus (CAU, UBERON:0001873)",
    "c ctx": "Cingulate cortex (C_CTX, UBERON:0009836)",
    "f ctx": "Frontal cortex (F_CTX, UBERON:0001870)",
    "hc":    "Hippocampus (HC, UBERON:0001954)",
    "ipl":   "Inferior parietal lobule (IPL, UBERON:0002810)",
    "mfg":   "Middle frontal gyrus (MFG, UBERON:0002770)",
    "mtg":   "Middle temporal gyrus (MTG, UBERON:0002771)",
    "para":  "Parahippocampal area (PARA, UBERON:0002728)",
    "pfc":   "Prefrontal cortex (PFC, UBERON:0001870)",
    "put":   "Putamen (PUT, UBERON:0001874)",
    "p ctx": "Parietal cortex (P_CTX, UBERON:0006091)",
    "sn":    "Substantia nigra (SN, UBERON:0002038)",
    "t ctx": "Temporal cortex (T_CTX, UBERON:0001875)",
    "pcg": "Posterior cingulate gyrus (PCG, UBERON:0002740)",
}


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

# path_mckeith — Lewy body pathology staging → CDE ValidCategory.
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
}

# path_braak_nft — Braak neurofibrillary tangle stage.
# Numeric strings → Roman numerals as required by CDE ValidCategories.
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
