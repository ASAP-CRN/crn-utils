"""
CDE controlled-vocabulary tables for the ASAP CRN metadata pipeline.

Centralizes fixed mappings between submitted/raw values and CDE-compliant
Enum values for fields that appear across multiple datasets.  Import these
tables in qc_hook scripts or auxiliary modules like biobank_subject_id.py
to apply consistent normalization logic.

Keys   = submitted / raw values (exactly as they appear in metadata/original files).
Values = CDE-compliant Enum values (exactly as they appear in ValidCategories).

The main goal is to avoid hardcoding the same mappings in multiple places,
and eventually normalize values for Publisher summary statistics and plots.

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
from typing import Callable

__all__ = [

    # Biobank
    "BIOBANK_NAME_NORMALIZATION",
    "BIOBANK_PATTERNS",
    "BIOBANK_DERIVATION_RULES",

    # Brain region
    "BRAIN_REGION_CODES",
    "BRAIN_REGION_TITLES",
    "BRAIN_L2_UBERON",

    # Condition / phenotype
    "CONDITION_LABEL_TO_ID",
    "GP2_PHENOTYPE_NORMALIZATION",
    "PRIMARY_DIAGNOSIS_CORRECTIONS",

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

# ==============================================================================
# MAINTENANCE — update these tables when new datasets are onboarded or CDE
# ValidCategories change.
#
# Keys   = submitted / raw values (exactly as they appear in metadata/original files).
# Values = CDE-compliant Enum values (exactly as they appear in ValidCategories).
#
# Initial sources:
#   - summary_stats.py (_brain_region_coder, _region_titles) — brain region tables
#   - biobank_subject_id.py for release v4.1.1+ — biobank name normalization, patterns, derivation rules
#   - qc_hooks for releases >= v4.0.1 (built by DNAstack curators): gp2_phenotype, sex ontology table
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
    "Banner_Sun_Health_USA": "Banner Sun Health Research Institute",
    "QSBB_UK":               "QSBB UK",
    "Imperial_UK":           "Imperial UK",
    "Edinburgh_UK":          "Edinburgh UK",
}

# CDE Enum value → expected regex for source_subject_id.
# Keys must exactly match SUBJECT.biobank_name ValidCategories.

BIOBANK_PATTERNS: dict[str, str] = {
    "Banner Sun Health Research Institute": r"^\d{2}-\d{2}$",
    "QSBB UK":                             r"^P\d+/\d+$",
    "Cambridge Brain Bank":                r"^(BB\d{2}\.\d{4}|NP\d{2}-\d{5})$",
    "New York Brain Bank":                 r"^T-\d+$",
    "Imperial UK":                         r"^(C\d+|PD\d+|PDC\d+)$",
    "SBB":                                 r"^SBB_Case_\d+$",
    "NKI/NYUGSOM":                         r"^hSDG\d+$",
    "BMC":                                 r"^BMC_Case_\d+$",
    "Edinburgh UK":                        r"^SD\d+/\d+$",
}

# CDE biobank name → callable(subject_id str) → source_subject_id str.
# Add a rule when a deterministic derivation exists (preferred over sibling lookup).
# Banner Sun: BN<MMDD> → MM-DD; Excel reformats month prefixes (MM ≤ 12) as dates.

BIOBANK_DERIVATION_RULES: dict[str, Callable[[str], str]] = {
    "Banner Sun Health Research Institute": lambda sid: f"{sid[2:4]}-{sid[4:6]}",
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

BRAIN_REGION_TITLES: dict[str, str] = {
    "ACG":   "Anterior Cingulate Gyrus",
    "ACC":   "Anterior Cingulate Cortex",
    "AMY":   "Amygdala",
    "CAU":   "Caudate",
    "C_CTX": "Cingulate Cortex",
    "F_CTX": "Frontal Cortex",
    "HC":    "Hippocampus",
    "IPL":   "Inferior Parietal Lobe",
    "MFG":   "Middle Frontal Gyrus",
    "MTG":   "Middle Temporal Gyrus",
    "PARA":  "Para-Hippocampal Gyrus",
    "PFC":   "Prefrontal Cortex",
    "PUT":   "Putamen",
    "P_CTX": "Parietal Cortex",
    "SN":    "Substantia Nigra",
    "T_CTX": "Temporal Cortex",
}

# Maps short code → full CDE >= v4.4 region_level_2 ValidCategory string.
# Used by qc_hooks to populate SAMPLE.region_level_2 from a normalized short code.
# Chain with BRAIN_REGION_CODES: raw value → short code → region_level_2 string.

BRAIN_L2_UBERON: dict[str, str] = {
    "ACC":   "Anterior cingulate cortex (ACC, UBERON:0009835)",
    "ACG":   "Anterior cingulate gyrus (ACG, UBERON:0009838)",
    "AMY":   "Amygdala (AMY, UBERON:0001876)",
    "CAU":   "Caudate nucleus (CAU, UBERON:0001873)",
    "C_CTX": "Cingulate cortex (C_CTX, UBERON:0009836)",
    "F_CTX": "Frontal cortex (F_CTX, UBERON:0001870)",
    "HC":    "Hippocampus (HC, UBERON:0001954)",
    "IPL":   "Inferior parietal lobule (IPL, UBERON:0002810)",
    "MFG":   "Middle frontal gyrus (MFG, UBERON:0002770)",
    "MTG":   "Middle temporal gyrus (MTG, UBERON:0002771)",
    "PARA":  "Parahippocampal area (PARA, UBERON:0002728)",
    "PFC":   "Prefrontal cortex (PFC, UBERON:0001870)",
    "PUT":   "Putamen (PUT, UBERON:0001874)",
    "P_CTX": "Parietal cortex (P_CTX, UBERON:0006091)",
    "SN":    "Substantia nigra (SN, UBERON:0002038)",
    "T_CTX": "Temporal cortex (T_CTX, UBERON:0001875)",
}


# ------------------------------------------------------------------------------
# Condition / phenotype
# ------------------------------------------------------------------------------

# Maps contributor abbreviation → CDE condition_id Enum value.
# Keys are the raw labels submitted in CONDITION.condition or SAMPLE.condition_id
# before QC.  Values match CDE ValidCategories for condition_id.
# Sources: scherzer-pmdbs-lr-wgs, scherzer-pmdbs-sn-rnaseq-midbrain-hybsel.

CONDITION_LABEL_TO_ID: dict[str, str] = {
    "PD":  "PD",
    "HC":  "Control",
    "ILB": "Prodromal",
}

# Maps raw gp2_phenotype value → CDE Enum.
# "Other" is not a CDE ValidCategory; ILB subjects use "Prodromal" consistent
# with their condition_id mapping above.
# Source: scherzer-pmdbs-sn-rnaseq-midbrain-hybsel; TODO confirm ILB→Prodromal
# with data contributor.

GP2_PHENOTYPE_NORMALIZATION: dict[str, str] = {
    "PD":      "PD",
    "Control": "Control",
    "Other":   "Prodromal",
}

# Known contributor typos in primary_diagnosis → corrected CDE-compliant string.
# Source: scherzer-pmdbs-lr-wgs SUBJECT.csv.

PRIMARY_DIAGNOSIS_CORRECTIONS: dict[str, str] = {
    "Other nuerological disorder": "Other neurological disorder",
}


# ------------------------------------------------------------------------------
# Ontology term IDs
# ------------------------------------------------------------------------------

# Maps plain-text sex value → CDE sex_ontology_term_id.
# Source: QC_biederer_pmdbs_spatial_geomx_lamda.py

SEX_ONTOLOGY: dict[str, str] = {
    "Male":   "PATO:0000384 (male)",
    "Female": "PATO:0000383 (female)",
}


# ------------------------------------------------------------------------------
# Neuropathology / CLINPATH field normalization
# All tables below were extracted from legacy QC scripts (lee and hardy datasets)
# for re-use in future updated releases.
# Sources: QC_lee_sn_rnaseq.py, hardy_sn_rnaseq.py
# ------------------------------------------------------------------------------

# path_mckeith — Lewy body pathology staging → CDE ValidCategory.
# Merges entries from both lee (capitalization variants) and hardy (full label
# variants) into a single table.

MCKEITH_LB_NORMALIZATION: dict[str, str] = {
    # Lee dataset variants (mixed capitalization of Roman numeral prefixes)
    "L. Olfactory Bulb-Only":    "Olfactory bulb only",
    "l. Olfactory Bulb-Only":    "Olfactory bulb only",
    "Lla. Brainstem Predominant": "Brainstem",
    "lla. Brainstem Predominant": "Brainstem",
    "Llb. Limbic Predominant":   "Limbic (transitional)",
    "llb. Limbic Predominant":   "Limbic (transitional)",
    "LV. Neocortical":           "Neocortical",
    "lV. Neocortical":           "Neocortical",
    "Lll. Brainstem/Limbic":     "Amygdala Predominant",
    "lll. Brainstem/Limbic":     "Amygdala Predominant",
    "0. No Lewy bodies":         "Absent",
    # Hardy dataset variants (descriptive labels)
    "Diffuse neocortical":       "Diffuse, neocortical (brainstem, limbic and neocortical involvement)",
    "Diffuse Neocortical":       "Diffuse, neocortical (brainstem, limbic and neocortical involvement)",
    "Limbic transitional":       "Limbic (transitional)",
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
    "At least 4": "4/5",
}

# path_nia_aa_a/b/c — NIA-AA ABC scoring components.
# Numeric string → letter-prefixed score.
# Sources: hardy_sn_rnaseq.py

NIA_AA_A_NORMALIZATION: dict[str, str] = {
    "0": "A0", "1": "A1", "2": "A2", "3": "A3",
}

NIA_AA_B_NORMALIZATION: dict[str, str] = {
    "0": "B0", "1": "B1", "2": "B2", "3": "B3",
}

NIA_AA_C_NORMALIZATION: dict[str, str] = {
    "0": "C0", "1": "C1", "2": "C2", "3": "C3",
}

# path_nia_ri — NIA Reagan Institute AD diagnosis criterion.
# Source: QC_lee_sn_rnaseq.py

PATH_NIA_RI_NORMALIZATION: dict[str, str] = {
    "Criteria not met": "None",
    "criteria not met": "None",
    "Not AD":           "None",
}

# path_ad_level — Alzheimer's disease neuropathological change level.
# Merges entries from lee and hardy.

PATH_AD_LEVEL_NORMALIZATION: dict[str, str] = {
    "No evidence": (
        "No evidence of Alzheimer's disease neuropathological change"
    ),
    "Microscopic changes of Alzheimer's disease, insufficient for diagnosis": (
        "Low level Alzheimer's disease neuropathological change"
    ),
    "Microscopic lesions of Alzheimer's disease, insufficient for diagnosis": (
        "Unknown"
    ),
}

# amyloid_angiopathy_severity_scale — cerebral amyloid angiopathy severity.
# Source: QC_lee_sn_rnaseq.py

AMYLOID_ANGIOPATHY_NORMALIZATION: dict[str, str] = {
    "Cerebral amyloid angiopathy, temporal and occipital lobe": "Severe",
    "Cerebral amyloid angiopathy, frontal lobe":                "Severe",
}

# path_autopsy_dx_main — primary autopsy diagnosis → CDE-compliant label.
# Source: hardy_sn_rnaseq.py

PATH_AUTOPSY_DX_MAIN_NORMALIZATION: dict[str, str] = {
    "Parkinson's disease with dementia":    "Parkinson's disease with dementia",
    "Parkinson's disease":                  "Parkinson's disease",
    "Control brain":                        "Control, no misfolded protein or significant vascular pathology",
    "Pathological ageing":                  "Control, no misfolded protein or significant vascular pathology",
    "Control brain / Path ageing":          "Control, no misfolded protein or significant vascular pathology",
    "Argyrophilic grain disease":           "Control, Argyrophilic grain disease",
    "Control brain, Cerebrovascular disease (small vessel)": (
        "Control, Cerebrovascular disease (atherosclerosis)"
    ),
    "Cerebrovascular disease (small vessel)": (
        "Control, Cerebrovascular disease (atherosclerosis)"
    ),
    "Control brain, Alzheimer`s disease (intermediate level AD pathological change)": (
        "Alzheimer's disease (intermediate level neuropathological change)"
    ),
    "Control brain / Path ageing, CAA": (
        "Control, Cerebrovascular disease (cerebral amyloid angiopathy)"
    ),
}
