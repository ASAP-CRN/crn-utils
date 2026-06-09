import pandas as pd
from numpy import nan as np_nan

_brain_region_coder = {
    "Anterior_Cingulate_Gyrus": "ACG",
    "Anterior Cingulate Gyrus": "ACG",
    "Caudate": "CAU",
    "Putamen": "PUT",
    "Hippocampus": "HIP",
    "Substantia nigra": "SN",
    "Amygdala": "AMG",
    "Substantia_Nigra ": "SN",
    "Substantia_Nigra": "SN",
    "AMY": "AMG",  # team Jakobsson
    "SND": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNV": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "VTA": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNM": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "SNL": "SN",  # team edwards SN sub-nuclei and adjascent regions
    "Prefrontal Cortex": "PFC",
    "Prefrontal cortex": "PFC",
    "inferior parietal lobe": "IPL",
    "Inferior Parietal Lobe": "IPL",
    "Anterior_Cingulate_Cortex": "ACC",
    "Anterior Cingulate Cortex": "ACC",
    "Antaerior Cortex": "ACC",
    "Antaerior Cingulate": "ACC",
    "Anterior_cingulate_cortex": "ACC",
    "Anterior cingulate cortex": "ACC",
    "Antaerior cortex": "ACC",
    "Antaerior cingulate": "ACC",
    "Frontal Cortex": "F_CTX",
    "Frontal_ctx": "F_CTX",
    "Frontal cortex": "F_CTX",
    "Frontal_Cortex": "F_CTX",
    "frontal_cortex": "F_CTX",
    "Frontal_Lobe": "F_CTX",
    "Frontal lobe": "F_CTX",
    "Parietal Cortex": "P_CTX",
    "Parietal cortex": "P_CTX",
    "Parietal_Cortex": "P_CTX",
    "Parietal lobe": "P_CTX",
    "Parietal_ctx": "P_CTX",
    "Parietal": "P_CTX",
    "Cingulate Cortex": "C_CTX",
    "Cingulate cortex": "C_CTX",
    "Cingulate_Cortex": "C_CTX",
    "Cingulate gyrus": "C_CTX",
    "temporal_ctx": "T_CTX",
    "Temporal Cortex": "T_CTX",
    "Temporal_ctx": "T_CTX",
    "Temporal cortex": "T_CTX",
    "Middle_Frontal_Gyrus": "MFG",
    "Middle frontal gyrus": "MFG",
    "Middle Frontal Gyrus": "MFG",
    "Middle Temporal Gyrus": "MTG",
    "Middle temporal gyrus": "MTG",
    "Parahippocampal Gyrus": "PARA",
    # TODO: The following are for unblocking the v4.1.1 and v5.0.0 release - revisit with update to brain region mapping
    "Midbrain": "MB",
    "Substantia nigra (SN, UBERON:0002038)": "SN",
    "Anterior cingulate gyrus (ACG, UBERON:0002756)": "ACG",
    "Posterior cingulate gyrus (PCG, UBERON:0002740)": "PCG",
    "Frontal lobe (FL, UBERON:0016525)": "FL",
    "Cerebellar hemisphere (CBH, UBERON:0002245)": "CH",
    "Prefrontal cortex (PFC, UBERON:0000451)": "PFC",
    "Putamen (PUT, UBERON:0001874)": "PUT",
    "Amygdala (AMY, UBERON:0001876)": "AMG",
    "Caudate nucleus (CAU, UBERON:0001873)": "CAU",
    "Inferior parietal lobule (IPL, UBERON:0006088)": "IPL",
    "Middle frontal gyrus (MFG, UBERON:0002702)": "MFG",
    "Middle temporal gyrus (MTG, UBERON:0002771)": "MTG",
    "Parahippocampal gyrus (PHG, UBERON:0002973)": "PHG",
    "Parietal lobe (PL, UBERON:0001872)": "PL",
    "Temporal lobe (TL, UBERON:0001871)": "TL",
    "Cingulate gyrus (CG, UBERON:0002967)": "CG",
    "Grey matter (GM, UBERON:0002020)": "GM",
    "Hippocampus (HC, UBERON:0002421)": "HC",
}


_region_titles = {
    "ACG": "Anterior Cingulate Gyrus",
    "CAU": "Caudate Nucleus",
    "PUT": "Putamen",
    "HIP": "Hippocampus",  # Note: HIP kept for legacy usage, HC was used for brain map update
    "HC": "Hippocampus",  
    "SN": "Substantia Nigra",
    "AMG": "Amygdala",
    "PFC": "Prefrontal Cortex",
    "IPL": "Inferior Parietal Lobule",
    "ACC": "Antaerior Cingulate Cortex",
    "F_CTX": "Frontal Cortex",
    "P_CTX": "Parietal Cortex",
    "C_CTX": "Cingulate Cortex",
    "T_CTX": "Temporal Cortex",
    "MFG": "Middle Frontal Gyrus",
    "MTG": "Middle Temporal Gyrus",
    "PARA": "Para-Hippocampal Gyrus",
    # TODO: The following are for unblocking the v4.1.1 and v5.0.0 release - revisit with update to brain region mapping
    "MB": "Midbrain",
    "PCG": "Posterior Cingulate Gyrus",
    "FL": "Frontal Lobe",
    "CH": "Cerebellar Hemisphere",
    "PHG": "Parahippocampal Gyrus",
    "PL": "Parietal Lobe",
    "TL": "Temporal Lobe",
    "CG": "Cingulate Gyrus",
    "GM": "Grey Matter",
}


def get_stats_table(dfs: dict[pd.DataFrame],
                    organism: str,
                    sample_source: str,
                    ) -> None:
    """
    Sorts organism and source to get dataset stats.
    Note: starting release v4.0.1 (CDE v4.1) it's required to define organism and source
          using controlled vocabularies from the CDE Google Spreadsheet tab ValidCategories.
    
    Parameters
    ----------
    dfs
        Dictionary of dataframes containing the metadata tables for a dataset release
    organism
        Organism name (e.g., "Human", "Mouse")
    source
        Sample source (e.g., "Brain")
    """

    if organism == "Human":
        if sample_source == "Brain":
            return get_stat_tabs_pmdbs(dfs)
        elif sample_source in ["Cell lines", "Cell", "iPSC", "InVitro"]:
            return get_stat_tabs_cell(dfs)
        else:
            return get_stat_tabs_human_non_brain(dfs)
        
    elif organism == "Mouse":
        if sample_source == "Brain":
            return get_stat_tabs_mouse(dfs)
        elif sample_source in ["Cell lines", "Cell", "iPSC", "InVitro"]:
            return get_stat_tabs_cell(dfs)
        else:
            return get_stat_tabs_mouse_non_brain(dfs)
    
    else:
        raise ValueError(f"get_stats_table: Unexpected categories: organism {organism}, source {sample_source}")


def make_stats_df_pmdbs(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "age_at_collection",
        "region_level_1",
        "region_level_2",
        "region_level_3",
        "condition_id",
    ]
    subject_cols = ["ASAP_subject_id", "sex"]
    condition_cols = ["condition_id", "intervention_name"]

    SUBJECT_ = dfs["SUBJECT"]
    if "gp2_phenotype" in SUBJECT_.columns:
        subject_cols.append("gp2_phenotype")
        SUBJECT_ = SUBJECT_[subject_cols]
    else:
        SUBJECT_ = SUBJECT_[subject_cols].copy()
        SUBJECT_["gp2_phenotype"] = "NA"

    df = pd.merge(dfs["SAMPLE"][sample_cols], dfs["CONDITION"][condition_cols], on="condition_id", how="left")
    return (
        pd.merge(df, SUBJECT_, on="ASAP_subject_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )


def get_stat_tabs_pmdbs(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_pmdbs(dfs)
    report = get_stats_pmdbs(df)
    return report, df


def get_stats_pmdbs(df: pd.DataFrame) -> dict:
    
    # Helper to select L2 > L1 > L3 for brain region annotation, with fallback to "Unknown"
    # At time of writing, L1 is more informative than L3 for most PMDBS datasets
    def _resolve_region(row):
        for col in ["region_level_2", "region_level_1", "region_level_3"]:
            val = row.get(col, None)
            if val and str(val).lower() not in ("na", "unknown", "unknown_brain_annotation", "nan"):
                return val
        return "Unknown"
    
    n_samples = df[["ASAP_sample_id", "replicate"]].drop_duplicates().shape[0]
    n_subjects = df["ASAP_subject_id"].nunique()
    df["resolved_region"] = df.apply(_resolve_region, axis=1)

    # sample-level
    sw_df = df[["ASAP_sample_id", "ASAP_subject_id", "replicate", "gp2_phenotype",
                "age_at_collection", "resolved_region", "condition_id", "sex"]].drop_duplicates()
    print(f"shape df: {df.shape}, shape sw_df: {sw_df.shape}")

    age_s = sw_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    samples = dict(
        n_samples=n_samples,
        brain_code = sw_df["resolved_region"].replace(_brain_region_coder).value_counts().to_dict(),
        brain_region = sw_df["resolved_region"].replace(_brain_region_coder).map(_region_titles).value_counts().to_dict(),
        PD_status=(sw_df["gp2_phenotype"].value_counts().to_dict(),),
        condition_id=(sw_df["condition_id"].value_counts().to_dict(),),
        sex=(sw_df["sex"].value_counts().to_dict(),),
        age_at_collection=dict(mean=f"{age_s.mean():.1f}", median=f"{age_s.median():.1f}",
                               max=f"{age_s.max():.1f}", min=f"{age_s.min():.1f}"),
    )

    # subject-level
    sub_df = df[["ASAP_subject_id", "gp2_phenotype", "sex", "age_at_collection", "condition_id"]].drop_duplicates(subset=["ASAP_subject_id"])
    age_s = sub_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    subject = dict(
        n_subjects=n_subjects,
        PD_status=(sub_df["gp2_phenotype"].value_counts().to_dict(),),
        condition_id=(sub_df["condition_id"].value_counts().to_dict(),),
        sex=(sub_df["sex"].value_counts().to_dict(),),
        age_at_collection=dict(mean=f"{age_s.mean():.1f}", median=f"{age_s.median():.1f}",
                               max=f"{age_s.max():.1f}", min=f"{age_s.min():.1f}"),
    )

    return dict(subject=subject, samples=samples)


def make_stats_df_human_non_brain(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a flat stats DataFrame for human non-brain datasets (e.g. Blood, Skin,
    Gastrointestinal tissue, etc.) by joining SAMPLE + CONDITION + SUBJECT.

    Unlike make_stats_df_pmdbs there is no PMDBS or CLINPATH join and therefore
    no brain-region columns.  Compatible with both CDE v4.1 (age_at_collection in
    SAMPLE, primary_diagnosis removed) and older schema versions (age_at_collection
    in SUBJECT, primary_diagnosis present).
    """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "condition_id",
    ]

    # replicate_count and assay_type were removed in CDE v4.1; include only if present
    for optional_col in ["replicate_count", "assay_type"]:
        if optional_col in dfs["SAMPLE"].columns:
            sample_cols.append(optional_col)

    subject_cols = [
        "ASAP_subject_id",
        "source_subject_id",
        "biobank_name",
        "sex",
    ]

    # race is Human-specific and required in v4.1 SUBJECT; include if present
    if "race" in dfs["SUBJECT"].columns:
        subject_cols.append("race")

    # primary_diagnosis / primary_diagnosis_text existed pre-v4.1
    for legacy_col in ["primary_diagnosis", "primary_diagnosis_text"]:
        if legacy_col in dfs["SUBJECT"].columns:
            subject_cols.append(legacy_col)

    condition_cols = [
        "condition_id",
        "intervention_name",
    ]

    # age_at_collection: in SUBJECT for pre-v4.1 datasets, in SAMPLE for v4.1+
    if "age_at_collection" in dfs["SUBJECT"].columns:
        subject_cols.append("age_at_collection")
    elif "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")
    else:
        raise ValueError(f"get_stats_human_non_brain: No age_at_collection column found in SUBJECT or SAMPLE")

    SAMPLE_ = dfs["SAMPLE"][sample_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    if "gp2_phenotype" in dfs["SUBJECT"].columns:
        subject_cols.append("gp2_phenotype")
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
    else:
        SUBJECT_ = dfs["SUBJECT"][subject_cols]
        if "primary_diagnosis" in SUBJECT_.columns:
            SUBJECT_ = SUBJECT_.copy()
            SUBJECT_["gp2_phenotype"] = SUBJECT_["primary_diagnosis"]
        else:
            SUBJECT_ = SUBJECT_.copy()
            SUBJECT_["gp2_phenotype"] = "NA"

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")
    df = (
        pd.merge(df, SUBJECT_, on="ASAP_subject_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_human_non_brain(dfs: dict[pd.DataFrame]):
    """
    Compute summary statistics for human non-brain datasets (e.g. Blood, Skin,
    Gastrointestinal tissue, etc.).

    Returns
    -------
    report : dict
        Nested statistics dictionary (same structure as get_stat_tabs_pmdbs but
        without brain_region / brain_code entries).
    df : pd.DataFrame
        Flat stats DataFrame produced by make_stats_df_human_non_brain.
    """
    df = make_stats_df_human_non_brain(dfs)
    report = get_stats_human_non_brain(df)
    return report, df


def get_stats_human_non_brain(df: pd.DataFrame) -> dict:
    """
    Compile summary statistics for human non-brain datasets from a flat stats
    DataFrame (as returned by make_stats_df_human_non_brain).

    The report structure mirrors get_stats_pmdbs but omits brain_region and
    brain_code since those are not applicable outside brain sample sources.
    """
    n_samples = df[["ASAP_sample_id", "replicate"]].drop_duplicates().shape[0]
    n_subjects = df["ASAP_subject_id"].nunique()

    # --- Sample-level stats (one row per unique sample × replicate) ---
    sw_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "replicate",
        "gp2_phenotype",
        "age_at_collection",
        "condition_id",
        "sex",
    ]
    sw_df = df[[c for c in sw_cols if c in df.columns]].drop_duplicates()

    print(f"shape df: {df.shape}, shape sw_df: {sw_df.shape}")

    PD_status = (sw_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sw_df["condition_id"].value_counts().to_dict(),)
    sex = (sw_df["sex"].value_counts().to_dict(),)

    age_at_collection = sw_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    samples = dict(
        n_samples=n_samples,
        PD_status=PD_status,
        condition_id=condition_id,
        age_at_collection=age,
        sex=sex,
    )

    # --- Subject-level stats (one row per unique subject) ---
    sub_cols = [
        "ASAP_subject_id",
        "gp2_phenotype",
        "sex",
        "age_at_collection",
        "condition_id",
    ]
    sub_df = df[[c for c in sub_cols if c in df.columns]].drop_duplicates()

    PD_status = (sub_df["gp2_phenotype"].value_counts().to_dict(),)
    condition_id = (sub_df["condition_id"].value_counts().to_dict(),)
    sex = (sub_df["sex"].value_counts().to_dict(),)

    age_at_collection = sub_df["age_at_collection"].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_at_collection.mean():.1f}",
        median=f"{age_at_collection.median():.1f}",
        max=f"{age_at_collection.max():.1f}",
        min=f"{age_at_collection.min():.1f}",
    )

    subject = dict(
        n_subjects=n_subjects,
        PD_status=PD_status,
        condition_id=condition_id,
        age_at_collection=age,
        sex=sex,
    )

    report = dict(
        subject=subject,
        samples=samples,
    )
    return report


def make_stats_df_cell(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a flat stats DataFrame for cell line / in-vitro datasets.
    
    dfs should be the dict of release metadata
    """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_subject_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "condition_id",
    ]

    subject_cols = [
        "ASAP_subject_id",
        "cell_line"
    ]
    
    condition_cols = [
        "condition_id",
        "intervention_name",
        "intervention_id",
        "protocol_id",
        "intervention_aux_table",
    ]

    SAMPLE_ = dfs["SAMPLE"][sample_cols]
    SUBJECT_ = dfs["SUBJECT"][subject_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")
    df = (
        pd.merge(df, SUBJECT_, on="ASAP_subject_id", how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_cell(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_cell(dfs)
    report = get_stats_cell(df)
    return report, df


def get_stats_cell(df: pd.DataFrame) -> dict:
    """
    get stats for the dataset from the stats table (tab)
    """

    # collapse to remove replicates...
    uq_df = df[
        [
            "ASAP_sample_id",
            "condition_id",
        ]
    ].drop_duplicates()
    print(f"shape df: {df.shape}, shape suq_dfw_df: {uq_df.shape}")

    # get stats for the dataset
    N = uq_df["ASAP_sample_id"].nunique()
    condition_id = (uq_df["condition_id"].value_counts().to_dict(),)
    report = dict(
        N=N,
        condition_id=condition_id,
    )
    return report


def make_stats_df_mouse(dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    """
    Build a flat stats DataFrame for mouse datasets by joining SAMPLE + CONDITION
    + SUBJECT (or the legacy MOUSE table for pre-v4.1 datasets).

    Schema changes in CDE v4.1:
    - MOUSE table removed; subject info (sex, strain) moved to the universal SUBJECT table.
    - ASAP_subject_id is the join key between SAMPLE and SUBJECT in v4.1.
    - ASAP_mouse_id is a secondary identifier now in SAMPLE (not the join key to SUBJECT).
    - age renamed to age_at_collection (now in SAMPLE).
    - replicate_count and assay_type removed from SAMPLE.
    """
    sample_cols = [
        "ASAP_sample_id",
        "ASAP_team_id",
        "ASAP_dataset_id",
        "replicate",
        "repeated_sample",
        "batch",
        "organism",
        "tissue",
        "condition_id",
    ]

    # ASAP_subject_id: join key to universal SUBJECT table in v4.1+
    if "ASAP_subject_id" in dfs["SAMPLE"].columns:
        sample_cols.append("ASAP_subject_id")

    # replicate_count and assay_type were removed in CDE v4.1
    for optional_col in ["replicate_count", "assay_type"]:
        if optional_col in dfs["SAMPLE"].columns:
            sample_cols.append(optional_col)

    # ASAP_mouse_id: secondary ID, include for reference if present
    if "ASAP_mouse_id" in dfs["SAMPLE"].columns:
        sample_cols.append("ASAP_mouse_id")

    # age_at_collection: in SAMPLE for v4.1+; legacy datasets use age in MOUSE
    if "age_at_collection" in dfs["SAMPLE"].columns:
        sample_cols.append("age_at_collection")

    condition_cols = [
        "condition_id",
        "intervention_name",
    ]
    for optional_col in ["intervention_id", "protocol_id", "intervention_aux_table"]:
        if optional_col in dfs["CONDITION"].columns:
            condition_cols.append(optional_col)

    SAMPLE_ = dfs["SAMPLE"][sample_cols]
    CONDITION_ = dfs["CONDITION"][condition_cols]

    df = pd.merge(SAMPLE_, CONDITION_, on="condition_id", how="left")

    # v4.1+: MOUSE table removed, subject info now in universal SUBJECT table.
    # Join key is ASAP_subject_id (present in both SAMPLE and SUBJECT).
    # Pre-v4.1: MOUSE table exists; join key is ASAP_mouse_id.
    if "MOUSE" in dfs:
        subject_df = dfs["MOUSE"]
        subject_join_key = "ASAP_mouse_id"
        subject_cols = ["ASAP_mouse_id"]
        for col in ["sex", "strain", "age"]:
            if col in subject_df.columns:
                subject_cols.append(col)
    else:
        subject_df = dfs["SUBJECT"]
        subject_join_key = "ASAP_subject_id"
        if subject_join_key not in df.columns:
            raise ValueError(f"get_stats_mouse: Cannot join SAMPLE to SUBJECT: {subject_join_key} not found in SAMPLE")
        subject_cols = [subject_join_key]
        for col in ["sex", "strain"]:
            if col in subject_df.columns:
                subject_cols.append(col)

    SUBJECT_ = subject_df[subject_cols]

    df = (
        pd.merge(df, SUBJECT_, on=subject_join_key, how="left")
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df


def get_stat_tabs_mouse(dfs: dict[pd.DataFrame]):
    """ """
    df = make_stats_df_mouse(dfs)
    # get stats for the dataset
    # 0. total number of samples
    report = get_stats_mouse(df)
    return report, df


def get_stat_tabs_mouse_non_brain(dfs: dict[pd.DataFrame]):
    """
    Compute summary statistics for mouse non-brain datasets.

    Mouse datasets do not carry brain-region columns regardless of sample source,
    so this is a direct alias for get_stat_tabs_mouse.
    """
    return get_stat_tabs_mouse(dfs)


def get_stats_mouse(df: pd.DataFrame) -> dict:
    """
    Compile stats from stats table (tab).
    Compatible with CDE v4.1 (age_at_collection in SAMPLE) and pre-v4.1 (age in MOUSE).
    """
    # age field: age_at_collection for v4.1+, age for pre-v4.1
    if "age_at_collection" in df.columns:
        age_col = "age_at_collection"
    elif "age" in df.columns:
        age_col = "age"
    else:
        raise ValueError(f"get_stats_mouse: No age column found in stats DataFrame (expected 'age_at_collection' or 'age')")

    uq_cols = ["ASAP_sample_id", "condition_id", "sex", age_col]
    uq_df = df[[c for c in uq_cols if c in df.columns]].drop_duplicates()

    print(f"shape df: {df.shape}, shape uq_df: {uq_df.shape}")

    age_series = uq_df[age_col].replace({"NA": np_nan}).astype("float")
    age = dict(
        mean=f"{age_series.mean():.1f}",
        median=f"{age_series.median():.1f}",
        max=f"{age_series.max():.1f}",
        min=f"{age_series.min():.1f}",
    )

    N = uq_df["ASAP_sample_id"].nunique()
    condition_id = (uq_df["condition_id"].value_counts().to_dict(),)
    sex = (uq_df["sex"].value_counts().to_dict(),)

    report = dict(
        N=N,
        condition_id=condition_id,
        age=age,
        sex=sex,
    )
    return report


def get_cohort_stats_table(dfs: dict[pd.DataFrame], organism: str, sample_source: str):
    """ """
    datasets = dfs["STUDY"]["ASAP_dataset_id"].unique()
    stat_df = pd.DataFrame()
    
    if organism == "Human" and sample_source == "Brain":
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_pmdbs(dfs_)
            stat_df = pd.concat([stat_df, df])
        report = get_stats_pmdbs(stat_df)
        
    elif sample_source in ["Cell lines", "Cell", "iPSC", "InVitro"]:
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_cell(dfs_)
            stat_df = pd.concat([stat_df, df])
        report = get_stats_cell(stat_df)
    
    elif organism == "Human":  # non-brain
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_human_non_brain(dfs_)
            stat_df = pd.concat([stat_df, df])
        report = get_stats_human_non_brain(stat_df)

    elif sample_source == "Mouse":
        for dataset in datasets:
            dfs_ = {k: v[v["ASAP_dataset_id"] == dataset] for k, v in dfs.items()}
            df = make_stats_df_mouse(dfs_)
            stat_df = pd.concat([stat_df, df])
        report = get_stats_mouse(stat_df)

    else:
        raise ValueError(f"get_cohort_stats_table: Unexpected categories: organism {organism}, source {sample_source}")
    
    N_datasets = stat_df["ASAP_dataset_id"].nunique()
    N_teams = stat_df["ASAP_team_id"].nunique()
    report["N_datasets"] = N_datasets
    report["N_teams"] = N_teams
        
    return report, stat_df
