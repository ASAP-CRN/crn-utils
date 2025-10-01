import pandas as pd
import json

# import ijson
from pathlib import Path
import argparse

from .util import (
    read_CDE,
    read_meta_table,
    read_CDE_asap_ids,
    load_tables,
    export_meta_tables,
)
import shutil

from .constants import *

from .validate import NULL

__all__ = ["prep_proteomics_metadata"]


def prep_proteomics_metadata(
    sdrf_df: pd.DataFrame, schema: pd.DataFrame, tables: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """
    convert a SDRF to the ASAP metadata tables.

    """
    if tables is None:
        tables = PROTEOMICS_TABLES
        # "STUDY",
        # "PROTOCOL",
        # "SAMPLE",
        # "PROTEOMICS",
        # "CONDITION",
        # "DATA",
    dfs = {}
    for table in tables:
        if table == "STUDY":
            df = prep_study_table(sdrf_df, schema)
        # elif table == "PROTOCOL":
        #     df = prep_protocol_table(sdrf_df, schema)
        elif table == "PROTEOMICS":
            df = prep_proteomics_table(sdrf_df, schema)
        elif table == "SAMPLE":
            df = prep_sample_table(sdrf_df, schema)
        # elif table == "CONDITION":
        #     df = prep_condition_table(sdrf_df, schema)
        elif table == "DATA":
            df = prep_data_table(sdrf_df, schema)
        else:
            df = pd.DataFrame(
                columns=schema[schema["Table"] == table]["Field"].tolist()
            )
            print(f"Unknown table {table}.  Creating empty table.")
            # raise ValueError(f"Unknown table {table}")
        dfs[table] = df

    # build CONDITION table from sample table
    condition_df = dfs["CONDITION"]
    condition_df["condition_id"] = sdrf_df[
        "factor value[condition]"
    ].unique()  # dfs["SAMPLE"]["condition_id"].unique()
    condition_df["intervention_name"] = sdrf_df["comment[condition]"].values[0]
    condition_df["intervention_id"] = NULL
    condition_df["protocol_id"] = NULL
    condition_df["intervention_aux_table"] = NULL
    dfs["CONDITION"] = condition_df

    # add SDRF table
    print(f"sdrf_df columns: {sdrf_df.columns}")

    # SDRF_df = prep_sdrf_table(sdrf_df, schema_version)
    dfs["SDRF"] = prep_sdrf_table(sdrf_df)
    print(f"SDRF columns: {dfs['SDRF'].columns}")

    return dfs


def prep_study_table(sdrf_df: pd.DataFrame, schema: pd.DataFrame):
    """
    convert a SDRF to the ASAP metadata tables.

    """
    study_df = pd.DataFrame(
        columns=schema[schema["Table"] == "STUDY"]["Field"].tolist()
    )
    return study_df


# def prep_protocol_table(sdrf_df: pd.DataFrame, schema: pd.DataFrame):
#     """
#     convert a SDRF to the ASAP metadata tables.

#     """
#     return protocol_df


def prep_proteomics_table(sdrf_df: pd.DataFrame, schema: pd.DataFrame):
    """
    convert a SDRF to the ASAP metadata tables.

    """
    proteomics_df = pd.DataFrame(
        columns=schema[schema["Table"] == "PROTEOMICS"]["Field"].tolist()
    )
    proteomics_df["sample_id"] = sdrf_df["source name"]
    proteomics_df["source_id"] = sdrf_df["source name"]
    proteomics_df["technology"] = sdrf_df["technology type"]
    proteomics_df["assay"] = sdrf_df["characteristics[sample type]"]
    proteomics_df["disease"] = sdrf_df["characteristics[disease]"]
    proteomics_df["instrument"] = sdrf_df["comment[instrument]"]
    # proteomics_df["biological_replicate"] = sdrf_df[
    #     "characteristics[biological replicate]"
    # ]
    proteomics_df["technical_replicate"] = sdrf_df["comment[technical replicate]"]
    proteomics_df["raw_file"] = sdrf_df["comment[file uri]"]
    proteomics_df["summary_file"] = NULL  # sdrf_df["comment[quant file]"]
    proteomics_df["SDRF_proteomics_table"] = "SDRF.csv"

    return proteomics_df


def prep_sample_table(sdrf_df: pd.DataFrame, schema: pd.DataFrame):
    """
    convert a SDRF to the ASAP metadata tables.

    """

    sample_df = pd.DataFrame(
        columns=schema[schema["Table"] == "SAMPLE"]["Field"].tolist()
    )

    sample_df["sample_id"] = sdrf_df["source name"]
    sample_df["source_id"] = sdrf_df["assay name"]

    sample_df["organism"] = sdrf_df["characteristics[organism]"]
    sample_df["assay_type"] = "Proteomic"
    # sample_df["organism_ontology_id"] = "NCBITaxon:9606"
    sample_df["age_at_collection"] = sdrf_df["characteristics[age]"]
    sample_df["condition_id"] = sdrf_df["factor value[condition]"]
    sample_df["alternate_id"] = sdrf_df["assay name"]

    #     - _*organism*_:  invalid values 💩'nan'
    #     - valid ➡️ 'Human', 'Mouse', 'Dog', 'Fly', 'Other', 'NA'
    # - _*assay_type*_:  invalid values 💩'nan'
    #     - valid ➡️ 'scRNAseq', 'snRNAseq', 'bulkRNAseq', 'spatialRNAseq', 'ATACseq', 'Proteomic', 'Metabolomic', 'DNAseq', 'WGS', 'NA'
    # - _*organism_ontology_term_id*_:  invalid values 💩'nan'
    #     - valid ➡️ 'NCBITaxon:9606', 'NCBITaxon:10090', 'NA'
    # - _*age_at_collection*_:  invalid values 💩'not available'
    #     - valid ➡️ float or NULL ('NA')
    # - _*sex_ontology_term_id*_:  invalid values 💩'nan'
    #     - valid ➡️ 'PATO:0000384 (male)', 'PATO:0000383 (female)', 'Unknown', 'NA'

    # sample_df["development_stage_ontology_term_id"] = ""
    # sample_df["sex_ontology_term_id"] = ""
    # sample_df["disease_ontology_term_id"] = ""
    # sample_df["tissue_ontology_term_id"] = ""
    # sample_df["assay_ontology_term_id"] = ""
    # sample_df["cell_type_ontology_term_id"] = ""
    technical_replicate = sdrf_df["comment[technical replicate]"]
    biological_replicate = sdrf_df["characteristics[biological replicate]"]

    # tecnical replacites are encode as lower case "rep"{n}
    # biological replicates are encode as upper case "Rep"{n}
    # if we have biological AND technical replicates, use the biological replicates

    sample_df["replicate"] = biological_replicate.where(
        biological_replicate.notnull(), technical_replicate
    )
    # if we have multiple replicates per sample_id, then we need to set replicate_count to > 1 and repeated sample to 1
    sample_df["replicate_count"] = 1
    sample_df["repeated_sample"] = 0
    for sample_id in sample_df["sample_id"].unique():
        if len(sample_df[sample_df["sample_id"] == sample_id]) > 1:
            sample_df.loc[sample_df["sample_id"] == sample_id, "replicate_count"] = len(
                sample_df[data_df["sample_id"] == sample_id]
            )
            sample_df.loc[data_df["sample_id"] == sample_id, "repeated_sample"] = 1

    # set batch to NULL
    sample_df["batch"] = NULL
    return sample_df


def prep_data_table(sdrf_df: pd.DataFrame, schema: pd.DataFrame):
    """
    convert a SDRF to the ASAP metadata tables.

    """
    data_df = pd.DataFrame(columns=schema[schema["Table"] == "DATA"]["Field"].tolist())
    data_df["sample_id"] = sdrf_df["source name"]
    data_df["file_description"] = sdrf_df["comment[proteomics data acquisition method]"]
    data_df["file_type"] = "Raw"
    data_df["file_name"] = sdrf_df["comment[data file]"]

    # - _*file_type*_:  invalid values 💩'Raw'
    #     - valid ➡️ 'fastq', 'Per sample raw file', 'Per sample processed file', 'Combined analysis files', 'annData', 'vHDF', 'plink2', 'VCF', 'csv', 'RDS', 'h5', 'Seurat Object', 'bam', 'cram', 'NA'
    # - _*adjustment*_:  invalid values 💩'nan'
    #     - valid ➡️ 'Raw', 'Processed', 'NA'
    # - _*content*_:  invalid values 💩'nan'
    #     - valid ➡️ 'Counts', 'Probabilities', 'Genotypes', 'Dosages', 'Reads', 'NA'

    technical_replicate = sdrf_df["comment[technical replicate]"]
    biological_replicate = sdrf_df["characteristics[biological replicate]"]

    # tecnical replacites are encode as lower case "rep"{n}
    # biological replicates are encode as upper case "Rep"{n}
    # if we have biological AND technical replicates, use the biological replicates

    data_df["replicate"] = biological_replicate.where(
        biological_replicate.notnull(), technical_replicate
    )
    # if we have multiple replicates per sample_id, then we need to set replicate_count to > 1 and repeated sample to 1
    data_df["replicate_count"] = 1
    data_df["repeated_sample"] = 0
    for sample_id in data_df["sample_id"].unique():
        if len(data_df[data_df["sample_id"] == sample_id]) > 1:
            data_df.loc[data_df["sample_id"] == sample_id, "replicate_count"] = len(
                data_df[data_df["sample_id"] == sample_id]
            )
            data_df.loc[data_df["sample_id"] == sample_id, "repeated_sample"] = 1

    # set batch to NULL
    data_df["batch"] = NULL

    return data_df


def prep_sdrf_table(sdrf_df: pd.DataFrame):
    """
    convert a SDRF to the ASAP metadata tables.

    """
    column_names = [
        "source name",
        "characteristics[organism]",
        "characteristics[organism part]",
        "characteristics[disease]",
        "characteristics[cell type]",
        "characteristics[biological replicate]",
        "characteristics[enrichment process]",
        "characteristics[sex]",
        "characteristics[developmental stage]",
        "characteristics[age]",
        "characteristics[mass]",
        "characteristics[sample type]",
        "characteristics[individual]",
        "characteristics[strain/breed]",
        "characteristics[cell line]",
        "characteristics[ancestry category]",
        "assay name",
        "technology type",
        "comment[proteomics data acquisition method]",
        "comment[label]",
        "comment[instrument]",
        "comment[fraction identifier]",
        "comment[technical replicate]",
        "comment[cleavage agent details]",
        "comment[cleavage agent details].1",
        "comment[modification parameters]",
        "comment[modification parameters].1",
        "comment[dissociation method]",
        "comment[precursor mass tolerance]",
        "comment[fragment mass tolerance]",
        "comment[file uri]",
        "comment[data file]",
        "comment[reduction reagent]",
        "comment[alkylation reagent]",
        "comment[condition]",
        "comment[ms1 scan range]",
        "comment[ms2 analyzer type]",
        "comment[ms2 analyzer type].1",
        "comment[database]",
        "comment[search engine]",
        "comment[quant file]",
    ]
    sdrf_out = pd.DataFrame(columns=column_names)
    for col in column_names:
        if col not in sdrf_df.columns:
            sdrf_df[col] = NULL
        else:
            sdrf_out[col] = sdrf_df[col]

    return sdrf_out
