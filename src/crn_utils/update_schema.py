import os
import logging
import pandas as pd
from pathlib import Path
import argparse
import datetime

# local helpers
from .util import (
    read_CDE, 
    read_meta_table,
    export_table,
    list_expected_metadata_tables,
    NULL,  # constant for missing values
)
    
from .validate import validate_table, create_valid_table
from crn_utils.google_spreadsheets import read_google_sheet

__all__ = [
    "update_tables_v3_0_to_3_2",
    "update_tables_v3_1tov3_2",
    "updated_tables_v3_0tov3_1",
    "update_tables_v3_1tov3_2",
    "reorder_table_to_CDE",
    "filter_table_columns",
    "move_table_columns",
    "v1_to_v2",
    "v2_to_v3_PMDBS",
    "update_metadata_to_version",
]


log = logging.getLogger(__name__)


def get_field_transfer_map() -> pd.DataFrame:
    """
    Dataframe that defines field mappings between old (CDE v3.x) and new (CDE v4.x) schemas.
    Handles two types of transformations:
    - Cross-table move: Field moves from one table to another (Old_Table != New_Table)
    - Same-table rename: Field is renamed within the same table (Old_Table == New_Table)
    
    !!!IMPORTANT!!! 
    Field transfer happens AFTER table consolidation. This means Old_Table and 
    New_Table must refer to POST-CONSOLIDATION table names.
    
    Example workflow for ASSAY_RNAseq.sequencing_instrument -> ASSAY.instrument:
    
    1. BEFORE field transfer (after table consolidation):
       - ASSAY_RNAseq -> ASSAY (via get_table_update_map)
       - Result: ASSAY table has column "sequencing_instrument"
    
    2. Field transfer entry:
       - ["ASSAY", "ASSAY", "sequencing_instrument", "instrument", None]
       - Old_Table = "ASSAY" (NOT "ASSAY_RNAseq" - consolidation already happened!)
       - This is a same-table rename
    
    3. AFTER field transfer:
       - Result: ASSAY table has column "instrument"
    
    Example for SUBJECT.age_at_collection -> SAMPLE.age_at_collection:
    
    1. BEFORE field transfer (after table consolidation):
       - SUBJECT table has column "age_at_collection"
       - SAMPLE table exists
    
    2. Field transfer entry:
       - ["SUBJECT", "SAMPLE", "age_at_collection", "age_at_collection", "subject_id"]
       - This is a cross-table move, so Join_Key is required
    
    3. AFTER field transfer:
       - SUBJECT table no longer has "age_at_collection"
       - SAMPLE table has new column "age_at_collection" (values mapped via subject_id) 
    """
    
    transfers = [
        # Format: [Old_Table, New_Table, Old_Field, New_Field, Join_key]
        # Join_key is required for cross-table moves, None for same-table renames
        
        # Cross-table move
        ["SUBJECT", "SAMPLE", "age_at_collection", "age_at_collection", "subject_id"],
        
        # Same-table renames
        ["STUDY", "STUDY", "project_dataset", "dataset_name", None],
        ["CLINPATH", "CLINPATH", "path_infarcs", "path_infarcts", None],
        ["ASSAY", "ASSAY", "sequencing_instrument", "instrument", None],
        
        # Add more as needed
    ]
    
    transfer_df = pd.DataFrame(transfers, columns=["Old_Table", "New_Table", "Old_Field", "New_Field", "Join_Key"])
    return transfer_df 


def apply_field_moves(
    tables: dict[str, pd.DataFrame],
    transfer_map: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Apply cross-table moves and same-table renames according to transfer_map.
    """
    # Make a copy to avoid mutating the input
    updated_tables = {k: v.copy() for k, v in tables.items()}
    
    for _, row in transfer_map.iterrows():
        old_table = row["Old_Table"]
        new_table = row["New_Table"]
        old_field = row["Old_Field"]
        new_field = row["New_Field"]
        join_key = row.get("Join_Key", None)
        
        # Pass if the table is not part of this dataset
        if old_table not in updated_tables or new_table not in updated_tables:
            continue
        
        src = updated_tables[old_table]
        dst = updated_tables[new_table]
        
        # Pass if the outdated field is not present in the old table
        if old_field not in src.columns:
            continue
        
        # Same-table rename
        if old_table == new_table:
            dst[new_field] = dst[old_field]
            dst = dst.drop(columns=[old_field])
            updated_tables[new_table] = dst
            log.info(f"Renamed '{old_table}.{old_field}' -> '{old_table}.{new_field}'")
            
        # Cross-table move
        else:
            if join_key is None:
                raise ValueError(f"ERROR: Join_Key must be provided for cross-table moves: {old_table}.{old_field} -> {new_table}.{new_field}")
            if join_key not in src.columns or join_key not in dst.columns:
                raise ValueError(f"ERROR: Join key '{join_key}' not found in source or destination table")
            
            # Create a mapping dict and apply
            mapper = dict(zip(src[join_key], src[old_field]))
            dst[new_field] = dst[join_key].map(mapper).fillna(NULL)
            src = src.drop(columns=[old_field])
            updated_tables[new_table] = dst
            updated_tables[old_table] = src
            log.info(f"Moved '{old_table}.{old_field}' -> '{new_table}.{new_field}'")
    
    return updated_tables


def get_table_update_map() -> dict:
    """
    CDE 4.X+ consolidated tables, this maps the old to new table names
    
    IMPORTANT: Table map happens BEFORE field/column map
    """
    table_map = {
        "MOUSE": "SUBJECT",
        "CELL": "SUBJECT",
        "PMDBS": "SAMPLE",
        "ASSAY_RNAseq": "ASSAY",
        "SPATIAL": "ASSAY",
        "PROTEOMICS": "ASSAY",
        # "SDRF": "ASSAY"  # Currently not in CDE 4.1
    }
    
    return table_map


def apply_table_update_map(
    tables: dict[str, pd.DataFrame],
    table_update_map: dict,
    join_key: str = "sample_id",
) -> dict[str, pd.DataFrame]:
    """
    Rename or merge old tables according to table_update_map.
    
    Handles 3 scenarios:
    1. Simple 1:1 rename: (MOUSE -> SUBJECT, SUBJECT doesn't exist)
    2. Merge into existing table: (PMDBS -> SAMPLE, SAMPLE already exists)
    3. Merge two tables into one (SPATIAL + ASSAY_RNAseq -> ASSAY)
    """
    # Initiate with existing tables and track of tables to remove after processing
    updated_tables = {k: v.copy() for k, v in tables.items()}
    remove_tables = set()
    
    # Group old tables by their target new table ("ASSAY": ["SPATIAL", "ASSAY_RNAseq"])
    new_to_old = {}
    for old_table, new_table in table_update_map.items():
        if new_table not in new_to_old:
            new_to_old[new_table] = []
        new_to_old[new_table].append(old_table)
        
    # Process each table, merging or renaming old -> new as needed
    for new_table, old_tables_possible in new_to_old.items():
        
        # Collect the source/old table names that exist in the dataset
        old_tables_exist = [t for t in old_tables_possible if t in tables]
        
        if not old_tables_exist:
            continue

        source_dfs = []  # Holds the actual tables
        for old_table in old_tables_exist:
            source_dfs.append(tables[old_table])
            remove_tables.add(old_table)

        # One source table: either rename or merge into existing
        if len(source_dfs) == 1:  
            old_table = old_tables_exist[0]
            
            # 1:1 rename where new table doesn't exist (MOUSE -> SUBJECT)
            if new_table not in tables:
                updated_tables[new_table] = source_dfs[0].copy()
                log.info(f"Renamed '{old_table}' -> {new_table}'")
                
            # Merge into existing table (PMDBS -> SAMPLE)
            else:
                if join_key not in source_dfs[0].columns:
                    raise ValueError(f"Join key {join_key} not found in {old_table}")
                if join_key not in tables[new_table].columns:
                    raise ValueError(f"ERROR: Join key '{join_key}' not found in {old_table}")
            
                updated_tables[new_table] = pd.merge(
                    updated_tables[new_table], 
                    source_dfs[0], 
                    on=join_key, 
                    how="outer"
                )
                log.info(f"Merged '{old_table}' -> '{new_table}' (joined on '{join_key}')")
            
        # Multiple source tables (SPATIAL + ASSAY_RNAseq -> ASSAY)
        else:
            if join_key not in source_dfs[0].columns:
                raise ValueError(f"Join key {join_key} not found in {old_tables_exist[0]}") 
            if join_key not in source_dfs[1].columns:
                raise ValueError(f"Join key {join_key} not found in {old_tables_exist[1]}") 
            
            updated_tables[new_table] = pd.merge(source_dfs[0], 
                                                 source_dfs[1], 
                                                 on=join_key, 
                                                 how="outer"
                                                )
            log.info(f"Merged '{', '.join(old_tables_exist)}' -> '{new_table}' (joined on '{join_key}')")
            
    # Remove old tables that have been processed
    for old_table in remove_tables:
        if old_table in updated_tables:
            del updated_tables[old_table]
    
    return updated_tables


def update_table_columns(
    table: pd.DataFrame,
    table_name: str,
    old_cde: pd.DataFrame,
    new_cde: pd.DataFrame,
) -> pd.DataFrame:
    """
    For a given metadata table and its associated old CDE, update its columns 
    to match the new CDE.
    Notes on behavior:
    - Columns that are in the old CDE but not the new CDE will be removed
    - Columns that are in the new CDE but not the old CDE will be added as empty
    - Columns that are in neither CDE but are present in the current table will be kept as-is
    """
    # Only get the non-assigned columns for the specific table
    old_cols = old_cde[(old_cde["Table"] == table_name) & (old_cde["Required"] != "Assigned")]["Field"].tolist()
    new_cols = new_cde[(new_cde["Table"] == table_name) & (new_cde["Required"] != "Assigned")]["Field"].tolist()
    
    old_cols_set = set(old_cols)
    new_cols_set = set(new_cols)
    current_cols_set = set(table.columns)
    
    updated_table = table.copy()
    
    # Log columns that are only in the old CDE - these will be removed
    deprecated_cols = old_cols_set - new_cols_set
    deprecated_cols = deprecated_cols.intersection(current_cols_set)
    if deprecated_cols:
        log.info(f"Deprecated columns for {table_name} will be removed: {', '.join(sorted(deprecated_cols))}")
    
    # Add columns that are in the new CDE but not already present after transfer 
    cols_to_add = new_cols_set - current_cols_set
    if cols_to_add:
        log.info(f"New columns for {table_name} will be added: {', '.join(sorted(cols_to_add))}")
        for col in cols_to_add:
            updated_table[col] = NULL
    
    # Log columns that are in current table but neither CDE - these will be kept
    extra_cols = current_cols_set - old_cols_set - new_cols_set
    if extra_cols:
        log.info(f"Extra columns for {table_name} will be kept as-is: {', '.join(sorted(extra_cols))}")
    
    # Preserve column order of old table
    final_cols = [col for col in updated_table.columns if col not in deprecated_cols]
    updated_table = updated_table[final_cols]
    
    return updated_table
    

def update_metadata_to_version(
    tables: dict[str, pd.DataFrame],
    old_cde_version: str,
    new_cde_version: str,
) -> dict[str, pd.DataFrame]:
    """
    Update tables from old CDE version to new CDE version
    
    Workflow:
    1. Load old and new CDE
    2. Apply table merges or renames
    3. Apply field transfers between tables or renames within tables
    4. Update table columns (add missing, remove deprecated)
    5. Filter to only expected tables for given source/modality
    
    IMPORTANT: Table mapping (MOUSE -> SUBJECT) happens BEFORE field transfers/
    renames, so the field transfer map must refer to the final target table names.
    """
    # TODO: replace when read_CDE() updated
    # old_cde = read_CDE(old_cde_version)
    # new_cde = read_CDE(new_cde_version)
    old_cde = read_google_sheet("1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc", tab_name=old_cde_version)
    new_cde = read_google_sheet("1c0z5KvRELdT2AtQAH2Dus8kwAyyLrR0CROhKOjpU4Vc", tab_name=new_cde_version)
    
    # Merge or rename tables according to table update map
    table_update_map = get_table_update_map()
    tables = apply_table_update_map(tables, table_update_map)
    
    # Transfer any fields that have moved between tables or have been renamed
    transfer_map = get_field_transfer_map()
    tables = apply_field_moves(tables, transfer_map)
    
    # Getting the names of tables in the new CDE
    new_cde_tables = set(new_cde["Table"].unique())
    
    # Update columns for all tables that currently exist
    updated_tables = {}
    for table_name in tables.keys():
        if table_name in new_cde_tables:
            updated_tables[table_name] = update_table_columns(
                table=tables[table_name],
                table_name=table_name,
                old_cde=old_cde,
                new_cde=new_cde,
            )
        else:
            raise ValueError(f"Table '{table_name}' not found in new CDE version '{new_cde_version}'")
        
    return updated_tables



# The following code was used to update incremental CDE versions, and is kept
# here for reference and to allow backwards compatibility if needed. Newer
# updates should use update_metadata_to_version() in update_schema.py.
# ------------------------------------------------------------------------------



# def v1_to_v2(tables_path: str|Path, out_dir: str, CDEv1: pd.DataFrame, CDEv2: pd.DataFrame):
#     """
#     load the tables from the tables_path, and update them to the CDEv2 schema.  export the new tables to a datstamped out_dir
#     """
#     import datetime

#     # Get the current date and time
#     current_date = datetime.datetime.now()


#     # Initialize the data types dictionary
#     STUDY = read_meta_table(f"{tables_path}/STUDY.csv")
#     PROTOCOL = read_meta_table(f"{tables_path}/PROTOCOL.csv")
#     SUBJECT = read_meta_table(f"{tables_path}/SUBJECT.csv")
#     CLINPATH = read_meta_table(f"{tables_path}/CLINPATH.csv")
#     SAMPLE = read_meta_table(f"{tables_path}/SAMPLE.csv")

#     # STUDY
#     STUDYv2 = create_valid_table(STUDY, "STUDY", CDEv1)
#     assert len(SAMPLE['preprocessing_references'].unique()) == 1
#     STUDYv2['preprocessing_references'] = SAMPLE['preprocessing_references'][0]
#     STUDYv2['team_dataset_id'] = STUDYv2['project_dataset'].str.replace(" ", "_").str.replace("-", "_")

#     # PROTOCOL
#     PROTOCOLv2 = create_valid_table(PROTOCOL, "PROTOCOL", CDEv1)

#     SAMP_CLIN = SAMPLE.merge(CLINPATH, on="sample_id", how="left")
#     SAMP_CLIN['source_sample_id'] = SAMP_CLIN['source_sample_id_x']
#     SAMP_CLIN = SAMP_CLIN.drop(columns=['source_sample_id_x','source_sample_id_y'])

#     SUBJ_SAMP_CLIN = SUBJECT.merge(SAMP_CLIN, on="subject_id", how="left")


#     SUBJECT_cde_df = CDEv2[CDEv2['Table'] == "SUBJECT"]
#     SUBJECT_cols = SUBJECT_cde_df["Field"].to_list()
#     SUBJECTv2 = SUBJ_SAMP_CLIN[SUBJECT_cols]
#     SUBJECTv2 = SUBJ_SAMP_CLIN[SUBJECT_cols].drop_duplicates(inplace=False).reset_index()

#     CLINPATH_cde_df = CDEv2[CDEv2['Table'] == "CLINPATH"]
#     CLINPATH_cols = CLINPATH_cde_df["Field"].to_list()
#     CLINPATHv2 = SUBJ_SAMP_CLIN[CLINPATH_cols]

#     SAMPLE_cde_df = CDEv2[CDEv2['Table'] == "SAMPLE"]
#     SAMPLE_cols = SAMPLE_cde_df["Field"].to_list()
#     # SAMPLEv2 = SUBJ_SAMP_CLIN[SAMPLE_cols]
#     SAMPLEv2 = SUBJ_SAMP_CLIN[SAMPLE_cols].drop_duplicates(inplace=False).reset_index()

#     DATA_cde_df = CDEv2[CDEv2['Table'] == "DATA"]
#     DATA_cols = DATA_cde_df["Field"].to_list()
#     DATAv2 = SAMPLE[DATA_cols]


#     STUDYv2 = reorder_table_to_CDE(STUDYv2, "STUDY", CDEv2)
#     PROTOCOLv2 = reorder_table_to_CDE(PROTOCOLv2, "PROTOCOL", CDEv2)
#     CLINPATHv2 = reorder_table_to_CDE(CLINPATHv2, "CLINPATH", CDEv2)
#     SAMPLEv2 = reorder_table_to_CDE(SAMPLEv2, "SAMPLE", CDEv2)
#     SUBJECTv2 = reorder_table_to_CDE(SUBJECTv2, "SUBJECT", CDEv2)
#     DATAv2 = reorder_table_to_CDE(DATAv2, "DATA", CDEv2)

#     # Format the date as a string in the format 'YYYYMMDD'
#     date_str = current_date.strftime('%Y%m%d')

#     tables_path = Path(tables_path)

#     export_root = tables_path / f"{out_dir}_{date_str}"
#     os.makedirs(export_root, exist_ok=True)


#     STUDYv2.to_csv( export_root / "STUDY.csv")
#     PROTOCOLv2.to_csv(export_root / "PROTOCOL.csv")
#     SAMPLEv2.to_csv(export_root / "SAMPLE.csv")
#     SUBJECTv2.to_csv(export_root / "SUBJECT.csv")
#     CLINPATHv2.to_csv(export_root / "CLINPATH.csv")
#     DATAv2.to_csv(export_root / "DATA.csv")


#     return STUDYv2, PROTOCOLv2, SAMPLEv2, SUBJECTv2, CLINPATHv2, DATAv2


def v1_to_v2(
    tables_path: str | Path,
    out_dir: str | None,
    CDEv1: pd.DataFrame,
    CDEv2: pd.DataFrame,
    team_dataset_id: str | None = None,
):
    """
    Load the tables from the tables_path, and update them to the CDEv2 schema.
    Export the new tables to a datestamped out_dir.
    """
    # Get the current date and time
    current_date = datetime.datetime.now()

    in_tables = ["STUDY", "PROTOCOL", "SUBJECT", "CLINPATH", "SAMPLE"]
    # in_tables = [table_name for table_name in in_tables if f"{table_name}.csv" in os.listdir(tables_path)]
    metadata_version = "v2.1"
    METADATA_VERSION_DATE = (
        f"{metadata_version}_{pd.Timestamp.now().strftime('%Y%m%d')}"
    )

    # Load the tables
    v1_tables = {}
    aux_tables = {}
    for table_name in in_tables:
        v1_df = read_meta_table(f"{tables_path}/{table_name}.csv")
        v1_tables[table_name], aux_df = create_valid_table(v1_df, table_name, CDEv1)
        if aux_df is not None:
            aux_tables[table_name] = aux_df

    v2_tables = {}

    # STUDY
    # assert len(SAMPLEv1['preprocessing_references'].unique()) == 1
    STUDYv2 = v1_tables["STUDY"].copy()
    STUDYv2["preprocessing_references"] = v1_tables["SAMPLE"][
        "preprocessing_references"
    ][0]
    # force replacement of team_dataset_id
    if team_dataset_id is not None:
        STUDYv2["team_dataset_id"] = team_dataset_id.replace(" ", "_").replace("-", "_")
    else:
        if (
            STUDYv2["team_dataset_id"].isnull().all()
            or STUDYv2["team_dataset_id"] == NULL
        ):
            STUDYv2["team_dataset_id"] = (
                STUDYv2["project_dataset"].str.replace(" ", "_").str.replace("-", "_")
            )

    STUDYv2["metadata_version_date"] = METADATA_VERSION_DATE

    v2_tables["STUDY"] = filter_table_columns(STUDYv2, CDEv2, "STUDY")

    # PROTOCOL
    v2_tables["PROTOCOL"] = v1_tables["PROTOCOL"]

    SAMP_CLIN = pd.merge(
        v1_tables["SAMPLE"], v1_tables["CLINPATH"], on="sample_id", how="left"
    )
    if "sample_id_x" in SAMP_CLIN.columns:
        SAMP_CLIN["sample_id"] = SAMP_CLIN["sample_id_x"]
        SAMP_CLIN = SAMP_CLIN.drop(columns=["sample_id_x", "sample_id_y"])
    if "subject_id_x" in SAMP_CLIN.columns:
        SAMP_CLIN["subject_id"] = SAMP_CLIN["subject_id_x"]
        SAMP_CLIN = SAMP_CLIN.drop(columns=["subject_id_x", "subject_id_y"])
    if "source_sample_id_x" in SAMP_CLIN.columns:
        SAMP_CLIN["source_sample_id"] = SAMP_CLIN["source_sample_id_x"]
        SAMP_CLIN = SAMP_CLIN.drop(columns=["source_sample_id_x", "source_sample_id_y"])
    if "source_subject_id_x" in SAMP_CLIN.columns:
        SAMP_CLIN["source_subject_id"] = SAMP_CLIN["source_subject_id_x"]
        SAMP_CLIN = SAMP_CLIN.drop(
            columns=["source_subject_id_x", "source_subject_id_y"]
        )
    SUBJ_SAMP_CLIN = pd.merge(
        v1_tables["SUBJECT"], SAMP_CLIN, on="subject_id", how="left"
    )

    v2_tables["SUBJECT"] = filter_table_columns(SUBJ_SAMP_CLIN, CDEv2, "SUBJECT")
    v2_tables["CLINPATH"] = filter_table_columns(SUBJ_SAMP_CLIN, CDEv2, "CLINPATH")
    v2_tables["SAMPLE"] = filter_table_columns(SUBJ_SAMP_CLIN, CDEv2, "SAMPLE")

    v2_tables["DATA"] = filter_table_columns(v1_tables["SAMPLE"], CDEv2, "DATA")

    if out_dir is not None:
        # Prepare output directory
        # date_str = current_date.strftime('%Y%m%d')
        os.makedirs(out_dir, exist_ok=True)

        # Export the tables
        for table_name, table in v2_tables.items():
            #  table.to_csv(export_root / f"{table_name}.csv", index=False)
            export_table(table_name, table, export_root)

        for table_name, aux_df in aux_tables.items():
            aux_df.to_csv(os.path.join(export_root, f"{table_name}_aux.csv"), index=False)
            export_table(f"{table_name}_aux", aux_df, export_root)

    return v2_tables, aux_tables


# this is replaced by gp2_phenotype for v3.1
# Update function for CDEv3 (transforming from v2.1 to v3)
def intervention_typer(x):
    control_types = set(
        (
            "Healthy Control",
            "No PD nor other neurological disorder",
            "no pd nor other neurological disorder",
            NULL,
            "HC",
            "Healthy",
            "Healthy",
            "healthy control",
            "Control",
            "control",
            "no_pd_nor_other_neurological_disorder",
            "healthy_control",
        )
    )

    case_types = set(
        (
            "Idiopathic PD",
            "Hemiparkinson/hemiatrophy syndrome",
            "Idiopathic PD",
            "Juvenile autosomal recessive parkinsonism",
            "Motor neuron disease with parkinsonism",
            "Neuroleptic-induced parkinsonism",
            "Psychogenic parkinsonism",
            "Vascular parkinsonism",
            "PD",
            "idiopathic PD",
            "Parkinson's Disease",
            "parkinsons",
            "parkinson's",
            "Parkinson's",
            "Parkinsons",
            "idiopathic pd",
            "hemiparkinson_hemiatrophy_syndrome",
            "pd",
            "idiopathic_pd",
        )
    )
    other_types = set(
        (
            "Frontotemporal dementia",
            "Corticobasal syndrome",
            "Multiple system atrophy",
            "Normal pressure hydrocephalus",
            "Progressive supranuclear palsy",
            "Dementia with Lewy bodies",
            "Dopa-responsive dystonia",
            "Essential tremor",
            "Alzheimer's disease",
            "Spinocerebellar Ataxia (SCA)",
            "Other neurological disorder",
            "other_neurological_disorder",
        )
    )

    prodromal_types = set(
        ("Prodromal non-motor PD", "Prodromal motor PD", "prodromal_motor_pd")
    )

    if x is None:
        return "Control"
    else:
        # x = x.lower()
        if x in control_types:
            return "Control"
        elif x in case_types:
            return "Case"
        elif x in other_types:  # we can assume for now that these are Controls
            return "Control"
        elif x in prodromal_types:  # we can assume for now that these are Controls
            return "Other"
        else:  # default "Other"
            print(f"Unknown intervention type: {x}")
            return "Other"


def v2_to_v3_PMDBS(
    tables_path: str | Path, out_dir: str, CDEv2: pd.DataFrame, CDEv3: pd.DataFrame
):
    """
    Load the tables from the tables_path, and update them to the CDEv3 schema.
    Export the new tables to a datestamped out_dir.
    """
    current_date = datetime.datetime.now()
    v2_meta_tables = ["STUDY", "PROTOCOL", "SUBJECT", "CLINPATH", "SAMPLE", "DATA"]
    v3_meta_tables = [
        "STUDY",
        "PROTOCOL",
        "SUBJECT",
        "SAMPLE",
        "DATA",
        "CLINPATH",
        "PMDBS",
        "CONDITION",
        "ASSAY_RNAseq",
    ]
    metadata_version = "v3.0"

    METADATA_VERSION_DATE = (
        f"{metadata_version}_{pd.Timestamp.now().strftime('%Y%m%d')}"
    )

    # Load the tables
    v2_tables = {}
    aux_tables = {}
    for table_name in v2_meta_tables:
        v2_df = read_meta_table(f"{tables_path}/{table_name}.csv")
        v2_tables[table_name], aux_df = create_valid_table(v2_df, table_name, CDEv2)
        if aux_df is not None:
            aux_tables[table_name] = aux_df

    v3_tables = {}
    # STUDY
    STUDYv3 = v2_tables["STUDY"]  # don't really need to copy here
    STUDYv3["metadata_tables"] = f"{v3_meta_tables}"
    # STUDYv3['number_samples'] = STUDYv2['number_of_brain_samples']
    # STUDYv3['sample_types'] = STUDYv2['brain_regions']
    STUDYv3.rename(
        columns={
            "number_of_brain_samples": "number_samples",
            "brain_regions": "sample_types",
        },
        inplace=True,
    )

    STUDYv3["metadata_version_date"] = METADATA_VERSION_DATE
    # fix ORCID which was misspelled prior to v3
    STUDYv3["PI_ORCID"] = v2_tables["STUDY"]["PI_ORCHID"]

    v3_tables["STUDY"] = filter_table_columns(STUDYv3, CDEv3, "STUDY")

    # PROTOCOL
    v3_tables["PROTOCOL"] = filter_table_columns(
        v2_tables["PROTOCOL"], CDEv3, "PROTOCOL"
    )

    # SUBJECT
    v3_tables["SUBJECT"] = filter_table_columns(v2_tables["SUBJECT"], CDEv3, "SUBJECT")

    # SAMPLE
    v3_tables["SAMPLE"] = filter_table_columns(v2_tables["SAMPLE"], CDEv3, "SAMPLE")

    if "alternate_sample_id" in v2_tables["SAMPLE"].columns:
        v3_tables["SAMPLE"]["alternate_id"] = v2_tables["SAMPLE"]["alternate_sample_id"]
    else:
        v3_tables["SAMPLE"]["alternate_id"] = NULL

    subject_id = v3_tables["SUBJECT"]["subject_id"]
    primary_diagnosis = (
        v3_tables["SUBJECT"]["primary_diagnosis"]
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("'", "")
    )

    diagnosis_mapper = dict(zip(subject_id, primary_diagnosis))
    v3_tables["SAMPLE"]["condition_id"] = v3_tables["SAMPLE"]["subject_id"].map(
        diagnosis_mapper
    )

    # DATA
    v3_tables["DATA"] = filter_table_columns(v2_tables["DATA"], CDEv3, "DATA")

    # PMDBS
    v3_tables["PMDBS"] = filter_table_columns(v2_tables["SAMPLE"], CDEv3, "PMDBS")

    # ASSAY_RNAseq
    ASSAY_RNAseqv3 = filter_table_columns(v2_tables["SAMPLE"], CDEv3, "ASSAY_RNAseq")
    ASSAY_RNAseqv3["technology"] = v2_tables["DATA"]["technology"][0]
    ASSAY_RNAseqv3["omic"] = v2_tables["DATA"]["omic"][0]
    v3_tables["ASSAY_RNAseq"] = ASSAY_RNAseqv3

    # CLINPATH
    CLINPATHv3 = filter_table_columns(v2_tables["CLINPATH"], CDEv3, "CLINPATH")
    # move these to CLINPATHv3 from SUBJECTv2
    cols_from_SUBJECTv2 = [
        "AMPPD_id",
        "GP2_id",
        "ethnicity",
        "family_history",
        "last_diagnosis",
        "age_at_onset",
        "age_at_diagnosis",
        "first_motor_symptom",
        "hx_dementia_mci",
        "hx_melanoma",
        "education_level",
        "smoking_status",
        "smoking_years",
        "APOE_e4_status",
        "cognitive_status",
        "time_from_baseline",
    ]
    CLINPATHv3_cols = [
        col for col in CLINPATHv3.columns if col not in cols_from_SUBJECTv2
    ]
    cols_from_SUBJECTv2 += ["subject_id"]
    v3_tables["CLINPATH"] = pd.merge(
        CLINPATHv3[CLINPATHv3_cols],
        v2_tables["SUBJECT"][cols_from_SUBJECTv2],
        on="subject_id",
        how="left",
    )

    # CONDITION
    # construct this table.  needs to be checked by hand
    CONDITIONv3 = pd.DataFrame(columns=CDEv3[CDEv3["Table"] == "CONDITION"]["Field"])
    CONDITIONv3["condition_id"] = v3_tables["SUBJECT"]["primary_diagnosis"].unique()
    CONDITIONv3["intervention_name"] = "Case-Control"
    CONDITIONv3["intervention_id"] = CONDITIONv3["condition_id"].apply(
        intervention_typer
    )
    CONDITIONv3["condition_id"] = (
        CONDITIONv3["condition_id"]
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("'", "")
    )

    CONDITIONv3 = CONDITIONv3.fillna(NULL)

    v3_tables["CONDITION"] = CONDITIONv3

    if out_dir is not None:
        # Prepare output directory
        # date_str = current_date.strftime('%Y%m%d')
        export_root = Path(out_dir)

        # Export the tables
        for table_name, table in v3_tables.items():
            # table.to_csv(export_root / f"{table_name}.csv", index=False)
            export_table(table_name, table, export_root)
        for table_name, aux_df in aux_tables.items():
            # aux_df.to_csv(export_root / f"{table_name}_aux.csv", index=False)
            export_table(f"{table_name}_aux", aux_df, export_root)

    return v3_tables, aux_tables


def updated_tables_v3_0tov3_1(tables: dict, STUDY_dataset_info: dict):
    """
    Update v3.0 tables to the CDEv3_1 schema.
    NOTE:  all v3.0 submissions are PMDBS
    """

    STUDY = tables["STUDY"].copy()
    SAMPLE = tables["SAMPLE"].copy()
    SUBJECT = tables["SUBJECT"].copy()

    # STUDY changes: remove 'team_dataset_id', and "project_dataset", replace with "dataset_title", "dataset_name", and "project_description"
    STUDY["dataset_title"] = STUDY_dataset_info["dataset_title"]
    STUDY["dataset_description"] = STUDY_dataset_info["dataset_description"]
    STUDY["dataset_name"] = STUDY_dataset_info["dataset_name"]
    if "team_dataset_id" in STUDY.columns:
        STUDY.drop(columns=["team_dataset_id"], inplace=True)
    if "project_dataset" in STUDY.columns:
        STUDY.drop(columns=["project_dataset"], inplace=True)

    # collect 'age_at_collection' from SUBJECT

    if "age_at_collection" in SUBJECT.columns:
        SAMPLE["age_at_collection"] = SAMPLE["subject_id"].map(
            dict(zip(SUBJECT["subject_id"], SUBJECT["age_at_collection"]))
        )
        SUBJECT.drop(columns=["age_at_collection"], inplace=True)

    # infer 'gp2_phenotype'
    gp2_phenotype_mapper = {
        "No PD nor other neurological disorder": "Control",
        "Healthy Control": "Control",
        "Idiopathic PD": "PD",
        "PD": "PD",
        "Prodromal motor PD": "Prodromal",
        "Prodromal Motor PD": "Prodromal",
        # "Other neurological disorder": "Control",  # lee "Other"
        "Hemiparkinson/hemiatrophy syndrome": "PD",  # wood bulk...
        "Alzheimer's disease": "PD",
        "Other neurological disorder": "PD",  # hardy sn
    }

    SUBJECT["gp2_phenotype"] = SUBJECT["primary_diagnosis"].map(gp2_phenotype_mapper)
    # HACK: for team Lee "Other neurological disorder" is actually "Control".  Look in "primary_diagnosis_text" for "Mild Cognitive Impairment"
    SUBJECT.loc[
        (SUBJECT["primary_diagnosis"] == "Other neurological disorder")
        & (SUBJECT["primary_diagnosis_text"] == "Mild Cognitive Impairment"),
        "gp2_phenotype",
    ] = "Control"

    subj_id_condition_id = dict(zip(SUBJECT["subject_id"], SUBJECT["gp2_phenotype"]))
    SAMPLE["condition_id"] = SAMPLE["subject_id"].map(subj_id_condition_id)

    tables["STUDY"] = STUDY
    tables["SUBJECT"] = SUBJECT
    tables["SAMPLE"] = SAMPLE

    return tables


def update_tables_v3_1tov3_2(tables: dict, CDEv3_2: pd.DataFrame):
    """
    Update v3.1 tables to the CDEv3_2 schema.
    NOTE: v3.1 to v3.2 is just adding new tables for new modalities, and expanded Enums... no changes to existing tables

    """
    for table_name, table in tables.items():
        print(f"Updating {table_name}")
        tables[table_name] = reorder_table_to_CDE(table, table_name, CDEv3_2)
    return tables


def update_tables_v3_0_to_3_2(
    tables: dict, STUDY_dataset_info: dict, CDEv3_2: pd.DataFrame | None = None
):
    """
    Update v3.0 tables to the CDEv3_2 schema.
    NOTE:  all v3.0 submissions are PMDBS
    """
    if CDEv3_2 is None:
        CDEv3_2 = read_CDE("v3.2")

    tables = updated_tables_v3_0tov3_1(tables, STUDY_dataset_info)

    # force CONDITION table to use gp2_phenotype
    CONDITION = pd.DataFrame(columns=CDEv3_2[CDEv3_2["Table"] == "CONDITION"]["Field"])
    CONDITION["condition_id"] = tables["SUBJECT"]["gp2_phenotype"].unique()
    CONDITION["intervention_name"] = "Case-Control"
    CONDITION = CONDITION.fillna(NULL)

    tables["CONDITION"] = CONDITION

    tables = update_tables_v3_1tov3_2(tables, CDEv3_2)
    return tables


def reorder_table_to_CDE(table: pd.DataFrame, table_name: str, CDE: pd.DataFrame):
    """
    Reorder the columns of a table to match the schema (CDE).

    Parameters:
    - table (pd.DataFrame): The DataFrame containing the table data.
    - table_name (str): The name of the table (e.g., 'SUBJECT', 'SAMPLE').
    - CDE (pd.DataFrame): The DataFrame containing the schema (CDEv2 or CDEv3).

    Returns:
    - pd.DataFrame: The reordered DataFrame with columns in the schema order.
    """
    schema_cols = CDE[CDE["Table"] == table_name]["Field"].tolist()
    return table[schema_cols]


def filter_table_columns(
    merged_table: pd.DataFrame, CDE: pd.DataFrame, table_name: str
):
    """
    Filter columns of a merged table based on the schema (CDE).

    Parameters:
    - merged_table (pd.DataFrame): The merged DataFrame containing data from multiple tables.
    - CDE (pd.DataFrame): The DataFrame containing the schema (CDEv2 or CDEv3).
    - table_name (str): The name of the table to filter columns for (e.g., 'SUBJECT', 'SAMPLE').

    Returns:
    - pd.DataFrame: The filtered DataFrame with only the columns that match the schema.
    """
    schema_cols = CDE[CDE["Table"] == table_name]["Field"].tolist()

    # add empty columns if they are missing using set differences
    missing_cols = set(schema_cols) - set(merged_table.columns)

    for col in missing_cols:
        merged_table[col] = NULL
    merged_table = merged_table.fillna(NULL)

    return (
        merged_table[schema_cols].drop_duplicates(inplace=False).reset_index(drop=True)
    )


def move_table_columns(
    df_to: pd.DataFrame, df_from: pd.DataFrame, CDE: pd.DataFrame, table_name: str
):
    """
    move columns from one table to another based on the schema (CDE).  assumes same row structure...
    should actually "join" on os.path.join(subject, sample_id) to be careful, but in most cases they are
    """
    schema_cols = CDE[CDE["Table"] == table_name]["Field"].tolist()

    for col in schema_cols:
        if col in df_from.columns:
            df_to[col] = df_from[col]

    return df_to


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="A command-line tool to update tables to a new schema version "
    )

    # Add arguments
    parser.add_argument(
        "--tables",
        default=Path.cwd(),
        help="Path to the directory containing meta TABLES. Defaults to the current working directory.",
    )
    parser.add_argument("--vin", default="v1", help="Input version.  Defaults to v1.")
    parser.add_argument("--vout", default="v3", help="Output version. Defaults to v3.")
    parser.add_argument(
        "--cde_path",
        default=Path.cwd(),
        help="Path to the directory containing the cdes CSD.csv, or Defaults to the current working directory.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # CDE_path = args.cde / "ASAP_CDE_v1.csv"
    # CDEv1 = pd.read_csv( Path.cwd() / "ASAP_CDE_v1.csv" )
    # CDEv2 = pd.read_csv( Path.cwd() / "ASAP_CDE_v2.csv" )

    # if vin is a string, interpret as a
    schema_path = Path(args.cde)

    schema_version = "v2.1"
    CDEv2 = read_CDE(schema_version, local_path=schema_path)
    schema_version = "v3.0"
    CDEv3 = read_CDE(schema_version, local_path=schema_path)

    _ = v2_to_v3(args.tables, CDEv1, CDEv2, args.outdir)

    # Assume that 'old_schema_v1' and 'new_schema_v2_1' are already loaded as DataFrames
    schema_changes_v1_to_v2_1 = summarize_schema_changes(old_schema_v1, new_schema_v2_1)
    schema_changes_v2_1_to_v3 = summarize_schema_changes(new_schema_v2_1, new_schema_v3)

    # Output the summarized changes for v1 to v2.1 and v2.1 to v3
    print("Changes from v1 to v2.1:")
    print(schema_changes_v1_to_v2_1)

    print("\nChanges from v2.1 to v3:")
    print(schema_changes_v2_1_to_v3)


def summarize_schema_changes(old_schema: pd.DataFrame, new_schema: pd.DataFrame):
    """
    Summarizes the changes made between two schema versions.

    Parameters:
    - old_schema (pd.DataFrame): The old schema DataFrame.
    - new_schema (pd.DataFrame): The new schema DataFrame.

    Returns:
    - dict: A summary of the changes.
    """
    summary = {}

    # Group schemas by tables
    old_schema_grouped = old_schema.groupby("Table")
    new_schema_grouped = new_schema.groupby("Table")

    all_tables = set(old_schema["Table"].unique()).union(
        set(new_schema["Table"].unique())
    )

    for table in all_tables:
        old_table_schema = (
            old_schema_grouped.get_group(table)
            if table in old_schema_grouped.groups
            else pd.DataFrame()
        )
        new_table_schema = (
            new_schema_grouped.get_group(table)
            if table in new_schema_grouped.groups
            else pd.DataFrame()
        )

        # Extract fields and required status from each version
        old_fields = set(old_table_schema["Field"].tolist())
        new_fields = set(new_table_schema["Field"].tolist())

        old_required = old_table_schema[old_table_schema["Required"] == "Required"][
            "Field"
        ].tolist()
        new_required = new_table_schema[new_table_schema["Required"] == "Required"][
            "Field"
        ].tolist()

        # Summarize the changes for each table
        moved_fields = {
            field
            for field in old_fields.intersection(new_fields)
            if old_table_schema[old_table_schema["Field"] == field]["Table"].iloc[0]
            != new_table_schema[new_table_schema["Field"] == field]["Table"].iloc[0]
        }

        summary[table] = {
            "added_fields": new_fields - old_fields,
            "deleted_fields": old_fields - new_fields,
            "moved_fields": moved_fields,
            "fields_now_required": set(new_required) - set(old_required),
            "fields_now_optional": set(old_required) - set(new_required),
        }

    return summary


# # Example usage:


if __name__ == "__main__":
    main()
