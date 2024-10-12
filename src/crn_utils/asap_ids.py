import pandas as pd
import json
# import ijson
from pathlib import Path
import argparse


DATASET_ID = "ASAP_PMBDS"
STUDY_PREFIX = f"{DATASET_ID}_"



ASAP_CDE = "ASAP_CDE_v2.csv"

# copy these helper here from utils/io.py
# Function to read a table with the specified data types
def read_meta_table(table_path,dtypes_dict):
    table_df = pd.read_csv(table_path,dtype=dtypes_dict, index_col=0)
    return table_df

# Function to get data types dictionary for a given table
def get_dtypes_dict(cde_df):
    # unnescessary.
    # # Filter the CDE data frame to get the fields and data types for the specified table
    # table_cde = cde_df[cde_df["Table"] == table_name]
    
    if cde_df is None:
        return None
    
    # Initialize the data types dictionary
    dtypes_dict = {}
    
    # Iterate over the rows to fill the dictionary
    for _, row in cde_df.iterrows():
        field_name = row["Field"]
        data_type = row["DataType"]
        
        # Set the data type to string for "String" and "Enum" fields
        if data_type in ["String", "Enum"]:
            dtypes_dict[field_name] = str
    
    return dtypes_dict

def read_CDE(cde_path: str|Path) -> tuple[pd.DataFrame, dict]:
    """Load CDE from local csv and cache it, return a dataframe and dictionary of dtypes"""
    # # Construct the path to CSD.csv
    # # google id for ASAP_CDE sheet
    # GOOGLE_SHEET_ID = "1xjxLftAyD0B8mPuOKUp5cKMKjkcsrp_zr9yuVULBLG8"
    # sheet_name = "ASAP_CDE_v2"
    # cde_url = f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    try:
        CDE_df = pd.read_csv(cde_path)
    except Exception as e:
        print("try local file")
        try:
            CDE_df = pd.read_csv("ASAP_CDE_v2.csv")
        except FileNotFoundError:

            # If the local file doesn't exist, return an error
            print("Error: Both path and local file are not available.")
            CDE_df = None

    dtypes_dict = get_dtypes_dict(CDE_df)
    return CDE_df, dtypes_dict




def load_id_mapper(id_mapper_path:Path) -> dict:
    """ load the id mapper from the json file"""
    id_mapper_path = Path(id_mapper_path)
    if Path.exists(id_mapper_path):
        with open(id_mapper_path, 'r') as f:
            id_mapper = json.load(f)
        print(f"id_mapper loaded from {id_mapper_path}")
    else:
        id_mapper = {}    
        print(f"id_mapper not found at {id_mapper_path}")
    return id_mapper
    

# # don't need the ijson version for now
# def load_big_id_mapper(id_mapper_path:Path, ids:list) -> dict:
#     """ load the id mapper from the json file"""
#     id_mapper = {}

#     if Path.exists(id_mapper_path):
#         with open(id_mapper_path, 'r') as f:
#             for k, v in ijson.kvitems(f, ''):
#                 if k in ids:
#                     id_mapper.update({k:v})    
#         print(f"id_mapper loaded from {id_mapper_path}")
#     else:
#         print(f"id_mapper not found at {id_mapper_path}")
            
#     return id_mapper

def write_id_mapper(id_mapper, id_mapper_path):
    """ write the id mapper to the json file"""
    # if Path.exists(id_mapper_path):
    #     mode = 'a'
    # else:
    #     mode = 'w'
    mode = 'w'
    with open(id_mapper_path, mode) as f:
        json.dump(id_mapper, f, indent=4)

    return 0


def get_sampr(v):
    return int(v.split("_")[3].replace("s","")) 

def get_id(v):
    return v[:17] 


def generate_asap_subject_ids(asapid_mapper:dict,
                             gp2id_mapper:dict,
                             sourceid_mapper:dict, 
                             subject_df:pd.DataFrame) -> tuple[dict,dict,dict,pd.DataFrame,int]:
    """
    generate new unique_ids for new subject_ids in subject_df table, 
    update the id_mapper with the new ids from the data table

    return t
    """

    # extract the max value of the mapper's third (last) section ([2] or [-1]) to get our n
    if bool(asapid_mapper):
        n = max([int(v.split("_")[2]) for v in asapid_mapper.values() if v]) + 1
    else:
        n = 1
    nstart = n

    # ids_df = subject_df[['subject_id','source_subject_id', 'AMPPD_id', 'GP2_id']].copy()
    ids_df = subject_df.copy()

    # might want to use 'source_subject_id' instead of 'subject_id' since we want to find matches across teams
    # shouldn't actually matter but logically cleaner
    uniq_subj = ids_df['subject_id'].unique()
    dupids_mapper = dict(zip(uniq_subj,
                        [num + nstart for num in range(len(uniq_subj))] ))

    n_asap_id_add = 0
    n_gp2_id_add = 0
    n_source_id_add = 0

    df_dup_chunks = []
    id_source = []
    for subj_id, samp_n in dupids_mapper.items():
        df_dups_subset = ids_df[ids_df.subject_id==subj_id].copy()

        # check if gp2_id is known
        # NOTE:  the gp2_id _might_ not be the GP2ID, but instead the GP2sampleID
        #        we might want to check for a trailing _s\d+ and remove it
        #        need to check w/ GP2 team about this.  The RepNo might be sample timepoint... 
        #        and hence be a "subject" in our context
        #    # df['GP2ID'] = df['GP2sampleID'].apply(lambda x: ("_").join(x.split("_")[:-1]))
        #    # df['SampleRepNo'] = df['GP2sampleID'].apply(lambda x: x.split("_")[-1])#.replace("s",""))

        gp2_id = None
        add_gp2_id = False
        # force skipping of null GP2_ids
        if df_dups_subset['GP2_id'].nunique() > 1:
            print(f"subj_id: {subj_id} has multiple gp2_ids: {df_dups_subset['GP2_id'].to_list()}... something is wrong")
            #TODO: log this
        elif not df_dups_subset['GP2_id'].dropna().empty: # we have a valide GP2_id
            gp2_id = df_dups_subset['GP2_id'].values[0] # values because index was not reset

        if gp2_id in set(gp2id_mapper.keys()):
            asap_subj_id_gp2 = gp2id_mapper[gp2_id]
        else:
            add_gp2_id = True
            asap_subj_id_gp2 = None

        # check if source_id is known
        source_id = None
        add_source_id = False
        if df_dups_subset['source_subject_id'].nunique() > 1:
            print(f"subj_id: {subj_id} has multiple source ids: {df_dups_subset['source_subject_id'].to_list()}... something is wrong")
            #TODO: log this
        elif df_dups_subset['source_subject_id'].isnull().any():
            print(f"subj_id: {subj_id} has no source_id... something is wrong")
            #TODO: log this
        else: # we have a valide source_id
            #TODO: check for `source_subject_id` naming collisions with other teams
            #      e.g. check the `biobank_name`
            source_id = df_dups_subset['source_subject_id'].values[0]

        if source_id in set(sourceid_mapper.keys()):
            asap_subj_id_source = sourceid_mapper[source_id]
        else:
            add_source_id = True
            asap_subj_id_source = None

        # TODO: add AMPPD_id test/mapper 

        # check if subj_id is known
        add_subj_id = False
        # check if subj_id (subject_id) is known
        if subj_id in set(asapid_mapper.keys()): # duplicate!!
            # TODO: log this
            # TODO: check for `subject_id` naming collisions with other teams
            asap_subj_id = asapid_mapper[subj_id]
        else:
            add_subj_id = True
            asap_subj_id = None

        # TODO:  improve the logic here so gp2 is the default if it exists.?
        #        we need to check the team_id to make sure it's not a naming collision on subject_id
        #        we need to check the biobank_name to make sure it's not a naming collision on source_subject_id

        testset = set((asap_subj_id, asap_subj_id_gp2, asap_subj_id_source))
        if None in testset:
            testset.remove(None)

        # check that asap_subj_id is not disparate between the maps
        if len(testset) > 1:
            print(f"collission between our ids: {(asap_subj_id, asap_subj_id_gp2, asap_subj_id_source)=}")
            print(f"this is BAAAAD. could be a naming collision with another team on `subject_id` ")

        if len(testset) == 0:  # generate a new asap_subj_id
            # print(samp_n)
            asap_subject_id = f"{STUDY_PREFIX}{samp_n:06}"
            # df_dups_subset.insert(0, 'ASAP_subject_id', asap_subject_id, inplace=True)
        else: # testset should have the asap_subj_id
            asap_subject_id = testset.pop() # but where did it come from?
            # print(f"found {subj_id }:{asap_subject_id} in the maps")
        
        src = []
        if add_subj_id:
            # TODO:  instead of just adding we should check if it exists...
            asapid_mapper[subj_id] = asap_subject_id
            n_asap_id_add += 1
            src.append('asap')

        if add_gp2_id and gp2_id is not None:
            # TODO:  instead of just adding we should check if it exists...
            gp2id_mapper[gp2_id] = asap_subject_id
            n_gp2_id_add += 1
            src.append('gp2')

        if add_source_id and source_id is not None:   
            # TODO:  instead of just adding we should check if it exists...
            sourceid_mapper[source_id] = asap_subject_id
            n_source_id_add += 1
            src.append('source')

        
        df_dup_chunks.append(df_dups_subset)
        id_source.append(src)


    df_dups_wids = pd.concat(df_dup_chunks)
    assert df_dups_wids.sort_index().equals(subject_df)
    print(f"added {n_asap_id_add} new asap_subject_ids")
    print(f"added {n_gp2_id_add} new gp2_ids")
    print(f"added {n_source_id_add} new source_ids")

    # print(id_source)

    return asapid_mapper, gp2id_mapper, sourceid_mapper

def generate_asap_sample_ids(asapid_mapper:dict, 
                             sample_df:pd.DataFrame, 
                             sampleid_mapper:dict) -> tuple[dict, pd.DataFrame]:
    """
    generate new unique_ids for new sample_ids in sample_df table, 
    update the id_mapper with the new ids from the data table

    return the updated id_mapper and updated sample_df
    """

    ud_sampleid_mapper = sampleid_mapper.copy()
    

    uniq_samp = sample_df.sample_id.unique()
    if samp_intersec := set(uniq_samp) & set(ud_sampleid_mapper.keys()): 
        print(f"found {len(samp_intersec)} sample_id's that have already been mapped!! BEWARE a sample_id naming collision!! If you are just reprocessing tables, it shoud be okay.")


    to_map = sample_df[~sample_df['sample_id'].apply(lambda x: x in samp_intersec)].copy()

    if not bool(to_map.shape[0]): 
        print("Nothing to see here... move along... move along .... \nNo new sample_ids to map")
        return ud_sampleid_mapper

    uniq_subj = to_map.subject_id.unique()
    # check for subject_id collisions in the sampleid_mapper
    if subj_intersec := set(uniq_subj) & set(ud_sampleid_mapper.values()): 
        print(f"found {len(subj_intersec)} subject_id collisions in the sampleid_mapper")
        
    df_chunks = []
    for subj_id in uniq_subj:

        df_subset = to_map[to_map.subject_id==subj_id].copy()
        asap_id = asapid_mapper[subj_id]

        dups = df_subset[df_subset.duplicated(keep=False, subset=['sample_id'])].sort_values('sample_id').reset_index(drop = True).copy()
        nodups = df_subset[~df_subset.duplicated(keep=False, subset=['sample_id'])].sort_values('sample_id').reset_index(drop = True).copy()
   
        asap_id = asapid_mapper[subj_id]
        if bool(ud_sampleid_mapper):
            # see if there are any samples already with this asap_id
            sns = [get_sampr(v) for v in ud_sampleid_mapper.values() if get_id(v)==asap_id]
            if len(sns) > 0: 
                rep_n = max(sns) + 1            
            else: 
                rep_n = 1   # start incrimenting from 1        
        else: # empty dicitonary. starting from scratch
            rep_n = 1


        if nodups.shape[0]>0:
            # ASSIGN IDS
            asap_nodups = [f'{asap_id}_s{rep_n+i:03}' for i in range(nodups.shape[0])]
            # nodups['ASAP_sample_id'] = asap_nodups
            nodups.loc[:, 'ASAP_sample_id'] = asap_nodups
            rep_n = rep_n + nodups.shape[0]
            samples_nodups = nodups['sample_id'].unique()

            nodup_mapper = dict(zip(nodups['sample_id'],asap_nodups))

            df_chunks.append(nodups)
        else:
            samples_nodups = []

        if dups.shape[0]>0:
            for dup_id in dups['sample_id'].unique():
                # first peel of any sample_ids that were already named in nodups, 

                if dup_id in samples_nodups:
                    asap_dup = nodup_mapper[dup_id]                    
                else:
                    # then assign ids to the rest.
                    asap_dup = f'{asap_id}_s{rep_n:03}'
                    dups.loc[dups.sample_id==dup_id, 'ASAP_sample_id'] = asap_dup
                    rep_n += 1
            df_chunks.append(dups)


    df_wids = pd.concat(df_chunks)
    id_mapper = dict(zip(df_wids['sample_id'],
                        df_wids['ASAP_sample_id']))

    ud_sampleid_mapper.update(id_mapper)


    # print(ud_sampleid_mapper)
    return ud_sampleid_mapper



def process_meta_files(table_path, 
                        CDE_path, 
                        subject_mapper_path = "ASAP_subj_map.json",
                        sample_mapper_path = "ASAP_samp_map.json",
                        gp2_mapper_path = "ASAP_gp2_map.json",
                        source_mapper_path = "ASAP_source_map.json",
                        export_path = None):
    """
    read in the meta data table, generate new ids, update the id_mapper, write the updated id_mapper to file
    """

    try:
        asapid_mapper = load_id_mapper(subject_mapper_path)
    except FileNotFoundError:
        asapid_mapper = {}
        print(f"{subject_mapper_path} not found... starting from scratch")

    try:
        sampleid_mapper = load_id_mapper(sample_mapper_path)
    except FileNotFoundError:
        sampleid_mapper = {}
        print(f"{sample_mapper_path} not found... starting from scratch")

    try:
        gp2id_mapper = load_id_mapper(gp2_mapper_path)
    except FileNotFoundError:
        gp2id_mapper = {}
        print(f"{gp2_mapper_path} not found... starting from scratch")

    try:
        sourceid_mapper = load_id_mapper(source_mapper_path)
    except FileNotFoundError:
        sourceid_mapper = {}
        print(f"{source_mapper_path} not found... starting from scratch")

    
    CDE, dtypes_dict = read_CDE(CDE_path)
    if CDE is None:
        return 0
    
    # add ASAP_team_id to the STUDY and PROTOCOL tables
    study_path = table_path / "STUDY.csv"
    if study_path.exists():
        study_df = read_meta_table(study_path, dtypes_dict)
        # print(f"before: {study_df['ASAP_team_name'].str.upper()} {[ord(s) for s in study_df['ASAP_team_name'][0].upper().replace('-', '_')]}")
        team_id = study_df['ASAP_team_name'].str.upper().replace('-', '_')  #this isn't actually replacing ...
        # print(f"after: {team_id},  {study_df['ASAP_team_name'].str.upper().replace('-', '_')}  ")

        study_df['ASAP_team_id'] = team_id
        # add ASAP_dataset_id = DATASET_ID to the STUDY tables
        study_df['ASAP_dataset_id'] = DATASET_ID
    else:
        study_df = None
        print(f"{study_path} not found... aborting")
        return 0

    protocol_path = table_path / "PROTOCOL.csv"
    if protocol_path.exists():
        protocol_df = read_meta_table(protocol_path, dtypes_dict)
        protocol_df['ASAP_dataset_id'] = DATASET_ID
    else:
        protocol_df = None
        print(f"{protocol_path} not found... aborting")
        return 0
    
    # add ASAP_subject_id to the SUBJECT tables
    subject_path = table_path / "SUBJECT.csv"
    if subject_path.exists():
        subject_df = read_meta_table(subject_path, dtypes_dict)
        output = generate_asap_subject_ids(asapid_mapper,
                                            gp2id_mapper,
                                            sourceid_mapper, 
                                            subject_df)
        asapid_mapper, gp2id_mapper,sourceid_mapper = output

        ASAP_subject_id = subject_df['subject_id'].map(asapid_mapper)
        subject_df.insert(0, 'ASAP_subject_id', ASAP_subject_id)

        # # add ASAP_dataset_id = DATASET_ID to the SUBJECT tables
        # subject_df['ASAP_dataset_id'] = DATASET_ID
    else:
        subject_df = None
        print(f"{subject_path} not found... aborting")
        return 0
    
    # add ASAP_sample_id and ASAP_dataset_id to the SAMPLE tables
    sample_path = table_path / "SAMPLE.csv"
    if sample_path.exists():
        sample_df = read_meta_table(sample_path, dtypes_dict)
        sampleid_mapper = generate_asap_sample_ids(asapid_mapper, sample_df, sampleid_mapper)
        sample_df['ASAP_dataset_id'] = DATASET_ID

        ASAP_sample_id = sample_df['sample_id'].map(sampleid_mapper)
        sample_df.insert(0, 'ASAP_sample_id', ASAP_sample_id)

    else:
        sample_df = None
        print(f"{sample_path} not found... aborting")
        return 0

    # add ASAP_sample_id to the CLINPATH tables
    clinpath_path = table_path / "CLINPATH.csv"
    if clinpath_path.exists():
        clinpath_df = read_meta_table(clinpath_path, dtypes_dict)
        clinpath_df['ASAP_subject_id'] = clinpath_df['subject_id'].map(asapid_mapper)

    # add ASAP_sample_id to the DATA tables
    data_path = table_path / "DATA.csv"
    if data_path.exists():
        data_df = read_meta_table(data_path, dtypes_dict)
        data_df['ASAP_sample_id'] = data_df['sample_id'].map(sampleid_mapper)


    # export updated tables
    if export_path is not None:

        #HACK: do we want to specify the full export path, or separate by team ID?
        asap_tables_path = export_path / study_df.ASAP_team_id[0]
        print(f"exporting to {asap_tables_path}")
        if  not asap_tables_path.exists():
            asap_tables_path.mkdir()

        if study_path.exists():
            study_df.to_csv(asap_tables_path / study_path.name)
        if protocol_path.exists():
            protocol_df.to_csv(asap_tables_path / protocol_path.name)
        if subject_path.exists():
            subject_df.to_csv(asap_tables_path / subject_path.name)
        if sample_path.exists():
            sample_df.to_csv(asap_tables_path / sample_path.name)
        if clinpath_path.exists():
            clinpath_df.to_csv(asap_tables_path / clinpath_path.name)
        if data_path.exists():
            data_df.to_csv(asap_tables_path / data_path.name)
    else:
        print("no ASAP_tables with ASAP_ID's exported")

    # write the updated id_mapper to file
    print(f"overwriting updated id_mapper to {subject_mapper_path},{sample_mapper_path}, etc.")
    write_id_mapper(asapid_mapper, subject_mapper_path)
    write_id_mapper(sourceid_mapper, source_mapper_path)
    write_id_mapper(gp2id_mapper, gp2_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)

    return 1


#########  script to generate the asap_ids.json file #####################
if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="A command-line tool to update tables from ASAP_CDEv1 to ASAP_CDEv2.")
    
    # Add arguments
    parser.add_argument("--tables", default=Path.cwd(),
                        help="Path to the directory containing meta TABLES. Defaults to the current working directory.")
    parser.add_argument("--cde", default=Path.cwd(),
                        help="Path to the directory containing CSD.csv. Defaults to the current working directory.")
    parser.add_argument("--map", default=Path.cwd(),
                        help="Path to the directory containing path to mapper.json files. Defaults to the current working directory.")
    parser.add_argument("--suf", default="test",
                        help="suffix to mapper.json. Defaults to 'map' i.e. ASAP_{samp,subj}_map.json")
    parser.add_argument("--outdir", default="v2",
                        help="Path to the directory containing CSD.csv. Defaults to the current working directory.")
    
    # Parse the arguments
    args = parser.parse_args()

    CDE_path = Path(args.cde) / ASAP_CDE
   
    subject_mapper_path = Path(args.map) / f"ASAP_subj_{args.suf}.json"
    sample_mapper_path = Path(args.map) / f"ASAP_samp_{args.suf}.json"
    gp2_mapper_path = Path(args.map) / f"ASAP_gp2_{args.suf}.json"
    source_mapper_path = Path(args.map) / f"ASAP_source_{args.suf}.json"

    table_root = Path(args.tables) 
    export_root= Path(args.outdir)
    print(f"exporting to:  {export_root}")
    
    process_meta_files(table_root, 
                        CDE_path, 
                        subject_mapper_path=subject_mapper_path, 
                        sample_mapper_path=sample_mapper_path, 
                        gp2_mapper_path=gp2_mapper_path,
                        source_mapper_path = source_mapper_path,
                        export_path = export_root)
    
    