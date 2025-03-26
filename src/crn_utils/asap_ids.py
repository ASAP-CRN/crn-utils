import pandas as pd
import json
# import ijson
from pathlib import Path
import argparse

from .util import read_CDE, read_meta_table, read_CDE_asap_ids, load_tables, export_meta_tables
import shutil

__all__ = ['load_id_mapper', 
            'write_id_mapper', 
            'get_sampr', 
            'get_id', 
            'generate_asap_team_id', 
            'generate_asap_dataset_id', 
            'generate_human_subject_ids', 
            'generate_human_sample_ids', 
            'load_pmdbs_id_mappers', 
            'export_pmdbs_id_mappers', 
            'update_pmdbs_id_mappers',
            'update_pmdbs_meta_tables_with_asap_ids',
            'load_tables',
            'export_meta_tables', 
            'process_meta_files',
            'load_mouse_id_mappers',
            'export_mouse_id_mappers',
            'update_mouse_id_mappers',
            'update_mouse_meta_tables',
            'load_mouse_id_mappers',
            'export_mouse_id_mappers']


    

#####################
# general id utils
#####################
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



# TODO: test this function save the old one before overwriting
def write_id_mapper(id_mapper:dict, id_mapper_path:Path):
    """ write the id mapper to the json file"""
    if id_mapper_path.exists():
        # copy the old file to a backup using datetime to make it unique
        # Get the current date and time

        backup_path = Path(f"{id_mapper_path.parent}/backup/{pd.Timestamp.now().strftime('%Y%m%d')}_{id_mapper_path.name}")
        shutil.copy2(id_mapper_path, backup_path)
        print(f"backed up old id_mapper to {backup_path}")

    if not id_mapper_path.parent.exists():
        id_mapper_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"created directory for id_mapper export at {id_mapper_path.parent}")
    mode = 'w'
    with open(id_mapper_path, mode) as f:
        try:
            json.dump(id_mapper, f, indent=4)
        except TypeError:
            print(f"error writing id_mapper to {id_mapper_path}")
            print(f"check that the id_mapper is a dictionary")
            return 1
        print(f"saved id_mapper to {id_mapper_path}")
    return 0


def get_sampr(v):
    return int(v.split("_")[3].replace("s","")) 

def get_id(v):
    return v[:17] 



def generate_asap_team_id(team:str) -> str:
    """input: team name and output: team_id"""
    return f"TEAM_{team.upper()}"

def generate_asap_dataset_id(dataset_id_mapper:dict, long_dataset_name:str) -> tuple[str,dict]:
    """
    generate new dataset_ids from the long_dataset_name.  Format will be "DS_xxxx"

    long_dataset_name: <team_name>_<source>_<dataset_name> i.e. the folder name in 'asap-crn-metadata'
    (do we actually need to use underscores?)
    """
    source = long_dataset_name.split("-")[1]

    if long_dataset_name in dataset_id_mapper.keys():
        print(f"{long_dataset_name} already has a dataset_id: {dataset_id_mapper[long_dataset_name]}")
        print(f"beware of potential long_dataset_name collision")
        
        return dataset_id_mapper[long_dataset_name], dataset_id_mapper
    else:
        n = len(dataset_id_mapper) + 1
        dataset_id = f"DS_{source.upper()}_{n:04}"
        dataset_id_mapper[long_dataset_name] = dataset_id
        return dataset_id,dataset_id_mapper



#####################
# source typed id utils
#####################



def generate_human_subject_ids(subjectid_mapper:dict,
                             gp2id_mapper:dict,
                             sourceid_mapper:dict, 
                             subject_df:pd.DataFrame,
                             source:str = "pmdbs") -> tuple[dict,dict,dict]:
    """
    generate new unique_ids for new subject_ids in subject_df table, 
    update the id_mapper with the new ids from the data table

    return t
    """

    # force NA to be Nan
    subject_df = subject_df.replace("NA", pd.NA)

    # extract the max value of the mapper's third (last) section ([2] or [-1]) to get our n
    if bool(subjectid_mapper):
        n = max([int(v.split("_")[2]) for v in subjectid_mapper.values() if v]) + 1
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
        if subj_id in set(subjectid_mapper.keys()): # duplicate!!
            # TODO: log this
            # TODO: check for `subject_id` naming collisions with other teams
            asap_subj_id = subjectid_mapper[subj_id]
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
            asap_subject_id = f"ASAP_{source.upper()}_{samp_n:06}"
            # df_dups_subset.insert(0, 'ASAP_subject_id', asap_subject_id, inplace=True)
        else: # testset should have the asap_subj_id
            asap_subject_id = testset.pop() # but where did it come from?
            # print(f"found {subj_id }:{asap_subject_id} in the maps")
        
        src = []
        if add_subj_id:
            # TODO:  instead of just adding we should check if it exists...
            subjectid_mapper[subj_id] = asap_subject_id
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

    return subjectid_mapper, gp2id_mapper, sourceid_mapper

def generate_human_sample_ids(subjectid_mapper:dict, 
                             sampleid_mapper:dict, 
                             sample_df:pd.DataFrame) -> dict:
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
        asap_id = subjectid_mapper[subj_id]

        dups = df_subset[df_subset.duplicated(keep=False, subset=['sample_id'])].sort_values('sample_id').reset_index(drop = True).copy()
        nodups = df_subset[~df_subset.duplicated(keep=False, subset=['sample_id'])].sort_values('sample_id').reset_index(drop = True).copy()
   
        asap_id = subjectid_mapper[subj_id]
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

###############################################
### PMDBS specific functions
###############################################
def load_pmdbs_id_mappers(map_path, suffix):
    source = "PMDBS"

    prototypes = ['dataset', 'subj', 'samp', 'gp2', 'sourcesubj']

    outputs = ()
    for prot in prototypes:
        if prot == 'dataset':
            fname = f"ASAP_{prot}_{suffix}.json"
        else:
            fname = f"ASAP_{source}_{prot}_{suffix}.json"
        
        try:
            id_mapper = load_id_mapper(map_path / fname)
        except FileNotFoundError:
            id_mapper = {}
            print(f"{map_path / fname} not found... starting from scratch")
        outputs += (id_mapper,)

    return outputs


def export_pmdbs_id_mappers(map_path, suffix, datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper):
    source = "PMDBS"

    subject_mapper_path = map_path / f"ASAP_{source}_subj_{suffix}.json"
    sample_mapper_path = map_path / f"ASAP_{source}_samp_{suffix}.json"
    gp2_mapper_path = map_path / f"ASAP_{source}_gp2_{suffix}.json"
    source_mapper_path = map_path / f"ASAP_{source}_sourcesubj_{suffix}.json"
    dataset_mapper_path = map_path / f"ASAP_dataset_{suffix}.json"
    # update the dataset_id_mapper
    write_id_mapper(datasetid_mapper, dataset_mapper_path)
    write_id_mapper(subjectid_mapper, subject_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)
    write_id_mapper(gp2id_mapper, gp2_mapper_path)
    write_id_mapper(sourceid_mapper, source_mapper_path)
    # print(f"wrote updated PMDBS ID mappers")


# isolate generation of the IDs and adding to mappers from updating the tables.
def update_pmdbs_id_mappers(clinpath_df, 
                        sample_df,
                        long_dataset_name,
                        datasetid_mapper,
                        subjectid_mapper,
                        sampleid_mapper,
                        gp2id_mapper,
                        sourceid_mapper):
    """
    read in the CLINPATH and SAMPLE data tables, generate new ids, update the id_mappers

    return updated id_mappers
    """
        
    _, datasetid_mapper = generate_asap_dataset_id(datasetid_mapper, long_dataset_name)
    
    subjec_ids_df = clinpath_df[["subject_id","source_subject_id","GP2_id"]]

    # add ASAP_subject_id to the SUBJECT tables
    output = generate_human_subject_ids(subjectid_mapper,
                                            gp2id_mapper,
                                            sourceid_mapper, 
                                            subjec_ids_df)
    subjectid_mapper, gp2id_mapper,sourceid_mapper = output
    
    sample_ids_df = sample_df[["sample_id","subject_id","source_sample_id"]]
    sampleid_mapper = generate_human_sample_ids(subjectid_mapper, sampleid_mapper, sample_ids_df)

    return datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper


# 
def update_pmdbs_meta_tables_with_asap_ids(dfs:dict, 
                        long_dataset_name:str,
                        asap_ids_schema:pd.DataFrame,
                        datasetid_mapper:dict,
                        subjectid_mapper:dict,
                        sampleid_mapper:dict,
                        gp2id_mapper:dict,
                        sourceid_mapper:dict,
                        pmdbs_tables:list|None = None) -> dict:
    """
    process the metadata tables to add ASAP_IDs to the tables with the mappers

    PMDBS tables:
        ['PMDBS', 'CONDITION', 'CLINPATH', 'SUBJECT', 'ASSAY_RNAseq', 'SAMPLE', 'DATA', 'STUDY', 'PROTOCOL']
    """

    if pmdbs_tables is None:
        pmdbs_tables = ['STUDY', 'PROTOCOL','SUBJECT', 'ASSAY_RNAseq', 'SAMPLE', 'PMDBS', 'CONDITION', 'CLINPATH', 'DATA']

    # pmdbs_tables = ['STUDY', 'PROTOCOL','SUBJECT', 'ASSAY_RNAseq', 'SAMPLE', 'PMDBS', 'CONDITION', 'CLINPATH', 'DATA']
    # pmdbs_tables = asap_ids_schema['Table'].to_list()

    ASAP_sample_id_tables = asap_ids_schema[asap_ids_schema['Field'] == 'ASAP_sample_id']['Table'].to_list()
    ASAP_subject_id_tables = asap_ids_schema[asap_ids_schema['Field'] == 'ASAP_subject_id']['Table'].to_list()
       

    DATASET_ID = datasetid_mapper[long_dataset_name] 

    if 'STUDY' in dfs.keys(): # un-necessary check
        TEAM_ID = dfs['STUDY']['ASAP_team_name'].str.upper().str.replace('-', '_')[0]
    else:  # this should NEVER happen
        print(f"STUDY table not found in dataset {long_dataset_name}")
        TEAM_ID = "TEAM_"+long_dataset_name.split('-')[0].upper()
    
    # now we add the IDs
    for tab in pmdbs_tables:
        if tab not in dfs:
            print(f"Table {tab} not found in dataset {long_dataset_name}")
            continue

        # insert the DATASET_ID at the beginning of the dataframe
        dfs[tab].insert(0, 'ASAP_dataset_id', DATASET_ID)
        # insert the TEAM_ID at the beginning of the dataframe
        dfs[tab].insert(0, 'ASAP_team_id', TEAM_ID)
    
        if tab in ASAP_subject_id_tables:
            # first do the ASAP_subject_id
            ASAP_subject_id = dfs[tab]["subject_id"].map(subjectid_mapper)
            dfs[tab].insert(0, 'ASAP_subject_id', ASAP_subject_id)

        if tab in ASAP_sample_id_tables:
            # second do the ASAP_sample_id
            ASAP_sample_id = dfs[tab]["sample_id"].map(sampleid_mapper)
            dfs[tab].insert(0, 'ASAP_sample_id', ASAP_sample_id)

    return dfs

###############################################
### MOUSE specific functions
###############################################
def load_mouse_id_mappers(map_path, suffix):
    source = "PMDBS"
    pass

def update_mouse_meta_tables():
    pass

def update_mouse_id_mappers(clinpath_df, sample_df,
                        long_dataset_name,
                        datasetid_mapper,
                        sampleid_mapper,):
    pass

def generate_mouse_sample_ids(subjectid_mapper:dict, 
                             sampleid_mapper:dict, 
                             sample_df:pd.DataFrame) -> dict:
    pass

def load_mouse_id_mappers(map_path, suffix):
    source = "MOUSE"
    prototypes = ['dataset', 'samp']

    outputs = ()
    for prot in prototypes:
        if prot == 'dataset':
            fname = f"ASAP_{prot}_{suffix}.json"
        else:
            fname = f"ASAP_{source}_{prot}_{suffix}.json"
        
        try:
            id_mapper = load_id_mapper(map_path / fname)
        except FileNotFoundError:
            id_mapper = {}
            print(f"{map_path / fname} not found... starting from scratch")
            outputs += (id_mapper,)

    return outputs

def export_mouse_id_mappers(map_path, suffix, datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper):
    source = "MOUSE"
    sample_mapper_path = map_path / f"ASAP_{source}_samp_{suffix}.json"
    dataset_mapper_path = map_path / f"ASAP_dataset_{suffix}.json"
    # update the dataset_id_mapper
    write_id_mapper(datasetid_mapper, dataset_mapper_path)
    write_id_mapper(sampleid_mapper, sample_mapper_path)


# isolate generation of the IDs and adding to mappers from updating the tables.
def update_mouse_id_mappers( sample_df,
                        long_dataset_name,
                        datasetid_mapper,
                        sampleid_mapper):
    pass



###############################################
### multi-source wrapper functions WIP
###############################################
def process_meta_files(long_dataset_name,
                        table_path, 
                        asap_ids_schema,
                        map_path, suffix,
                        export_path = None):

    print("unimplimented>...  returning 0")

    # infer source from dataset name
    source = long_dataset_name.split("-")[1]

    if source.upper() == "PMDBS":
        # define tables to process
        pmdbs_tables = ['STUDY', 'PROTOCOL','SUBJECT', 'ASSAY_RNAseq', 'SAMPLE', 'PMDBS', 'CONDITION', 'CLINPATH', 'DATA']

        # load the tables 
        dfs = load_tables(table_path, pmdbs_tables)

        # load the id mappers
        table1 = "CLINPATH"
        table2 = "SAMPLE"
        load_id_mappers = load_pmdbs_id_mappers
        datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper = load_id_mappers(map_path, suffix)

        # update the id mappers 
        update_id_mappers = update_pmdbs_id_mappers
        datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper = update_id_mappers(dfs[table1], dfs[table2], long_dataset_name, datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper)
        # update the tables
        update_meta_tables = update_pmdbs_meta_tables_with_asap_ids
        dfs = update_meta_tables(dfs, long_dataset_name, asap_ids_schema, datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper)

        # export the tables
        if export_path is not None:
            export_meta_tables(dfs, export_path)


        # export the id mappers
        export_id_mappers = export_pmdbs_id_mappers
        export_id_mappers(map_path, suffix, datasetid_mapper, subjectid_mapper, sampleid_mapper, gp2id_mapper, sourceid_mapper)
        

    elif source.upper() == "MOUSE":
        pass

    elif source.upper() == "IPSC":
        pass
    else:
        print(f"source: {source} not recognized assuing human/pmdbs")
        pass


    return 1


#########  script to generate the asap_ids.json file #####################
if __name__ == "__main__":

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="A command-line tool to update tables from ASAP_CDEv1 to ASAP_CDEv2.")
    
    # Add arguments
    parser.add_argument("--dataset", default=".",
                        help="long_dataset_name: <team_name>_<source>_<dataset_name> i.e. the folder name in 'asap-crn-metadata'. Defaults to the current working directory.")
    parser.add_argument("--tables", default=Path.cwd(),
                        help="Path to the directory containing meta TABLES. Defaults to the current working directory.")
    parser.add_argument("--schema", default=Path.cwd(),
                        help="Path to the directory containing ASAP_ID schema.csv. Defaults to the current working directory.")
    parser.add_argument("--map", default=Path.cwd(),
                        help="Path to the directory containing path to mapper.json files. Defaults to the current working directory.")
    parser.add_argument("--suf", default="test",
                        help="suffix to mapper.json. Defaults to 'map' i.e. ASAP_{samp,subj}_map.json")
    parser.add_argument("--outdir", default="v2",
                        help="Path to the directory containing CSD.csv. Defaults to the current working directory.")
    
    # Parse the arguments
    args = parser.parse_args()

    asap_ids_schema = read_CDE_asap_ids(local_path=args.cde)
    table_root = Path(args.tables) 
    export_root= Path(args.outdir)

    process_meta_files( args.dataset,
                        table_root, 
                        asap_ids_schema,
                        args.map, 
                        args.suf,
                        export_path = export_root)