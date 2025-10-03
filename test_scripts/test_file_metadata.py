# %%
import pandas as pd
from pathlib import Path
from crn_utils.util import (
    NULL,
    export_meta_tables,
    load_tables,
)

import pandas as pd
from pathlib import Path
import os
import docx
import json

from crn_utils.asap_ids import *
from crn_utils.validate import validate_table, ReportCollector, process_table

from crn_utils.checksums import extract_md5_from_details2, get_md5_hashes
from crn_utils.file_metadata import (
    gen_raw_bucket_summary,
    update_data_table_with_gcp_uri,
    update_spatial_table_with_gcp_uri,
    gen_spatial_bucket_summary,
    make_file_metadata,
)
from crn_utils.constants import *
from crn_utils.doi import update_study_table_with_doi

from crn_utils.file_metadata import (
    get_artifacts_df,
    get_fastqs_df,
    get_spatial_df,
)

%load_ext autoreload
%autoreload 2
##############
# %%
root_path = Path.home() / ("Projects/ASAP/asap-crn-cloud-dataset-metadata")
datasets_path = root_path / "datasets"

# %%
# ### Starting with v3.1 table
team = "edwards"
dataset_name = "spatial-geomx-th"
source = "pmdbs"
spatial = True

long_dataset_name = f"{team}-{source}-{dataset_name}"

# %%
tables = MOUSE_TABLES if source == "mouse" else PMDBS_TABLES
tables = tables + ["SPATIAL"] if spatial else tables
schema_version = "v3.1"

# %%
ds_path = datasets_path / long_dataset_name

# %%
root_path = Path.home() / ("Projects/ASAP/asap-crn-cloud-dataset-metadata")
datasets_path = root_path / "datasets"
ds_path = datasets_path / long_dataset_name
metadata_path = ds_path / "metadata"

dataset_version = "1.0"

# %%
map_path = root_path / "asap-ids/master"
suffix = "ids"
# %%
dataset_name = ds_path.name
print(f"Processing {ds_path.name}")
ds_parts = dataset_name.split("-")
team = ds_parts[0]
source = ds_parts[1]
short_dataset_name = "-".join(ds_parts[2:])
raw_bucket_name = f"asap-raw-team-{team}-{source}-{short_dataset_name}"
flatten = False

visium = "geomx" not in dataset_name

# %%
# ds_path.mkdir(parents=True, exist_ok=True)
mdata_path = ds_path / "metadata" / "release"
tables = [
    table
    for table in mdata_path.iterdir()
    if table.is_file() and table.suffix == ".csv"
]

req_tables = MOUSE_TABLES if source == "mouse" else PMDBS_TABLES
if spatial:
    req_tables.append("SPATIAL")
table_names = [table.stem for table in tables if table.stem in req_tables]

# %%






# %%

dfs = load_tables(mdata_path, table_names)

file_metadata_path = ds_path / "file_metadata"
# %%
dl_path = file_metadata_path
data_df = dfs["DATA"].copy()


# %%

gen_raw_bucket_summary(
    raw_bucket_name, file_metadata_path, dataset_name, flatten=flatten
)

# %%
gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)
# %%
make_file_metadata(ds_path, file_metadata_path, dfs["DATA"], spatial=spatial)
# %%

dfs["STUDY"] = update_study_table_with_doi(dfs["STUDY"], ds_path)
# %%

dfs["DATA"] = update_data_table_with_gcp_uri(dfs["DATA"], ds_path)
# %%

# gen_spatial_bucket_summary(raw_bucket_name, file_metadata_path, dataset_name)

# %%
dfs["SPATIAL"] = update_spatial_table_with_gcp_uri(
        dfs["SPATIAL"], ds_path, visium=visium
    )

# %%
# %%
# %%
# %%
# %%
# %%
# %%