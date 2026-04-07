# %%
import pandas as pd
from pathlib import Path

from crn_utils.constants import *

from crn_utils.util import *

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
tables = MOUSE_TABLES.copy() if source == "mouse" else PMDBS_TABLES.copy()
tables = tables.copy() + ["SPATIAL"] if spatial else tables

schema_version = "v3.2"

# %%

CDE_df = read_CDE(metadata_version=schema_version, include_aliases=True, include_asap_ids=True)





# %%

archive_CDE(schema_version, "resource/CDE")


# %%
schema_version = "v3.3"
archive_CDE(schema_version, "resource/CDE")


# %%
