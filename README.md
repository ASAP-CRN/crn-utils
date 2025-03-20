# crn-utils

Set of utility python functions and scripts for managing metadata and relase documents for the ASAP CRN cloud. 

Should unify the business logic for the following resources and tools:
- [asap-crn-cloud-dataset-metadata] (https://github.com/ASAP-CRN/asap-crn-cloud-dataset-metadata)
- [asap-crn-cloud-release-resources ] (https://github.com/ASAP-CRN/asap-crn-cloud-release-resources) which generates unique asap_ids for the submitted datasets.
- [crn-meta-validate] (https://github.com/ASAP-CRN/crn-meta-validate )which defines the metadata validation [streamlit app] (https://asap-meta-qc.streamlit.app/)
- etc

## metadata versioning and validation
`validate.py`
`checksums.py`
`update_schema.py`

## ASAP ID generation / management
`asap_ids.py`

## release automation and archiving
`resource_tools.py`

## general utilities
`util.py`

## bucket file utilities
`bucket_util.py`

# TODO: make this pip-installable with versioned updates... (currently relying on commit to document provenance)

-------------------------------
