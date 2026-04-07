# crn-utils

Set of utility python functions and scripts for managing metadata and relase documents for the ASAP CRN cloud. 

The following ASAP repos have crn-utils dependencies:
- [asap-crn-cloud-dataset-metadata](https://github.com/ASAP-CRN/asap-crn-cloud-dataset-metadata)
- [asap-crn-cloud-release-resources, deprecated in release v4.0.1](https://github.com/ASAP-CRN/asap-crn-cloud-release-resources)

## Repo Structure

```
.
├── src/
│   ├── deprecating/                 # Legacy code
│   └── crn_utils/
│       ├── regression_test/         # Code for regression tests between releases
│       ├── asap_ids.py              # ASAP ID generation and management
│       ├── bucket_util.py           # GCP bucket file utilities
│       ├── checksums.py             # File checksum utilities
│       ├── constants.py             # Shared constants
│       ├── doi.py                   # DOI handling
│       ├── file_metadata.py         # File metadata utilities
│       ├── google_spreadsheets.py   # Google Sheets access helpers
│       ├── orphans.py               # Orphaned resource detection
│       ├── proteomics.py            # Proteomics-specific utilities
│       ├── release_util.py          # Release automation and archiving
│       ├── summary_stats.py         # Summary statistics helpers
│       ├── update_schema.py         # CDE schema update helpers
│       ├── util.py                  # General-purpose utilities
│       ├── validate.py              # Metadata validation logic
│       └── zenodo_util.py           # Zenodo integration utilities
├── resource/
│   └── CDE/                         # Local copy of CDE versions
└── .github/
    └── pull_request_template.md
```

## Pull requests
- Template available at: [.github/pull_request_template.md](https://github.com/ASAP-CRN/crn-utils/blob/main/.github/pull_request_template.md)
