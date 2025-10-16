__all__ = [
    "MOUSE_TABLES",
    "PMDBS_TABLES",
    "CELL_TABLES",
    "IPSC_TABLES",
    "PROTEOMICS_TABLES",
]

MOUSE_TABLES = [
    "STUDY",
    "PROTOCOL",
    "ASSAY_RNAseq",
    "SAMPLE",
    "MOUSE",
    "CONDITION",
    "DATA",
]

PMDBS_TABLES = [
    "STUDY",
    "PROTOCOL",
    "SUBJECT",
    "SAMPLE",
    "ASSAY_RNAseq",
    "DATA",
    "PMDBS",
    "CLINPATH",
    "CONDITION",
]


CELL_TABLES = [
    "STUDY",
    "PROTOCOL",
    "ASSAY_RNAseq",
    "SAMPLE",
    "CELL",
    "CONDITION",
    "DATA",
]

IPSC_TABLES = CELL_TABLES.copy()

# TODO: allow for aliasing of CELL, SUBJECT, or MOUSE
PROTEOMICS_TABLES = [
    "STUDY",
    "SAMPLE",
    "CELL",  # optional, could be SUBJECT, or MOUSE
    "PROTEOMICS",
    "CONDITION",
    "DATA",
    "SDRF",
]
