#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import dataclasses

from collections import defaultdict

from .util import read_CDE, read_meta_table, NULL, export_table

from .constants import PMDBS_TABLES, MOUSE_TABLES

# I want to make a simple class to organize our CDE


@dataclasses.dataclass
class MetadataField:
    name: str
    dtype: str
    required: bool
    description: str
    shared_key: bool | None = None
    asap_assigned_key: bool = False
    values: pd.Series | None = None
    alias: str | None = None
    validation: str | None = None
    
    def __post_init__(self):
        if self.shared_key is None:
            self.shared_key = self.name in ["sample_id", "subject_id", "assay", "file_name", "file_MD5"]
        if self.asap_assigned_key is None:
            self.asap_assigned_key = self.required and self.name not in ["sample_id", "subject_id", "assay", "file_name", "file_MD5"]
        if self.validation is None:
            self.validation = ""
        if self.alias is None:
            self.alias = self.name
        if self.dtype is None:
            self.dtype = "str"
        if self.description is None:
            self.description = ""
        if self.required is None:
            self.required = False
        if self.shared_key is None:
            self.shared_key = False
        if self.asap_assigned_key is None:
            self.asap_assigned_key = False
        if self.values is None:
            self.values = pd.Series(dtype="object")
        if self.alias is None:
            self.alias = self.name
        if self.validation is None:
            self.validation = ""

    def _to_df(self):
        return pd.DataFrame(
            {
                "Field": self.name,
                "DataType": self.dtype,
                "Required": self.required,
                "Description": self.description,
                "Shared_key": self.shared_key,
                "Assigned": self.asap_assigned_key,
                "Validation": self.validation,
            },
            index=[0],
        )


class MetadataTable:
    name: str
    fields: list[MetadataField]
    df: pd.DataFrame
    aux_df: pd.DataFrame
    aux_table: list[MetadataField]
    
    def __init__(self, name: str, df: pd.DataFrame, cde_df: pd.DataFrame):
        self.name = name
        self.df = df
        self.cde_df = cde_df
        self.fields = self.load_fields()
        self.aux_table = []
        self.aux_df = pd.DataFrame()

  

    def load_fields(self):
        fields = []
        for _, row in self.cde_df.iterrows():
            fields.append(
                MetadataField(
                    name=row["Field"],
                    dtype=row["DataType"],
                    required=row["Required"],
                    description=row["Description"],
                    shared_key=row["Shared_key"],
                    asap_assigned_key=row["Required"] == "Assigned",
                    validation=row["Validation"],
                )
            )
        return fields


keys = {
    "STUDY":["NA"],
    "PROTOCOL":["NA"],
    "SUBJECT":["subject_id"],
    "SAMPLE":["sample_id","replicate", "condiition_id"],
    "DATA":["sample_id","replicate","file_name"],
    "PMDBS":["sample_id","subject_id"],
    "CLINPATH":["subject_id"],
    "CONDITION":["condition_id"],
    "MOUSE":["subject_id"],
    "ASSAY_RNAseq":["sample_id"],
    "SPATIAL":["sample_id"],
}




class Schema:
    version: str
    fields: list[MetadataField]
    tables: list[pd.DataFrame]
    table_names: list[str]
    supertable: pd.DataFrame

    def __init__(self, name: str, datafreames: dict[str, pd.DataFrame]:
        self.name = name
        self.dfs = dataframes
        self.tables =dataframes.keys()
        self.fields = self.load_fields(dfs)

    def load_fields(self, dfs):
        fields = []
        for table_name, df in dfs.items():
            for _, row in df.iterrows():
                fields.append(
                    MetadataField(
                        name=row["Field"],
                        dtype=row["DataType"],
                        required=row["Required"],
                        description=row["Description"],
                        shared_key=row["Shared_key"],
                        asap_assigned_key=row["Required"] == "Assigned",
                        validation=row["Validation"],
                    )
                )
        return fields

    @staticmethod
    def load_from_df(name: str, df: pd.DataFrame):
        fields = []
        for _, row in df.iterrows():
            fields.append(
                MetadataField(
                    name=row["Field"],
                    description=row["Description"],
                    dtype=row["DataType"],
                    required=row["Required"],
                    validation=row["Validation"],
                    shared_key=row["Shared_key"] if "Shared_key" in row else None,
                    asap_assigned_key=row["Required"] == "Assigned",
                )
            )
        return Schema(name, fields)

    def to_df(self):
        df = pd.DataFrame(columns=["Field", "DataType", "Required", "Description", "Shared_key", "Assigned", "Validation"])
        for field in self.fields:
            df = pd.concat([df, field._to_df()], ignore_index=True)
        return df

    @property
    def table_names(self):
        return [field.name for field in self.fields if field.shared_key]

    @property
    def supertable(self):
        # merge all the tables...

        return pd.concat([table for table in self.tables], ignore_index=True)
