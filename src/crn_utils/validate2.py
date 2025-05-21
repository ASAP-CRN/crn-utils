import pandas as pd
from pathlib import Path
import dataclasses

from collections import defaultdict

from util import read_CDE, read_meta_table, NULL, export_table


## make an asap_crn_schema class using dataclasses
# resource/CDE/ASAP_CDE_v3.1.csv will be contained within this class
# it will have methods to validate tables against the schema
# it will have methods to update the schema
# it will have methods to compare itself to other schemas
# Placeholder for ASAP_CRN_Schema class implementation
class ASAP_CRN_Schema:
    def __init__(self, cde_path):
        self.cde_df = pd.read_csv(cde_path)
        self.version = self.cde_df["Version"].iloc[0]
        self.date = self.cde_df["Date"].iloc[0]

    def validate_table(self, df, table_name):
        errors = []
        try:
            # Validation logic here
            pass
        except Exception as e:
            errors.append(f"Error validating {table_name}: {str(e)}")

        if errors:
            print(f"Validation errors for {table_name}:")
            for error in errors:
                print(error)
        else:
            print(f"{table_name} validated successfully")

    def update_schema(self, new_cde_path):
        try:
            self.cde_df = pd.read_csv(new_cde_path)
            self.version = self.cde_df["Version"].iloc[0]
            self.date = self.cde_df["Date"].iloc[0]
            print("Schema updated successfully")
        except Exception as e:
            print(f"Error updating schema: {str(e)}")

    def compare_schemas(self, other_schema):
        try:
            # Comparison logic here
            pass
        except Exception as e:
            print(f"Error comparing schemas: {str(e)}")


@dataclasses.dataclass
class MetadataField:
    name: str
    description: str
    data_type: str
    required: bool
    validation: str | None
    shared_key: bool | None
    asap_assigned_key: bool  # for ASAP assigned keys

    values: pd.Series | None = None
    og_values: pd.Series | None = None
    valid_idx: list[int] | None = None
    errors: list[str] | None = None
    invalid_idx: list[int] | None = None

    def __post_init__(self):
        # convert shared_key to bool
        if self.shared_key is not None:
            self.shared_key = bool(self.shared_key)
        # convert required to bool and find asap_assigned_key
        if self.required == "Required":
            self.required = True
        elif self.required == "Optional":
            self.required = False
        elif self.required == "Assigned":
            self.required = False
            self.asap_assigned_key = True
        # if we already assigned as bool just pass
        elif isinstance(self.required, bool):
            pass
        else:
            raise ValueError(f"Unknown required value: {self.required}")

        if self.values is not None:
            self.og_values = self.values.copy()
            self.validate_values()

    def _validate_item(self, value):
        """Validate a single value according to its data type.
        Returns a tuple of (is_valid, error_message)
        """
        # Handle NULL values separately
        if value == NULL:
            return NULL, True, None

        try:
            if self.data_type == "Enum":
                if value not in eval(self.validation):
                    return (
                        value,
                        False,
                        f"'{value}' is not a valid value for {self.name}. Valid values: {eval(self.validation)}",
                    )
                return value, True, None

            elif self.data_type in ["Integer", "Int"]:
                try:
                    val = int(value)  # Try to convert to int
                    return val, True, None
                except (ValueError, TypeError):
                    return (
                        value,
                        False,
                        f"'{value}' is not a valid integer for {self.name}",
                    )

            elif self.data_type == "Float":
                try:
                    val = float(value)  # Try to convert to float
                    return val, True, None
                except (ValueError, TypeError):
                    return (
                        value,
                        False,
                        f"'{value}' is not a valid float for {self.name}",
                    )

            elif self.data_type in ["String", "Str"]:
                if not isinstance(value, str):
                    return (
                        value,
                        False,
                        f"'{value}' is not a valid string for {self.name}",
                    )
                return value, True, None

            else:
                return (
                    value,
                    False,
                    f"Unknown data type '{self.data_type}' for {self.name}",
                )

        except Exception as e:
            return value, False, f"Error validating {self.name}: {str(e)}"

    def _force_type(self):
        """
        use 'object' dtype for Int and Float to allow for NULL values
            note: NULL is defined as "NA" so it will not be cast to int or float

        """

        def _int_typer(x):
            if x == NULL:
                return x
            else:
                # check for scientific notation string. e.g. '1.65E+05'
                if "E" in x.upper():
                    try:
                        val = int(float(x))
                    except ValueError:
                        val = x
                else:
                    val = int(x)
                return val

        # TODO fix bug in CDE resource shouldn't be "Int"
        if self.data_type in ["Integer", "Int"]:
            ret = self.values.apply(_int_typer)
            ret_type = "object"
        elif self.data_type == "Float":
            ret = self.values.apply(lambda x: float(x) if x != NULL else x)
            ret_type = "object"
        elif self.data_type in ["String", "Enum"]:
            ret = self.values.apply(lambda x: str(x) if x != NULL else x)
            ret_type = "str"
        else:
            raise ValueError(f"Unknown 'Type': {self.data_type} \n values unchanged")
            ret_type = "object"
            ret = self.values

        self.values = ret.astype(ret_type)

    def add_values(self, values: pd.Series) -> tuple[bool, list[str]]:
        self.valid = True
        self.errors = []

        self.og_values = values.copy()
        self.values = values
        self.validate_values()
        return self.valid, self.invalid, self.errors

    def validate_values(self):
        self.valid = True
        self.errors = []

        if self.values is None:
            print("no values to validate")
            return

        self._force_type()

        valid_indices = []
        invalid_indices = []
        error_messages = []
        values = self.values

        for idx, value in enumerate(self.values):
            val, is_valid, error_msg = self._validate_item(value)
            if is_valid:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
                error_messages.append(f"val[{idx}]={value}: {error_msg}")
            values[idx] = val

        self.valid_idx = valid_indices
        self.invalid_idx = invalid_indices
        self.errors = error_messages

        self.values = values


class SchemaTable:
    name: str
    fields: list[MetadataField]

    def __init__(self, name: str, fields: list[MetadataField]):
        self.name = name
        self.fields = fields

    @staticmethod
    def load_from_df(name: str, df: pd.DataFrame):
        fields = []
        for _, row in df.iterrows():
            print(row)
            fields.append(
                MetadataField(
                    name=row["Field"],
                    description=row["Description"],
                    data_type=row["DataType"],
                    required=row["Required"],
                    validation=row["Validation"],
                    shared_key=row["Shared_key"] if "Shared_key" in row else None,
                    asap_assigned_key=row["Required"] == "Assigned",
                    values=None,
                    valid=None,
                    errors=None,
                )
            )
        return SchemaTable(name, fields)


class CDE:
    version: str
    date: str
    df: pd.DataFrame
    tables: list[MetadataField]

    def __init__(self, df: pd.DataFrame, version: str, date: str, source: str):
        self.version = version
        self.date = date
        self.df = df

    def __post_init__(self):
        self.tables = self.load_tables()

    @classmethod
    def load_tables(self):
        tables = self.df["Table"].unique()
        for table in tables:
            schema = self.df[self.df["Table"] == table]
            fields = []
            for _, row in schema.iterrows():
                fields.append(
                    MetadataField(
                        name=row["Field"],
                        description=row["Description"],
                        data_type=row["DataType"],
                        required=row["Required"],
                        validation=row["Validation"],
                        shared_key=row["Shared_key"],
                        asap_assigned_key=row["Required"] == "Assigned",
                    )
                )
            self.tables.append(SchemaTable(name=table, fields=fields))

    def get_table(self, name: str) -> SchemaTable:
        for table in self.tables:
            if table.name == name:
                return table
        raise ValueError(f"Table {name} not found in CDE")


@dataclasses.dataclass
class ValidationResult:
    valid_df: pd.DataFrame
    aux_df: pd.DataFrame
    valid_fields: list[MetadataField]
    invalid_fields: list[MetadataField]

    report: str


class MetadataTable:
    name: str
    table: list[MetadataField]
    df: pd.DataFrame
    aux_table: pd.DataFrame

    def __init__(self, name: str, df: pd.DataFrame, cde_df: pd.DataFrame):
        self.name = name
        self.df = df

        table = []
        aux_table = pd.DataFrame()
        schema = cde_df[cde_df["Table"] == self.name]
        for field, data in self.df.items():
            # find the field in the cde_df
            if field not in schema["Field"].values:
                raise ValueError(
                    f"Field {field} not found in CDE for table {self.name}"
                )
                if aux_table.empty:
                    # we also need to grab the any indices..
                    idx_cols = schema[schema["Shared_key"] == 1, "Field"]
                    aux_table = self.df[idx_cols]
                aux_table[field] = data
            else:
                row_idx = schema["Field"] == field

                md_field = MetadataField(
                    field,
                    schema.loc[row_idx, "Description"].item(),
                    schema.loc[row_idx, "DataType"].item(),
                    schema.loc[row_idx, "Required"].item(),
                    schema.loc[row_idx, "Validation"].item(),
                    schema.loc[row_idx, "Shared_key"].item(),
                    schema.loc[row_idx, "Required"].item() == "Assigned",
                    data,
                )

                table.append(md_field)
        self.table = table
        self.aux_table = aux_table

    @staticmethod
    def init_metadata_table_from_cde(
        name: str, df: pd.DataFrame, cde: CDE
    ) -> list[MetadataField]:
        cde_df = CDE.df
        return MetadataTable(name, df, cde_df)

    def validate(self):
        # simply collects the field's validation states
        results = {}
        for field in self.table:
            results[field.name] = (field.valid, field.invalid, field.errors)

        # make a ValidationResult from the results
        # the report sring needs to summarize the incorrect values versus what are required by the schema
        report = "Validation Report \n\n"
        report += f"Table: {self.name}\n\n"
        valid_df = self.df.copy()
        aux_df = self.aux_table.copy()
        valid_fields = []
        invalid_fields = []
        for field in self.table:

            if length(field.invalid_idx) > 0:
                invalid_fields.append(field)
                invalid_values = set(field.values[field.invalid_idx])
                invalid_val_str = ", ".join(map(str, invalid_values))
                report += f"**{self.name}[{field.name}]** errors: {field.errors}\n"
                report += f"\n* invalid values: {invalid_val_str.lstrip(',')}\n"
                report += f"\n* valid values: {field.validation}\n\n"

            else:
                valid_fields.append(field)

        return ValidationResult(valid_df, aux_df, valid_fields, invalid_fields, report)


class MetadataCollection:
    """
    A metadata collection contains the following:
    - name
    - table_names
    - tables
    - aux_tables
    - CDE
    - schema
    - path
    - date
    - dfs
    """

    name: str
    table_names: list[str]
    tables: list[MetadataTable]
    aux_tables: list[MetadataTable]
    cde: CDE
    path: Path
    date: str
    dfs: list[pd.DataFrame]
    all_fields: list[MetadataField]
    table_list: list[str]

    def __init__(
        self,
        name: str | None,
        path: Path | str,
        schema: pd.DataFrame,
        table_names: list[str] | None = None,
        cde_version: str = "v3.1",
        source: str = "",
        date: str | None = None,
    ):
        self.name = name

        self.table_names = (
            schema["Table"].unique().tolist() if table_names is None else table_names
        )
        self.path = Path(path)
        self.cde = CDE(schema, cde_version, date, source)
        self.invalid_tables = []
        self.source = source
        self.date = date if date else pd.Timestamp.now().strftime("%Y%m%d")
        self.aux_tables = []
        self.table_list = []
        self.all_fields = []
        self.tables = []

        self.dfs = self.load_dfs(self.path)
        self.tables = self.load_tables(self.dfs)
        all_fields, table_list = self.load_fields(self.tables)
        self.all_fields = all_fields
        self.table_list = table_list

    def load_dfs(self, path: Path | None):
        dfs = {}
        for table in self.table_names:
            table_path = path / f"{table}.csv"
            if table_path.exists():
                df = read_meta_table(table_path)
                dfs[table] = df
            else:
                print(f"{table} table not found.  need to construct")
                schema = self.cde.df[self.cde.df["Table"] == table]
                df = pd.DataFrame(columns=schema["Field"])
                dfs[table] = df
        return dfs

    def load_tables(self, path: Path):
        tables = []
        for table_name in self.table_names:
            df = self.dfs[table_name]
            table = MetadataTable(table_name, df, self.cde.df)
            tables.append(table)
        return tables

    def load_fields(self, tables: list[MetadataTable]):
        fields = []
        table_names = []
        for table in tables:
            for field in table.table:
                fields.append(field)
                table_names.append(table.name)
        return (fields, table_names)

    def validate_tables(self):
        """
        Validate self.tables against self.cde

        """
        reports = {}
        for table in self.tables:
            report = table.validate()
            reports[table.name] = report
        return reports

    @staticmethod
    def validate_schema(
        df: pd.DataFrame,
        table_names: list[str],
        collection: self,
        cde: CDE,
    ):
        """
        Validate the table against the specific table entries from the CDE
        """

        # Your validation logic here
        # For demonstration, let's assume we just return the input DataFrame
        # need to make the mechanics to make the schema compliatnt table + extras.

        # also need to compose the dictionary into an actual
        results = []
        for table in collection.tables:
            report = table.validate()
            # define helper function to smash report dict to a list of bad entry values and a summary
            #   of what they ought to be.
            df = dfs[table.name]

            df_aux = pd.DataFrame(columns=df.columns)

            results.append(ValidationResult(df, df_aux, name="Validation report"))

        return results


class MetadataPMDBS(MetadataCollection):
    """
    A PMDBS metadata collection contains the following tables:
    - STUDY
    - PROTOCOL
    - SUBJECT
    - SAMPLE
    - ASSAY_RNAseq
    - DATA
    - PMDBS
    - CLINPATH
    - CONDITION
    - MOUSE
    - SPATIAL
    """

    def __init__(
        self,
        name: str | None,
        path: Path | str,
        schema: pd.DataFrame,
        table_names: list[str] | None = None,
        cde_version: str = "v3.1",
        source: str = "team-X-pmdbs-Y",
        date: str | None = None,
    ):
        # force overwrite of table names
        table_names = [
            "SUBJECT",
            "STUDY",
            "PROTOCOL",
            # "SUBJECT",
            "SAMPLE",
            "ASSAY_RNAseq",
            "DATA",
            "PMDBS",
            "CLINPATH",
            "CONDITION",
        ]

        super().__init__(
            name,
            path,
            schema,
            table_names=table_names,
            cde_version=cde_version,
            source=source,
            date=date,
        )
        # name: str | None,
        # path: Path | str,
        # schema: pd.DataFrame,
        # table_names: list[str] | None = None,
        # cde_version: str = "v3.1",
        # source: str = "",
        # date: str | None = None,


# class SpatialMetadataPMDBS(SchemaTable):
#     """
#     A PMDBS spatial metadata collection contains the following tables:
#     - STUDY
#     - PROTOCOL
#     - SUBJECT
#     - SAMPLE
#     - ASSAY_RNAseq
#     - DATA
#     - PMDBS
#     - CLINPATH
#     - CONDITION
#     - SPATIAL


#     """

#     name: str = "pmdbs-spatial"


#     fields: list[SchemaField] = dataclasses.field(default_factory=list)


class MetadataMouse(MetadataCollection):
    """
    A Mouse metadata collection contains the following tables:
    - STUDY
    - PROTOCOL
    - SUBJECT
    - ASSAY_RNAseq
    - DATA
    - CONDITION
    - MOUSE

    """

    def __init__(
        self,
        name: str | None,
        path: Path | str,
        schema: pd.DataFrame,
        table_names: list[str] | None = None,
        cde_version: str = "v3.1",
        source: str = "team-X-pmdbs-Y",
        date: str | None = None,
    ):
        # force overwrite of table names
        table_names = [
            "SAMPLE",
            "STUDY",
            "PROTOCOL",
            "MOUSE",
            "ASSAY_RNAseq",
            "DATA",
            "CONDITION",
        ]

        super().__init__(
            name,
            path,
            schema,
            table_names=table_names,
            cde_version=cde_version,
            source=source,
            date=date,
        )
        # name: str | None,
        # path: Path | str,
        # schema: pd.DataFrame,
        # table_names: list[str] | None = None,
        # cde_version: str = "v3.1",
        # source: str = "",
        # date: str | None = None,


# class ASAP_CRN_MetadataTypeTable:
#     tables: list[SchemaTable]
#     version: str
#     date: str
#     source: st

# @dataclasses.dataclass
# class ASAP_CRN_Schema:
#     cde_df: pd.DataFrame
#     version: str
#     date: str
#     source: str

#     def validate_table(self, df: pd.DataFrame, table_name: str) -> ValidationResult:
#         specific_cde_df = self.cde_df[self.cde_df["Table"] == table_name]
#         return validate_table(df, table_name, specific_cde_df)


def test_script():

    ## define paths

    cde_path = Path.home() / "Projects/ASAP/crn-utils/resource/CDE"
    cde_version = "v3.2"
    cde_df = read_CDE(cde_version, local_path=cde_path)

    print("read CDE")
    meta_path = (
        Path.home()
        / "Projects/ASAP/asap-crn-cloud-dataset-metadata/datasets/cragg-mouse-sn-rnaseq-striatum/metadata"
    )
    # data_df = read_meta_table(meta_path / "DATA.csv")

    print("init CDE")
    cde = CDE(cde_df, cde_version, "20240115", "mouse")
    print("init collection")
    collection = MetadataMouse(
        "mouse-pmdbs",
        meta_path,
        cde_df,
        cde_version=cde_version,
        source="cragg-mouse-sn-rnaseq-striatum",
    )
    print("validate tables")
    report = collection.validate_tables()
    print(report)


if __name__ == "__main__":

    test_script()
