# imports
import pandas as pd

# wrape this in try/except to make suing the ReportCollector portable
# probably an abstract base class would be better
try:
    import streamlit as st
    print("Streamlit imported successfully")

except ImportError:
    class DummyStreamlit:
        @staticmethod
        def markdown(self,msg):
            pass
        def error(self,msg):
            pass
        def header(self,msg):
            pass        
        def subheader(self,msg):
            pass    
        def divider(self):
            pass
    st = DummyStreamlit()
    print("Streamlit NOT successfully imported")


NULL = "NA"


def columnize( itemlist ):
    NEWLINE_DASH = ' \n- '
    if len(itemlist) > 1:
        return f"- {itemlist[0]}{NEWLINE_DASH.join(itemlist[1:])}"
    else:
        return f"- {itemlist[0]}"
    
def read_meta_table(table_path):
    # read the whole table
    try:
        table_df = pd.read_csv(table_path,dtype=str)
    except UnicodeDecodeError:
        table_df = pd.read_csv(table_path, encoding='latin1',dtype=str)

    # drop the first column if it is just the index
    if table_df.columns[0] == "Unnamed: 0":
        table_df = table_df.drop(columns=["Unnamed: 0"])

    table_df.replace({"":NULL, pd.NA:NULL, "none":NULL, "nan":NULL, "Nan":NULL}, inplace=True)

    return table_df


class ReportCollector:
    """
    Class to collect and log messages, errors, and markdown to a log file and/or streamlit
    """

    def __init__(self, destination="both"):
        self.entries = []
        self.filename = None

        if destination in ["both", "streamlit"]:
            self.publish_to_streamlit = True
        else:
            self.publish_to_streamlit = False


    def add_markdown(self, msg):
        self.entries.append(("markdown", msg))
        if self.publish_to_streamlit:
            st.markdown(msg)


    def add_error(self, msg):
        self.entries.append(("error", msg))
        if self.publish_to_streamlit:
            st.error(msg)

    def add_header(self, msg):
        self.entries.append(("header", msg))
        if self.publish_to_streamlit:    
            st.header(msg)

    def add_subheader(self, msg):
        self.entries.append(("subheader", msg))
        if self.publish_to_streamlit:    
            st.subheader(msg)

    def add_divider(self):
        self.entries.append(("divider", None))
        if self.publish_to_streamlit:    
            st.divider()

    
    def write_to_file(self, filename):
        self.filename = filename
        with open(filename, 'w') as f:
            report_content = self.get_log()
            f.write(report_content)
    

    def get_log(self):
        """ grab logged information from the log file."""
        report_content = []
        for msg_type, msg in self.entries:
            if msg_type == "markdown":
                report_content += msg + '\n'
            elif msg_type == "error":
                report_content += f"🚨⚠️❗ **{msg}**\n"
            elif msg_type == "header":
                report_content += f"# {msg}\n"
            elif msg_type == "subheader":
                report_content += f"## {msg}\n"
            elif msg_type == "divider":
                report_content += 60*'-' + '\n'
        
        return "".join(report_content)

    def reset(self):
        self.entries = []
        self.filename = None

    def print_log(self):
        print(self.get_log())

def process_table(df, table_name, cde_schema):
    """Process a table according to CDE schema and extract auxiliary fields.
    
    Args:
        df (pd.DataFrame): Input dataframe to process
        table_name (str): Name of the table being processed
        cde_schema (pd.DataFrame): CDE schema dataframe
        
    Returns:
        tuple: (processed_df, auxiliary_df, report)
            - processed_df: DataFrame with only valid CDE fields
            - auxiliary_df: DataFrame with shared keys and extra fields
            - report: Validation report
    """
    schema = cde_schema[cde_schema['Table'] == table_name]
    report = ReportCollector(destination="NA")
    full_table, report = validate_table(df.copy(), table_name, schema, report)
    report.print_log()

    # Extract valid CDE fields
    processed_df = full_table[schema['Field'].tolist()]
    
    # Get auxiliary fields (shared keys + extra columns)s
    aux_fields = list(set(df.columns) - set(schema['Field'].tolist()))
    if len(aux_fields) == 0:
        return processed_df, pd.DataFrame(), report
    else:
        auxiliary_df = df[aux_fields]
        aux_fields = (schema[schema["Shared_key"]==1]["Field"].to_list() + 
                 aux_fields)
        auxiliary_df = df[aux_fields]
        return processed_df, auxiliary_df, report

def validate_table(df: pd.DataFrame, table_name: str, specific_cde_df: pd.DataFrame, out: ReportCollector ):
    """
    Validate the table against the specific table entries from the CDE
    """
    def my_str(x):
        return f"'{str(x)}'"
        
    missing_required = []
    missing_optional = []
    null_fields = []
    invalid_entries = []
    total_rows = df.shape[0]
    for field in specific_cde_df["Field"]:
        entry_idx = specific_cde_df["Field"]==field

        opt_req = "REQUIRED" if specific_cde_df.loc[entry_idx, "Required"].item()=="Required" else "OPTIONAL"

        if field not in df.columns:
            if opt_req == "REQUIRED":
                missing_required.append(field)
            else:
                missing_optional.append(field)

            # print(f"missing {opt_req} column {field}")

        else:
            datatype = specific_cde_df.loc[entry_idx,"DataType"]
            if datatype.item() == "Integer":
                # recode "Unknown" as NULL
                print(f"recoding {field} as int")

                df.replace({"Unknown":NULL, "unknown":NULL}, inplace=True)
                try:
                    df[field].apply(lambda x: int(x) if x!=NULL else x )
                except Exception as e:
                    # print(e)
                    # print(f"Error in {field}")
                    invalid_values = df[field].unique()
                    n_invalid = invalid_values.shape[0]
                    valstr = "int or NULL ('NA')"
                    invalstr = ', '.join(map(my_str,invalid_values))
                    invalid_entries.append((opt_req, field, n_invalid, valstr, invalstr))

                # test that all are integer or NULL, flag NULL entries
            elif datatype.item() == "Float":
                # recode "Unknown" as NULL
                df.replace({"Unknown":NULL, "unknown":NULL}, inplace=True)
                try:
                    df[field] = df[field].apply(lambda x: float(x) if x!=NULL else x )
                except Exception as e:
                    # print(e)
                    # print(f"Error in {field}")
                    invalid_values = df[field].unique()
                    n_invalid = invalid_values.shape[0]
                    valstr = "float or NULL ('NA')"
                    invalstr = ', '.join(map(my_str,invalid_values))
                    invalid_entries.append((opt_req, field, n_invalid, valstr, invalstr))

                # test that all are float or NULL, flag NULL entries
            elif datatype.item() == "Enum":

                valid_values = eval(specific_cde_df.loc[entry_idx,"Validation"].item())
                valid_values += [NULL]
                entries = df[field]
                valid_entries = entries.apply(lambda x: x in valid_values)
                invalid_values = entries[~valid_entries].unique()
                n_invalid = invalid_values.shape[0]
                if n_invalid > 0:
                    valstr = ', '.join(map(my_str, valid_values))
                    invalstr = ', '.join(map(my_str,invalid_values))
                    invalid_entries.append((opt_req, field, n_invalid, valstr, invalstr))
            else: #dtype == String
                pass
            
            n_null = (df[field]==NULL).sum()
            if n_null > 0:            
                null_fields.append((opt_req, field, n_null))


    # now compose report...
    if len(missing_required) > 0:
        out.add_error(f"Missing Required Fields in {table_name}: {', '.join(missing_required)}")
        for field in missing_required:
            df[field] = NULL

    else:
        out.add_markdown(f"All required fields are present in *{table_name}* table.")

    if len(missing_optional) > 0:
        out.add_error(f"Missing Optional Fields in {table_name}: {', '.join(missing_optional)}")
        for field in missing_optional:
            df[field] = NULL

    if len(null_fields) > 0:
        # print(f"{opt_req} {field} has {n_null}/{df.shape[0]} NULL entries ")
        out.add_error(f"{len(null_fields)} Fields with empty (NULL) values:")
        for opt_req, field, count in null_fields:
            out.add_markdown(f"\n\t- {field}: {count}/{total_rows} empty rows ({opt_req})")
    else:
        out.add_markdown(f"No empty entries (NULL) found .")


    if len(invalid_entries) > 0:
        out.add_error(f"{len(invalid_entries)} Fields with invalid entries:")
        for opt_req, field, count, valstr, invalstr in invalid_entries:
            str_out = f"- _*{field}*_:  invalid values 💩{invalstr}\n"
            str_out += f"    - valid ➡️ {valstr}"
            out.add_markdown(str_out)
    else:
        out.add_markdown(f"No invalid entries found in Enum fields.")

    for field in df.columns:
        if field not in specific_cde_df["Field"].values:
            out.add_error(f"Extra field in {table_name}: {field}")
   


    return df, out


def create_valid_table(df: pd.DataFrame, table_name: str, cde_df: pd.DataFrame):
    """
    use validate_table to create a validated table
    """
    schema = cde_df[cde_df['Table'] == table_name]

    report = ReportCollector(destination="NA")
    df_out, report = validate_table(df.copy(), table_name, schema, report)
    
    valid_fields = schema['Field'].unique()
    aux_fields = set(df.columns) - set(valid_fields)
    if aux_fields:
        df_aux = df[list(aux_fields)]
    else:
        df_aux = None

    return df_out, df_aux



def calculate_missingness(df: pd.DataFrame, table_name: str, schema: pd.DataFrame, missing_threshold: float=0.5, metadata_version: str="v3.1.0":
    """
    get missingness for each column in the table

    first look at all required columns, then all optional columns

    """
    missingness = lambda x: x.isnull().mean()

    # Function to get True columns for each row
    def get_true_columns(row):
        return row[row == True].index.tolist()

    # now compose report...
    def fmtlist( itemlist ):
        NEWLINE_DASH = ', ' #' \n- '
        if len(itemlist) > 1:
            return f": {itemlist[0]}{NEWLINE_DASH.join(itemlist[1:])}"
        else:
            return f": {itemlist[0]}"

    all_fields = schema['Field']

    # note that these fields are all lower case
    required_fields = schema[schema['Required'] == 'Required']['Field'].str.lower().to_list()
    optional_fields = schema[schema['Required'] == 'Optional']['Field'].str.lower().to_list()

    df_required = df[required_fields].copy()
    df_optional = df[optional_fields].copy()

    # overall missingness
    required_missingness = df_required.isnull().mean()
    optional_missingness = df_optional.isnull().mean()

    # missingness grouped by ds
    required_missingness_by_ds = df.groupby('dataset')[required_fields].apply(missingness)
    optional_missingness_by_ds = df.groupby('dataset')[optional_fields].apply(missingness)

    # apply threshold to get the "missing" fields
    required_missing = required_missingness[required_missingness > missing_threshold].index.to_list()
    optional_missing = optional_missingness[optional_missingness > missing_threshold].index.to_list()

    missing_required_by_ds = (required_missingness_by_ds > missing_threshold).apply(get_true_columns, axis=1).to_dict()
    missing_optional_by_ds = (optional_missingness_by_ds > missing_threshold).apply(get_true_columns, axis=1).to_dict()


    # format
    required_missingness = required_missingness.apply(lambda x: f"{100*x:.1f}%")
    optional_missingness = optional_missingness.apply(lambda x: f"{100*x:.1f}%")

    required_missingness_by_ds = required_missingness_by_ds.map(lambda x: f"{100*x:.1f}%")
    optional_missingness_by_ds = optional_missingness_by_ds.map(lambda x: f"{100*x:.1f}%")


    report = f"""\n
# Missingness Report for {table_name} Table (CDE {metadata_version})

## Overall Missingness
"""
    tmp = required_missingness.reset_index().rename(columns={'index': 'Required Fields', 0: 'Missingness'}).to_markdown(index=False)
    report += f"{tmp}\n"

    tmp = optional_missingness.reset_index().rename(columns={'index': 'Optional Fields', 0: 'Missingness'}).to_markdown(index=False)
    report += f"\n\n{tmp}\n"

    thresh_str = f"{100*missing_threshold:.1f}%"

    report += f"""\n
### Overall Missing Fields (\> {thresh_str} missingness)
"""

    tmp1 = fmtlist(required_missing)
    tmp2 = fmtlist(optional_missing)
    report += f"""\n
_Missing **Required** fields_ {tmp1}

_Missing **Optional** fields_ {tmp2}

---------------------------------

## Missingness by Dataset 
**Required fields:**
{required_missingness_by_ds.to_markdown()}

**Optional fields:**
{optional_missingness_by_ds.to_markdown()}    

### Missing Fields by Dataset (\> {thresh_str} missingness):

_Missing **Required** fields_

"""

    for k,v in missing_required_by_ds.items():
        report += f"- *{k}*{fmtlist(v)}\n"

    report += f"""\n

_Missing **Optional** fields_

"""

    for k,v in missing_optional_by_ds.items():
        report += f"- *{k}*{fmtlist(v)}\n"


    return report
