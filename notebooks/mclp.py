import pandas as pd
import io

def create_infogroup_df(file_name, city, state):
    lines_left = True
    with open(file_name, encoding="ISO-8859-1") as f:
        header = f.readline()
        whole_df = None
        while True:
            lines = [header]
            for i in range(5000):
                line = f.readline()
                if not line:
                    lines_left = False
                    break
                lines.append(line)
            df = pd.read_csv(io.StringIO("\n".join(lines)))
            # print(df.columns)
            df_filtered = df.loc[(df.loc[:, "CITY"] == city) & (
                    df.loc[:, "STATE"] == state), :]
            if len(df_filtered) > 0:
                whole_df = df_filtered if whole_df is None else pd.concat([whole_df, df_filtered])
            if not lines_left:
                return whole_df
            

def change_infogroup_column_names(df):
    df.loc[:, ["infogroup"]] = True
    df = df.loc[:, ["COMPANY", "CITY", "ADDRESS LINE 1", "LATITUDE", "LONGITUDE", "SALES VOLUME (9) - LOCATION", "EMPLOYEE SIZE (5) - LOCATION", "PARENT ACTUAL SALES VOLUME"]]
    df = df.rename(columns={"ADDRESS LINE 1": "street1", "CITY": "city", "COMPANY": "name", "LATITUDE": "latitude", "LONGITUDE": "longitude", "SALES VOLUME (9) - LOCATION": "sales_volume", "EMPLOYEE SIZE (5) - LOCATION": "employee_size", "PARENT ACTUAL SALES VOLUME": "parent_sales_volume"})
    return df


def change_foottraffic_column_names(df):
    df.loc[:, ["foottraffic"]] = True
    df = df.rename(columns={"location_name": "name", "street_address": "street1"})
    return df