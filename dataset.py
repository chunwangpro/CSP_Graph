# Description: This script loads a CSV file and processes it in the following way:

## Converts all floats to integers by multiplying them by 10^d, where d is the maximum number of decimal places in the dataset.
## Sorts the columns by the number of unique values in each column.

## The script also contains a function to recover the original table from the processed table.

import pandas as pd


def _load_data(file_path):
    if file_path[:4] == "test":
        return pd.read_csv(f"datasets/{file_path}.csv", header=None)

    if file_path == "census-2":
        df_name = "census"
        columns_to_read = [0, 2]
    elif file_path == "census-3":
        df_name = "census"
        columns_to_read = [0, 2, 4]
    elif file_path == "wine-2":
        df_name = "wine"
        columns_to_read = [0, 1]
    elif file_path == "wine-3":
        df_name = "wine"
        columns_to_read = [0, 1, 2]
    return pd.read_csv(f"datasets/{df_name}.csv", header=None, usecols=columns_to_read)


# def _convert_datatype_to_int(df):
#     def find_decimal_places(x):
#         if isinstance(x, float):
#             decimal_part = str(x).split(".")[1]
#             return len(decimal_part.rstrip("0"))
#         return 0

#     max_decimal_places = df.map(find_decimal_places).max().max()
#     df_int = (df * 10**max_decimal_places).astype(int)
#     return df_int, max_decimal_places


def _convert_datatype_to_int(df):
    def find_decimal_places(x):
        if isinstance(x, float):
            decimal_part = str(x).split(".")[1]
            return len(decimal_part.rstrip("0"))
        return 0

    max_decimal_places_per_column = df.applymap(find_decimal_places).max()
    df_int = df.apply(lambda col: (col * 10 ** max_decimal_places_per_column[col.name]).astype(int))
    return df_int, max_decimal_places_per_column


def _sort_by_column_unique_number(df, ascending=False):
    original_table_columns = df.columns.tolist()
    sorted_table_columns = df.nunique().sort_values(ascending=ascending).index.tolist()
    modified_data = df[sorted_table_columns].copy().to_numpy().reshape(len(df), -1)
    return modified_data, original_table_columns, sorted_table_columns


def load_and_process_dataset(file_path, resultsPath):
    # df = pd.read_csv(f"datasets/{file_path}.csv", header=None)
    df = _load_data(file_path)
    # df = pd.read_csv(f"datasets/{file_path}.csv", header=None)
    df = _load_data(file_path)
    df.to_csv(f"{resultsPath}/original_table.csv", index=False, header=False)
    df_int, max_decimal_places = _convert_datatype_to_int(df)
    modified_data, original_table_columns, sorted_table_columns = _sort_by_column_unique_number(
        df_int, ascending=False
    )
    return modified_data, original_table_columns, sorted_table_columns, max_decimal_places


# def recover_table_as_original(
#     data, original_table_columns, sorted_table_columns, max_decimal_places
# ):
#     df = pd.DataFrame(data, columns=sorted_table_columns)
#     recovered_df = df[original_table_columns].copy()
#     recovered_df /= 10**max_decimal_places
#     return recovered_df


def recover_table_as_original(
    data, original_table_columns, sorted_table_columns, max_decimal_places_per_column
):
    df = pd.DataFrame(data, columns=sorted_table_columns)
    recovered_df = df[original_table_columns].copy()
    recovered_df = recovered_df.apply(
        lambda col: col / 10 ** max_decimal_places_per_column[col.name]
    )
    return recovered_df


if __name__ == "__main__":
    file_path = "wine"
    resultsPath = "results/"
    modified_data, original_table_columns, sorted_table_columns, max_decimal_places = (
        load_and_process_dataset(file_path, resultsPath)
    )
    # modified_data.to_csv(f"{file_path}_processed.csv", index=False, header=False)

    # recovered_Table_Generated = recover_table_as_original(
    #     Table_Generated, original_table_columns, sorted_table_columns, max_decimal_places
    # )
