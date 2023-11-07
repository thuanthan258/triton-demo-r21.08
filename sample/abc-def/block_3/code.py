import os
import pandas as pd

# Input your configured Environment variable here
# For example, we want to get config value a, with default value 0 as in the snippet beflow
window = int(os.getenv("window", 2)) # User input this


def rename_column(input_col: str):
    """
    Rename column
    This function will apply the renaming logic to all of the column in the block.
    Currently we use f_string to handle this.
    For example, if we have input_col = "feature_1" and new_column = f"{input_col}_123", new_column will be feature_1_123
    As default, result will be the same as input column
    """
    new_column = f"{input_col}"
    return new_column

def execute(
    df: pd.DataFrame, weight=None, window_size: int=window
    ):
    """
    Transform the data
    This will apply any logic you write here to the data passing through the block.
    For example, if we return df * 2, all the value of the df will be multiplied by 2
    You can also use config values define above here. For example return df * 2 + a.
    For any execution with model weight, please return also the weight as well as the dataframe so that other set can be used to transform the data.
    Model weight and be the weight of the model, or the full model object. The service will pickle it for later use.
    As default, result will be the same as df
    """
    result = {
        "data": None,
        "model": None,
        "max_historical_days": window_size
    }
    df = df.copy()

    # TODO: implement your logic here

    # For any transformation with model, you should return
    # return _df, model_weight
    for column in df.columns:
        # For each column, create lag features up to the specified window size
        for lag in range(1, window_size):
            # The new feature is added to the DataFrame with a name in the format column_lag
            df[column + "_" + str(lag)] = df[column].shift(lag)

    result["data"] = df
    return result
    