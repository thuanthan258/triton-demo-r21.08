import os
import pandas as pd

# Input your configured Environment variable here
# For example, we want to get config value a, with default value 0 as in the snippet beflow
max_historical_days = int(os.getenv("max_historical_days", "0")) # User input this


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
    df: pd.DataFrame, method: str = "ffill", value=None, rolling_params=None
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
        "max_historical_days": 1
    }
    df = df.copy()

    # TODO: implement your logic here

    # For any transformation with model, you should return
    # return _df, model_weight
    if method == "rolling" and rolling_params is None:
        raise ValueError("'rolling_params' must be provided when method is 'rolling'")
    elif method == "value" and value is None:
        raise ValueError("'value' must be provided when method is 'value'")

    if method == "ffill":
        df.ffill(inplace=True)
        df.bfill(inplace=True)  # Fill remaining NaNs
    elif method == "bfill":
        df.bfill(inplace=True)
        df.ffill(inplace=True)  # Fill remaining NaNs
    elif method == "rolling":
        df.fillna(df.rolling(**rolling_params).mean(), inplace=True)
        df.bfill(inplace=True)  # Fill remaining NaNs
        df.ffill(inplace=True)  # Fill remaining NaNs
    elif method == "value":
        df.fillna(value, inplace=True)
    else:
        raise ValueError(f"Invalid fill method: {method}")

    result["data"] = df
    return result
    