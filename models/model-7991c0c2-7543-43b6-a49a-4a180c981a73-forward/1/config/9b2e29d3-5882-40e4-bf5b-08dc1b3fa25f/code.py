import json
import os
import pandas as pd
from typing import List
from data_processing_libs.transforms import BaseTransform

window = int(os.environ.get("window", "2"))
print(window)


class LagFeature(BaseTransform):
    def __init__(self, working_dir, **kwargs):
        super().__init__(working_dir, **kwargs)
        self.init_kwargs = {}
        self.config = {}
        if kwargs:
          self.init_kwargs = dict(kwargs)
          
    def get_serving_config(self):
        serving_config = {
          "historical_days": window
        }
        return serving_config

    def get_init_kwargs(self):
        return self.init_kwargs

    def dump_config(self):
        output_path = os.path.join(self.working_dir, "config.json")
        with open(output_path, "w") as f:
            json.dump(self.config, f)

    def load_config(self):
        output_path = os.path.join(self.working_dir, "config.json")
        with open(output_path, "r") as f:
            config = json.load(f)
        return config

    def compute_columns_name(self, columns: List[str]) -> List[str]:
        return [f"{i}" for i in columns]

    def transform(self, df: pd.DataFrame):
        lagged_df = pd.concat([df.shift(i) for i in range(1, window + 1)], axis=1)
        columns_name = [
            [f"{col}_{i}" for col in df.columns] for i in range(1, window + 1)
        ]
        columns_name = [item for sublist in columns_name for item in sublist]

        lagged_df.columns = columns_name
        self.config["historical_days"] = window
        self.dump_config()
        df = pd.concat([df, lagged_df], axis=1)
        return df

    def forward(self, df: pd.DataFrame):
        self.config = self.load_config()
        historical_days = self.config["historical_days"]
        lagged_df = pd.concat(
            [df.shift(i) for i in range(1, historical_days + 1)], axis=1
        )
        columns_name = [
            [f"{col}_{i}" for col in df.columns] for i in range(1, historical_days + 1)
        ]
        columns_name = [item for sublist in columns_name for item in sublist]

        lagged_df.columns = columns_name
        df = pd.concat([df, lagged_df], axis=1)
        return df

    def backward(self, df: pd.DataFrame):
        return df