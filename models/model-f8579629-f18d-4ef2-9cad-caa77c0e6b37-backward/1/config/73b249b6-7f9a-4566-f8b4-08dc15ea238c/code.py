import json
import os
import pandas as pd
from typing import List
from data_processing_libs.transforms import BaseTransform


class TargetPass(BaseTransform):
    def __init__(self, working_dir, **kwargs):
        super().__init__(working_dir, **kwargs)
        self.init_kwargs = {}
        if kwargs:
          self.init_kwargs = dict(kwargs)

    def get_serving_config(self):
        serving_config = {
          "historical_days": 0
        }
        return serving_config
  
    def get_init_kwargs(self):
        return self.init_kwargs

    def compute_columns_name(self, columns: List[str]) -> List[str]:
        return [f"{i}" for i in columns]

    def transform(self, df: pd.DataFrame):
        return df

    def forward(self, df: pd.DataFrame):
        return df

    def backward(self, df: pd.DataFrame):
        return df