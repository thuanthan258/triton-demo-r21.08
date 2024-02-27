import json
import os
import pandas as pd
from typing import List
from data_processing_libs.transforms import BaseTransform

value = int(os.environ.get("value", "0"))


class DefaultTransformer(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_columns_name(self, columns: List[str]) -> List[str]:
        return [f"{i}" for i in columns]

    def transform(self, df: pd.DataFrame):
        return df

    def forward(self, df: pd.DataFrame):
        return df

    def backward(self, df: pd.DataFrame):
        return df
