import json
import os
import pandas as pd
from typing import List
from data_processing_libs.transforms import BaseTransform

class SampleClass(BaseTransform):
    def __init__(self, working_dir, **kwargs):
        super().__init__(working_dir, **kwargs)

        # Config values from UI will be retrieved here
        self.env_variables = kwargs.get("env_variables", {})

    def compute_columns_name(self, columns: List[str]) -> List[str]:
        """
        Apply rename logic to columns before performing transformation
        """
        return [f"{i}" for i in columns]

    def transform(self, df: pd.DataFrame):
        """
        Transformation logic used when training
        """
        return df

    def forward(self, df: pd.DataFrame):
        """
        Transformation logic use when serving (inference)
        """
        return self.transform(df)

    def backward(self, df: pd.DataFrame):
        """
        Revert transformation of forward, used when serving for blocks with targets to return them to original values
        """
        return df