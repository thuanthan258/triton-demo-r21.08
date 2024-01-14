from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import pandas as pd
from typing import List
from abc import ABC, abstractmethod
from data_processing_libs.transforms import BaseTransform


class MinMaxTransform(BaseTransform):
    def __init__(self, working_dir, **kwargs):
        super().__init__(working_dir, **kwargs)
        self.init_kwargs = {}
        if kwargs:
            self.init_kwargs = dict(kwargs)

    def get_serving_config(self):
        serving_config = {"historical_days": 0}
        return serving_config

    def get_init_kwargs(self):
        return self.init_kwargs

    def compute_columns_name(self, columns: List[str]) -> List[str]:
        return [f"{i}" for i in columns]

    def transform(self, df: pd.DataFrame):
        scaler = MinMaxScaler()
        scaler.fit(df)

        output_path = os.path.join(self.working_dir, "model.pickle")
        with open(output_path, "wb") as f:
            pickle.dump(scaler, f)
        result = scaler.transform(df)
        return pd.DataFrame(result, columns=df.columns)

    def forward(self, df: pd.DataFrame):
        model_path = os.path.join(self.working_dir, "model.pickle")
        with open(model_path, "rb") as f:
            scaler = pickle.load(f)
        print(df.columns)
        print(scaler.n_features_in_)
        result = scaler.transform(df.values)
        return pd.DataFrame(result, columns=df.columns)

    def backward(self, df: pd.DataFrame):
        model_path = os.path.join(self.working_dir, "model.pickle")
        with open(model_path, "rb") as f:
            scaler = pickle.load(f)
        result = scaler.inverse_transform(df)
        return pd.DataFrame(result, columns=df.columns)
