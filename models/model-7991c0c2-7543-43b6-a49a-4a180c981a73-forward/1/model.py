import os
import json
import logging
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils

from pathlib import Path
from typing import Optional
from datetime import datetime
from pydantic_settings import BaseSettings

from mls_ml_libs.db.timeseries import TimeseriesDBClient
from data_processing_libs.serving.graph import Graph


class TestSettings(BaseSettings):
    Redis__Host: str = ""
    Redis__Port: int = 0
    Redis__Database: int = 0
    Redis__Password: Optional[str] = ""
    Redis__Ssl: bool = False
    Redis__User: Optional[str] = ""
    Authentication__ClientId: str = ""
    Authentication__ClientSecret: str = ""
    Authentication__Authority: str = ""
    API__TimeseriesDB__Execute__Url: str = ""
    x_subscription_id: str = ""
    x_tenant_id: str = ""
    x_project_id: str


settings = TestSettings()


class TritonPythonModel:
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)

        settings = TestSettings()

        self.client = TimeseriesDBClient(
            x_subscription_id=settings.x_subscription_id,
            x_tenant_id=settings.x_tenant_id,
            settings=settings,
        )

        self.graph = Graph(self.client)

        self.output0_dtype = np.float64
        self.output1_dtype = np.object_

    def execute(self, requests):
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)

        now = datetime.now()  # current date and time

        logging.basicConfig(
            filename=f'{now.strftime("%Y%m%d%H%M%S")}-output.txt',
            level=logging.DEBUG,
            format="",
        )

        responses = []
        for request in requests:
            # Processing input requests data
            # mapping for query timeseries data
            mapping = pb_utils.get_input_tensor_by_name(request, "NAME_MAPPING")
            mapping = [i.decode("ascii") for i in mapping.as_numpy()]
            data_mapping = json.loads(mapping[0])

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            cols = [i.decode("ascii") for i in in_0.as_numpy()]
            feats = [i.decode("ascii") for i in in_1.as_numpy()]

            full_map = dict(zip(cols, feats))

            data_key = full_map["data_key"]
            timestamp = full_map["serving_timestamp"]
            mode = full_map["mode"]

            del full_map["mode"]
            del full_map["data_key"]
            del full_map["serving_timestamp"]

            columns = list(full_map.keys())
            features = [[float(i) for i in full_map.values()]]

            request_df = pd.DataFrame(features, columns=columns)
            logging.info(f"[REQUEST_DF]{request_df}")
            current_df = request_df.copy()

            logging.info(f"[INPUT DF] {current_df}")

            config_path = str(Path(__file__).resolve().parent)

            self.graph.initialize(config_dir=config_path)

            if mode != self.graph.mode:
                raise pb_utils.TritonModelException(f"{mode} mode is NOT valid")

            logging.info(f"[GRAPH] Initialized")

            logging.info(
                f"[GRAPH] Executing:\n"
                f"- [TYPE]: {current_df}\n"
                f"- [CURRENT_DF]: {current_df}\n"
                f"- [DATA_MAPPING]: {data_mapping}\n"
                f"- [DATA_KEY]: {data_key}\n"
                f"- [TIMESTAMP]: {timestamp}\n"
            )

            if mode == "backward":
                result_df = self.graph.backward(input_dataframe=current_df)
            else:
                result_df = self.graph.execute(
                    input_dataframe=current_df,
                    name_mapping=data_mapping,
                    data_key=data_key,
                    to_timestamp=timestamp,
                )

            final_df = result_df.where(
                pd.notnull(result_df), 0
            )  # replace NaN to 0 for json response

            logging.info(f"[FINAL RESULT] {final_df}")

            out_0 = np.array(list(final_df.columns)).astype(np.object_)

            out_1 = final_df.values[-1]

            output0_dtype = self.output0_dtype
            output1_dtype = self.output1_dtype

            logging.info(f"[OUT_0] {out_0}")
            logging.info(f"[OUT_1] {out_1}")

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0.astype(output1_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

            self.graph.reset()

            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
