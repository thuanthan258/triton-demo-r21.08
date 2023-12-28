import json
import os
import pickle
import datetime
import triton_python_backend_utils as pb_utils
import asyncio
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from pydantic_settings import BaseSettings
from typing import Optional
from mls_ml_libs.db.timeseries import TimeseriesDBClient


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


settings = TestSettings()


class TritonPythonModel:
    def read_function_from_file(self, file_path: str) -> callable:
        """
        Reads a function from a file and returns it.

        Args:
            file_path: The path to the folder containing the file's function.
            compute_type: The type of computation to perform.

        Returns:
            The function to be executed.
        """
        # Construct the full file path
        if os.path.isdir(file_path):
            files = os.listdir(file_path)
            file_path = os.path.join(file_path, files[0])

        logger.info(f"Reading function from file: {file_path}")
        # Read the contents of the file
        with open(file_path, "r") as file:
            code = compile(file.read(), file_path, "exec")
            namespace = {"__file__": file_path}

            # Execute the code in the namespace
            exec(code, namespace)
            # Return the function from the namespace
            return namespace["execute"]

    def get_timeseries_data(
        self,
        feats: list,
        num_historical_days: int,
        timestamp: int,
        timescale_client: TimeseriesDBClient = None,
    ):
        logger.info(f"\tFeatures: {feats}")
        logger.info(f"\tNumber historical days: {num_historical_days}")
        logger.info(f"\tTimestamp: {timestamp}")

        if not timescale_client:
            return pd.DataFrame(
                [[1 for i in range(len(feats))] for j in range(num_historical_days)],
                columns=feats,
            )
        loop = asyncio.get_event_loop()

        # Http client doesn't support async, so this is a workaround
        # https://github.com/triton-inference-server/server/issues/3671
        response = loop.run_until_complete(
            timescale_client.get_data_from_db(
                data_key="{{timeseries_db_key}}",
                data_metrics=feats,
                from_timestamp=timestamp - 2000,
                to_timestamp=timestamp,
            )
        )
        logger.info(f"Done getting data! {response}")
        # Build dataframe
        result_dict = {}
        for i, name in enumerate(feats):
            current_values = response[i]["values"]
            for val in current_values:
                result_dict[i].append(val["value"])
        logger.info(f"Result is {result_dict}")
        df = pd.DataFrame(result_dict)
        return df[-num_historical_days:]

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
        # TODO: Implement time series db client when ready
        settings = TestSettings()
        # TODO: Fix when update TimeseriesDBClient
        # self.client = TimeseriesDBClient(
        #     x_subscription_id=settings.x_subscription_id,
        #     x_tenant_id=settings.x_tenant_id,
        #     settings=settings,
        # )
        self.client = None

        config_path = str(Path(__file__).resolve().parent)
        config_file = os.path.join(config_path, "config/execution_plan.json")
        with open(config_file) as f:
            self.config = json.load(f)
        self.topo_ids = self.config["topo_ids"]
        self.modules = {}
        self.models = {}

        for _id in self.topo_ids:
            current_block_folder = os.path.join(config_path, f"config/{_id}")

            module = None
            model = None
            # Load code module
            module_file = os.path.join(current_block_folder, "code.py")
            if os.path.isfile(module_file):
                # module = self.read_function_from_file(module_file)
                module = module_file

            model_file = os.path.join(current_block_folder, "model.pickle")
            if os.path.isfile(model_file):
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
            self.modules[_id] = module
            self.models[_id] = model

        self.output0_dtype = np.float64
        self.output1_dtype = np.object_

    def execute(self, requests):
        responses = []
        for request in requests:
            block_output = {}
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            cols = [i.decode("ascii") for i in in_0.as_numpy()]
            feats = [in_1.as_numpy()]

            df = pd.DataFrame(feats, columns=cols)
            logger.info(df)
            current_df = df.copy()
            block_output["input"] = current_df

            for _id in self.topo_ids:
                current_config: dict = self.config[_id]
                parents: dict = current_config["inputs"]
                max_historical_days: int = current_config["max_historical_days"]

                all_parents = []
                for parent_id, parent_config in parents.items():
                    input_cols: str = parent_config["input_cols"]
                    name_mapping: dict = parent_config["name_mapping"]
                    parent_df: pd.DataFrame = block_output[parent_id].copy()
                    parent_df = parent_df[input_cols]
                    parent_df.rename(name_mapping, inplace=True)
                    all_parents.append(parent_df)
                current_df: pd.DataFrame = pd.concat(all_parents, axis=1)

                if current_config["env_var"]:
                    for key, val in current_config["env_var"].items():
                        os.environ[key] = str(val)

                if max_historical_days != 0:
                    os.environ["max_historical_days"] = str(max_historical_days)
                    current_timestamp = 1577852940 + 1000
                    historical_df: pd.DataFrame = self.get_timeseries_data(
                        feats=list(current_df.columns),
                        num_historical_days=max_historical_days,
                        timestamp=current_timestamp,
                        timescale_client=self.client,
                    )
                    current_df = pd.concat([historical_df, current_df], axis=0)

                module = None
                module_file = self.modules[_id]
                if module_file:
                    module = self.read_function_from_file(module_file)
                model = self.models[_id]
                if module:
                    if model:
                        current_df = module(current_df, model)
                        logger.info("Transformed with model")
                    else:
                        current_df = module(current_df)
                    current_df = current_df["data"][-1:]
                    logger.info(f"DF after transform: \n{current_df}")
                logger.info(current_df)
                logger.info(f"The  id is {_id}")
                logger.info(f"The input cols are {current_df.columns}")
                logger.info(f"The dataframe is \n{current_df}")

                if current_config["env_var"]:
                    del os.environ[key]

                block_output[_id] = current_df

            final_df = block_output["output"]
            out_0 = final_df.values[-1]
            out_1 = np.array(list(final_df.columns)).astype(np.object_)
            output0_dtype = self.output0_dtype
            output1_dtype = self.output1_dtype
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_1.astype(output1_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_0.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)

            return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
