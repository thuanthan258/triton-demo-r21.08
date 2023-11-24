import json
import os
import pickle
import triton_python_backend_utils as pb_utils
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

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

    def get_timeseries_data(self, feats: list,num_historical_days: int):
      return pd.DataFrame([[1 for i in range(len(feats))] for j in range(num_historical_days)], columns=feats)

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
        logger.info(f"The data is {help(pb_utils.InferenceRequest)}")
        model_config = json.loads(args["model_config"])
        config_path = Path(__file__).resolve().parent
        config_file = os.path.join(config_path, "config/config.json")
        logger.info(config_file)
        with open(config_file, "r") as f:
            config = json.load(f)
        input_columns = config["name_mapping"].keys()
        self.block_config = {}
        self.block_config["input_columns"] = input_columns
        self.block_config["max_historical_days"] = config["max_historical_days"]
        
        code_path = os.path.join(config_path, "config/code.py")
        if os.path.isfile(code_path):
            code_module = self.read_function_from_file(code_path)
            self.block_config["module"] = code_module
        else:
            self.block_config["module"] = None

        model_path = os.path.join(config_path, "config/model.pickle")
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.block_config["model"] = model
        else:
            self.block_config["model"] = None
        self.output0_dtype = np.float64
        self.output1_dtype = np.object_

    def execute(self, requests):
      responses = []
      data = requests[0]
      logger.info(data.inputs())
      for request in requests:
        # Get INPUT0
        in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
        in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
        logger.info(dir(in_0))

        cols = [i.decode("ascii") for i in in_0.as_numpy()]
        feats = [in_1.as_numpy()]

        df = pd.DataFrame(feats, columns=cols)
        logger.info(df)
        current_df = df.copy()

        historical_days = self.block_config["max_historical_days"]
        if historical_days != 0:
            ts_df = self.get_timeseries_data(feats=cols, num_historical_days=historical_days)
            current_df = pd.concat([ts_df, current_df], axis=0)
            logger.info(f"DF with ts: \n{current_df}")
          
        input_columns = self.block_config["input_columns"]
        module = self.block_config["module"]
        model = self.block_config["model"]
        if module:
            if model:
                current_df = module(current_df, model)
                logger.info("Transformed with model")
            else:
                current_df = module(current_df)
            current_df = current_df["data"][-1:]
            logger.info(f"DF after transform: \n{current_df}")
        logger.info(current_df)
    
        out_0 = current_df.values[-1]
        out_1 = np.array(list(current_df.columns)).astype(np.object_)
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
        print('Cleaning up...')
