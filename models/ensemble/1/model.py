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
        self.output0_dtype = np.float64
        self.output1_dtype = np.object_

    def execute(self, requests):
        responses = []
        for request in requests:
            model_name = ["block_1", "block_2"]
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
            
            in_0_cur = in_0 
            in_1_cur = in_1

            for model_name_string in model_name: 
                infer_request = pb_utils.InferenceRequest(
                    model_name=model_name_string,
                    requested_output_names=["OUTPUT0", "OUTPUT1"],
                    inputs=[in_0_cur, in_1_cur]
                )

                # Perform synchronous blocking inference request
                infer_response = infer_request.exec()
                logger.info("Done exec!")
                logger.info(infer_response.output_tensors())
                # Make sure that the inference response doesn't have an error. If
                # it has an error, raise an exception.
                if infer_response.has_error():
                    raise pb_utils.TritonModelException(
                        infer_response.error().message())
                output_tensors = infer_response.output_tensors()
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=output_tensors)

                out_0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
                out_1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT1")

                in_0_cur = pb_utils.Tensor("INPUT0", out_0.as_numpy().astype(np.object_))
                in_1_cur = pb_utils.Tensor("INPUT1", out_1.as_numpy().astype(np.float64))

                responses = [inference_response]
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
