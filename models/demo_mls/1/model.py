import json
import os
import pickle
import triton_python_backend_utils as pb_utils

import numpy as np
from loguru import logger

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
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

    def load_config_from_minio(self, execution_id: str):
      """
      This is only an example and will read from inside the repo
      """
      path = f"/opt/tritonserver/triton-demo-r21.08/sample/{execution_id}"
      list_blocks = list(os.walk(path))
      list_blocks = {i: os.path.join(path, i) for i in list_blocks[0][1]}
      return list_blocks


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
        model_config = json.loads(args["model_config"])
        block_execution_id = model_config["default_model_filename"]

        # Load config from postgres
        postgres_config = {
          "topo_ids": ["input", "block_1", "block_2", "block_3", "output"],
          "blocks": {
            "input": {
              "max_historical_days": 0,
            }, "block_1": {
              "max_historical_days": 1,
            }, "block_2": {
              "max_historical_days": 0,
            }, "block_3": {
              "max_historical_days": 2,
            }, "output": {
              "max_historical_days": 0,
            }
          }
        }

        blocks = self.load_config_from_minio(block_execution_id)
        logger.info(blocks)
        self.block_config = {}
        for block_id, path in blocks.items():
          self.block_config[block_id] = {}
          config_file = os.path.join(path, "config.json")
          logger.info(config_file)
          with open(config_file, "r") as f:
            config = json.load(f)
          input_columns = config["name_mapping"].keys()
          self.block_config[block_id]["config"] = config
          self.block_config[block_id]["input_columns"] = input_columns

          code_path = os.path.join(path, "code.py")
          if os.path.isfile(code_path):
            code_module = self.read_function_from_file(code_path)
            self.block_config[block_id]["module"] = code_module

          model_path = os.path.join(path, "model.pickle")
          if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
              model = pickle.load(f)
            self.block_config[block_id]["model"] = model
        logger.info(self.block_config)
        self.output0_dtype = np.float64

    def execute(self, requests):
      responses = []
      data = requests[0]
      logger.info(data.inputs())
      for request in requests:
          # Get INPUT0
          in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
          in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")
          logger.info(dir(in_0))
          logger.info(in_0.as_numpy())
          logger.info(in_1.as_numpy())


          # Create output tensors. You need pb_utils.Tensor
          # objects to create pb_utils.InferenceResponse.
          out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                          out_0.astype(output0_dtype))
          out_tensor_1 = pb_utils.Tensor("OUTPUT0",
                                          out_0.astype(output0_dtype))

          # Create InferenceResponse. You can set an error here in case
          # there was a problem with handling this inference request.
          # Below is an example of how you can set errors in inference
          # response:
          #
          # pb_utils.InferenceResponse(
          #    output_tensors=..., TritonError("An error occured"))
          inference_response = pb_utils.InferenceResponse(
              output_tensors=[out_tensor_0])
          responses.append(inference_response)

      # You should return a list of pb_utils.InferenceResponse. Length
      # of this list must match the length of `requests` list.
      return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
