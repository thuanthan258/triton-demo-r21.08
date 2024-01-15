import json
import logging
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils
from pathlib import Path
import os
from datetime import timedelta
from loguru import logger

from typing import Optional
from datetime import datetime
from pydantic_settings import BaseSettings


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


import os
import yaml
import pandas as pd

from typing import Dict
from mls_ml_libs.db.timeseries import TimeseriesDBClient
from data_processing_libs.transforms import BaseTransform
import ast

CONFIG_FILE_NAME = "graph_configs.yaml"
ROOT_DIR = os.getcwd()

import pandas as pd
import importlib


def read_class_from_file(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the class from the module
    custom_class = getattr(module, class_name)
    return custom_class


def get_previous_timestamp(timestamp, nb_previous_periods: int, data_unit: str):
    if data_unit == "second":
        delta = timedelta(seconds=nb_previous_periods)
    elif data_unit == "minute":
        delta = timedelta(minutes=nb_previous_periods)
    elif data_unit == "hour":
        delta = timedelta(hours=nb_previous_periods)
    elif data_unit == "day":
        delta = timedelta(days=nb_previous_periods)
    elif data_unit == "week":
        delta = timedelta(weeks=nb_previous_periods)
    else:
        raise ValueError("Invalid data_unit")

    previous_timestamp = timestamp - int(delta.total_seconds())
    return previous_timestamp


def select_columns_with_filter(dict_dfs, filters, logging=logging):
    result_df = pd.DataFrame()
    for df_name, df in dict_dfs.items():
        if df_name in filters:
            filter_dict = filters[df_name]["mappings"]
            # print(df)
            logging.info(f"[KEYS] {filter_dict.keys()}")
            df_with_cols = df[list(filter_dict.keys())].copy()
            filtered_df = df_with_cols.rename(columns=filter_dict)
            result_df = pd.concat([result_df, filtered_df], axis=1)
    return result_df


from mls_ml_libs.oauth2.auth import MLSOAuth2Client
from mls_ml_libs.cache.redis_client import RedisClient
from typing import Dict, List
import inspect
from urllib.parse import urljoin
from pydantic_settings import BaseSettings
import requests

logger = logging.getLogger(__name__)

QUERY_DATA_PATH = "tms/TimeSeries/query"
REGISTER_METRICS_PATH = "tms/TimeSeries/register-metrics-labels"
INSERT_DATA_PATH = "tms/TimeSeries/metrics"


class TimeseriesDBClient(object):
    """
    Client for interacting with a time series database.
    """

    def __init__(
        self,
        x_tenant_id: str,
        x_subscription_id: str,
        settings: BaseSettings,
    ) -> None:
        """
        Initialize the client with tenant ID, subscription ID, and settings.
        """
        # Setup Redis client
        redis_settings = {
            "host": settings.Redis__Host,
            "port": settings.Redis__Port,
            "db": settings.Redis__Database,
            "password": settings.Redis__Password,
            "use_ssl": settings.Redis__Ssl,
            "username": settings.Redis__User,
        }
        self._redis_client = RedisClient(**redis_settings)

        # Setup OAuth2 client
        auth_settings = {
            "client_id": settings.Authentication__ClientId,
            "client_secret": settings.Authentication__ClientSecret,
            "auth_url": settings.Authentication__Authority,
            "redis_client": self._redis_client,
        }
        self._oauth_client = MLSOAuth2Client(**auth_settings)

        # Setup other attributes
        self.api_execute_timeseries_database_url = (
            settings.API__TimeseriesDB__Execute__Url
        )
        self.headers = {
            "Content-Type": "application/json",
            "x-tenant-id": x_tenant_id,
            "x-subscription-id": x_subscription_id,
        }
        self.token = None

    async def __aenter__(self):
        """
        Asynchronously enter the runtime context related to this object.
        """
        self.token = await self._oauth_client.get_token()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Asynchronously exit the runtime context related to this object.
        """
        ...

    async def get_response(self, api_url, payload):
        """
        Send a POST request to the given URL with the given payload, and return the response.
        """
        headers = {**self.headers, "Authorization": f"Bearer {self.token}"}

        response = requests.request(
            "POST",
            api_url,
            headers=headers,
            data=json.dumps(payload),
        )

        # Check for errors
        if response.status_code != 200:
            return {"error": response.text}

        return json.loads(response.text)

    def handle_request_response(self, response_content: List, data_metrics: List):
        """
        Handle the response content from a request.
        """
        try:
            # Check for errors
            if "error" in response_content:
                return response_content

            result = {}
            for i, metric in enumerate(data_metrics):
                result[metric] = [j["value"] for j in response_content[i]["values"]]

            # Convert to DataFrame
            df = pd.DataFrame(result)
            return df

        except Exception as e:
            error_str = f"[TimeseriesDBClient] Error handling request response: {e}"
            logger.error(error_str)
            return {"error": error_str}

    async def get_data_from_db(
        self,
        data_key: str,
        data_metrics: List[str],
        from_timestamp: int,
        to_timestamp: int,
    ):
        """
        Get data from the database for the given key and metrics, between the given timestamps.
        """
        # Get access token
        if not self.token:
            self.token = await self._oauth_client.get_token()

        # Check timestamps
        if from_timestamp > to_timestamp:
            return {"error": "from_timestamp must be less than to_timestamp"}

        # Prepare payload
        payload = {
            "key": data_key,
            "metrics": data_metrics,
            "FromTimestamp": from_timestamp,
            "ToTimestamp": to_timestamp,
        }

        # Send request and handle response
        print(urljoin(self.api_execute_timeseries_database_url, QUERY_DATA_PATH))
        response = await self.get_response(
            api_url=urljoin(self.api_execute_timeseries_database_url, QUERY_DATA_PATH),
            payload=payload,
        )

        if "error" in response or not isinstance(response, list):
            return response

        return self.handle_request_response(response, data_metrics)

    async def register_metrics_key(self, key: str, metrics: List[str]):
        """
        Register a new metrics key.
        """
        # Prepare payload
        payload = {"key": key, "metrics": metrics}

        # Send request and return response
        response = await self.get_response(
            api_url=urljoin(
                self.api_execute_timeseries_database_url, REGISTER_METRICS_PATH
            ),
            payload=payload,
        )
        return response

    async def insert_data_to_db(
        self, data_key: str, data_metrics_dict: Dict[str, float], timestamp: int
    ):
        """
        Insert data into the database.
        """
        # Prepare payload
        payload = {
            "key": data_key,
            "metrics": data_metrics_dict,
            "Timestamp": timestamp,
        }

        # Send request and handle response
        response = await self.get_response(
            api_url=urljoin(self.api_execute_timeseries_database_url, INSERT_DATA_PATH),
            payload=payload,
        )

        if "error" in response or not response.get("isSuccess"):
            return response

        logger.info(f"Data inserted successfully for key: {data_key}")
        return response

    def get_response_sync(self, api_url, payload):
        """
        Send a POST request to the given URL with the given payload, and return the response.
        """
        headers = {**self.headers, "Authorization": f"Bearer {self.token}"}

        response = requests.request(
            "POST",
            api_url,
            headers=headers,
            data=json.dumps(payload),
        )

        # Check for errors
        if response.status_code != 200:
            return {"error": response.text}

        return json.loads(response.text)

    def get_timeseries_data_sync(
        self,
        data_key: str,
        data_metrics: List[str],
        from_timestamp: int,
        to_timestamp: int,
    ):
        """
        Get data from the database for the given key and metrics, between the given timestamps.
        """
        if not self.token:
            self.token = self._oauth_client.get_token()

        # Check timestamps
        if from_timestamp > to_timestamp:
            return {"error": "from_timestamp must be less than to_timestamp"}

        # Prepare payload
        payload = {
            "key": data_key,
            "metrics": data_metrics,
            "FromTimestamp": from_timestamp,
            "ToTimestamp": to_timestamp,
        }
        # Send request and handle response
        print(self.api_execute_timeseries_database_url)
        print(urljoin(self.api_execute_timeseries_database_url, QUERY_DATA_PATH))
        response = self.get_response_sync(
            api_url=urljoin(self.api_execute_timeseries_database_url, QUERY_DATA_PATH),
            payload=payload,
        )

        if "error" in response or not isinstance(response, list):
            return response

        return self.handle_request_response(response, data_metrics)


class GetHistoryData:
    def __init__(
        self,
        num_historical_periods: int,
        data_unit: str,
        timescale_client: TimeseriesDBClient,
        use_cache=True,
    ):
        self.use_cache = use_cache
        self.cache = None
        self.num_historical_periods = num_historical_periods
        self.data_unit = data_unit
        self.timescale_client = timescale_client

    def process_with_cache(self, timestamp):
        """
        Given a timestamp, check the

        :param timestamp:
        :return:
        """
        return

    def query_timeseries_data(
        self,
        feats: list,
        to_timestamp: int,
        data_key: str,
        name_mapping: dict,
        logging: logging,
    ):
        db_feats = [name_mapping[i] for i in feats]
        revert_mapping = {value: key for key, value in name_mapping.items()}

        from_timestamp = get_previous_timestamp(
            to_timestamp, self.num_historical_periods, self.data_unit
        )

        df = self.timescale_client.get_timeseries_data_sync(
            data_key=data_key,
            data_metrics=db_feats,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        logger.info(f"[TIMESERIES DF] {df}")

        df.rename(columns=revert_mapping, inplace=True)

        return df.iloc[-self.num_historical_periods :]


def import_class(module_path, class_name):
    module = importlib.import_module(module_path)
    class_obj = getattr(module, class_name)
    return class_obj


class Node:
    def __init__(
        self,
        id,
        transform_class,
        init_kwargs,
        parents,
        input_features,
        expected_outputs,
        config_path,
    ):
        """
        Initialize a Node object.

        Parameters:
        id (str): The id of the node.
        transform_class (class): The class that performs the transformation.
        parents (list): A list of parent Node objects.
        input_features (dict): A dictionary of input features.
        expected_outputs (list): A list of expected output columns.
        """
        self.id = id
        self.init_kwargs = init_kwargs
        module_path = os.path.join(config_path, "code.py")
        self.transform_class = read_class_from_file(
            module_path=module_path, class_name=transform_class
        )(working_dir=config_path, **self.init_kwargs)
        self.parents = parents  # list of Node objects
        self.input_features = input_features
        self.expected_outputs = expected_outputs
        self.output = None

    def execute(self, dataframe: pd.DataFrame):
        """
        Execute the transformation on the input dataframe.

        Parameters:
        dataframe (pd.DataFrame or dict): The input dataframe or a dictionary of dataframes.

        Returns:
        pd.DataFrame: The output dataframe after transformation.

        """
        if hasattr(self.transform_class, "forward") and callable(
            getattr(self.transform_class, "forward")
        ):
            # Call the 'forward' method
            self.output = self.transform_class.forward(dataframe)
            print("Result:", self.output)
        else:
            print("The 'forward' method does not exist in the class.")

        output_cols = self.output.columns

        if not set(output_cols).issubset(self.expected_outputs):
            raise ValueError(
                "[ERROR] Output data columns do not match: \n ",
                "Output: \n",
                f"{output_cols} \n",
                "Expected Output: \n",
                self.expected_outputs,
            )

        return self.output


class Graph:
    def __init__(self, timeseries_client: TimeseriesDBClient = None):
        self.nodes = {}  # Stores node instances
        self.dependencies = {}  # Maps node names to their dependencies
        self.outputs_dataframes = {}  # Stores the output dataframes of the nodes
        self.topological_order = None  # Set topological order from yaml file
        self.history_data_retriever = None
        self.timeseries_metadata = {}
        self.timeseries_client = timeseries_client

    def initialize(self, config_dir="./"):
        """
        This method initializes the Graph object.

        It loads the configuration file, sets the topological order of the nodes, creates Node objects for each node
        in the configuration, and adds them to the graph.

        If the graph type is 'timeseries', it also sets up a history data retriever and stores the timeseries metadata.
        """

        config_file_path = os.path.join(config_dir, CONFIG_FILE_NAME)

        configs = self.load_yaml_config_file(config_file_path)

        graph_configs = configs.get("graph")
        nodes_configs = configs.get("nodes")

        self.topological_order = graph_configs[0]["topo_ids"]

        for node_config in nodes_configs:
            node = Node(
                id=node_config["id"],
                config_path=os.path.join(config_dir, node_config["id"]),
                init_kwargs=node_config["init_kwargs"],
                transform_class=node_config["transform_class"],
                parents=node_config["parents"],
                input_features=node_config["inputs"],
                expected_outputs=node_config["outputs"],
            )
            self.add_node(node)

        if graph_configs[0]["type"] == "timeseries":
            self.timeseries_metadata = graph_configs[0].get("metadata", {})

            self.history_data_retriever = GetHistoryData(
                num_historical_periods=self.timeseries_metadata.get(
                    "max_historical_days"
                ),
                data_unit=self.timeseries_metadata.get("data_unit"),
                timescale_client=self.timeseries_client,
            )

    def add_node(self, node):
        self.nodes[node.id] = node
        self.dependencies[node.id] = node.parents

    def load_yaml_config_file(self, yaml_file_path):
        with open(yaml_file_path) as f:
            dag = yaml.load(f, Loader=yaml.FullLoader)

        return dag

    def execute(
        self,
        logging,
        input_dataframe: pd.DataFrame,
        name_mapping: Dict = dict,
        to_timestamp: int = None,
        data_key: str = "",
    ):
        """
        Given a self graph, execute the nodes in topological order
        """

        logging.info("[GRAPH] Executing....")
        # if history
        if self.history_data_retriever:
            logging.info("[GRAPH] [TIMESERIES] Querying data....")
            logger.info(f"The timestamp is {to_timestamp}")
            logger.info(f"The data key is {data_key}")
            history_data_df = self.history_data_retriever.query_timeseries_data(
                feats=list(name_mapping.keys()),
                data_key=data_key,
                to_timestamp=to_timestamp,
                name_mapping=name_mapping,
                logging=logging,
            )
            input_dataframe = pd.concat([history_data_df, input_dataframe], axis=0)

        logger.info(f"[INPUT DF] {input_dataframe}")

        self.outputs_dataframes["input"] = input_dataframe

        for node_id in self.topological_order:  # renamed node_name to node_id
            if node_id == "input":
                continue
            logger.info(f"[EXECUTE NODE] - {node_id}")

            # Get node
            node = self.nodes[node_id]

            logger.info(f"INFO: {node.__dict__}")
            logger.info(f"[*] {self.outputs_dataframes}")

            # Get node input df
            node_input_df = select_columns_with_filter(
                self.outputs_dataframes, node.input_features, logging=logging
            )
            logger.info(f"[NODE INPUT DF] {node_input_df}")

            node_result = node.execute(node_input_df)
            logger.info(f"[NODE RESULT] {node_result}")

            self.outputs_dataframes[
                node_id
            ] = node_result  # renamed node_name to node_id

        result_df = self.outputs_dataframes[self.topological_order[-1]]

        return result_df


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
            block_output = {}

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
            # mode = full_map["mode"]
            data_key = full_map["data_key"]

            timestamp = int(full_map["Timestamp"])

            del full_map["mode"]
            del full_map["data_key"]

            columns = list(full_map.keys())
            features = [[float(i) for i in full_map.values()]]

            request_df = pd.DataFrame(features, columns=columns)
            logger.log(f"[request_df]{request_df}")
            current_df = request_df.copy()
            # block_output["input"] = current_df

            logger.log(f"[GRAPH] Initilizing....")

            logging.info("[GRAPH] Initilizing....")
            logging.info(f"[Input DF] {current_df}")

            config_path = str(Path(__file__).resolve().parent)
            self.config_path = os.path.join(config_path, "config/")
            self.graph.initialize(config_dir=self.config_path)

            logger.log(f"[GRAPH] Initilized")

            logger.log(f"[GRAPH] Executing...")

            result_df = self.graph.execute(
                input_dataframe=current_df,
                name_mapping=data_mapping,
                data_key=data_key,
                to_timestamp=timestamp,
                logging=logging,
            )

            logging.info(f"[FINAL RESULT] {result_df}")
            final_df = result_df

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
