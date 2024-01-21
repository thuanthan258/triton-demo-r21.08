from tritonclient.utils import *
import tritonclient.http as httpclient
import json
import numpy as np

model_name = "model-805d47a6-4015-434d-a4b6-bdaadb53c132-forward"
shape = [5]

with httpclient.InferenceServerClient("localhost:8000") as client:
    feates = [f"feat{i}" for i in range(1, 13)]
    all_val = [2 for i in range(len(feates))]
    input0_data = np.array(
        ["data_key", "mode", "Timestamp", "serving_timestamp"] + feates
    ).astype(np.object_)

    values = ["chiller_test_7", "forward", 1577853940, 1577853940] + all_val
    values = [str(i) for i in values]
    input1_data = np.array(values).astype(np.object_)

    name_mapping = {i: i for i in feates}
    name_mapping = np.array([json.dumps(name_mapping)]).astype(np.object_)

    print(input1_data)
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
        httpclient.InferInput(
            "NAME_MAPPING", name_mapping.shape, np_to_triton_dtype(name_mapping.dtype)
        ),
    ]
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)
    inputs[2].set_data_from_numpy(name_mapping)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    columns = [i.decode("utf-8") for i in response.as_numpy("OUTPUT0")]
    features = response.as_numpy("OUTPUT1")
    print(columns)
    print(features)
