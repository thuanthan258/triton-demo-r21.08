from tritonclient.utils import *
import tritonclient.http as httpclient

import numpy as np

model_name = "model-1ade3538-3ede-4db2-b504-ed4289ed7e4e"
shape = [5]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.array(["data_key", "mode", "Timestamp", "feat1", "feat2"]).astype(np.object_)
    values = ["chiller_test_7", "forward", 1577852940 + 1000, 0.1, 0.2]
    values = [str(i) for i in values]
    input1_data = np.array(values).astype(np.object_)
    print(input1_data)
    inputs = [
        httpclient.InferInput(
            "INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)
        ),
        httpclient.InferInput(
            "INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)
        ),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    columns = [i.decode('utf-8') for i in response.as_numpy("OUTPUT0")]
    features = response.as_numpy("OUTPUT1")
    print(columns)
    print(features)