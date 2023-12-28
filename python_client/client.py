from tritonclient.utils import *
import tritonclient.http as httpclient

import numpy as np

model_name = "final_model"
shape = [3]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.array(["Timestamp", "feat1", "feat2"]).astype(np.object_)
    input1_data = np.array([1577852940 + 1000, 0.1, 0.2]).astype(np.float64)
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
    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, response.as_numpy("OUTPUT0")
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, response.as_numpy("OUTPUT1")
        )
    )
