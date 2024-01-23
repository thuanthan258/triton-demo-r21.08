from tritonclient.utils import *
import tritonclient.http as httpclient
import json
import numpy as np

model_name = "model-7991c0c2-7543-43b6-a49a-4a180c981a73-backward"
shape = [5]

with httpclient.InferenceServerClient("localhost:8000") as client:
    feates = [
        "feat12_22",
        "feat1",
        "feat10",
        "feat10_1",
        "feat10_2",
        "feat10_3",
        "feat11",
        "feat11_1",
        "feat11_2",
        "feat11_3",
        "feat1_1",
        "feat1_2",
        "feat1_3",
        "feat2",
        "feat2_1",
        "feat2_2",
        "feat2_3",
        "feat3",
        "feat3_1",
        "feat3_2",
        "feat3_3",
        "feat4",
        "feat4_1",
        "feat4_2",
        "feat4_3",
        "feat5",
        "feat5_1",
        "feat5_2",
        "feat5_3",
        "feat6",
        "feat6_1",
        "feat6_2",
        "feat6_3",
        "feat7",
        "feat7_1",
        "feat7_2",
        "feat7_3",
        "feat8",
        "feat8_1",
        "feat8_2",
        "feat8_3",
        "feat9",
        "feat9_1",
        "feat9_2",
        "feat9_3",
    ]
    all_val = [2.0 for i in range(len(feates))]
    input0_data = np.array(["data_key", "mode", "serving_timestamp"] + feates).astype(
        np.object_
    )

    values = ["chiller_test_7", "backward", 1577853940] + all_val
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
