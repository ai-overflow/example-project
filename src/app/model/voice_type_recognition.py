######################
#  THIS FILE IS WIP! #
######################

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import librosa
import numpy as np

model_name = 'voice_type_recognition'


def parse_model_http(model_metadata, model_config):
    if len(model_metadata['inputs']) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata['inputs'])))

    if len(model_config['input']) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config['input'])))

    input_metadata = model_metadata['inputs'][0]
    output_metadata = model_metadata['outputs']

    return (input_metadata['name'], output_metadata,
            model_config['max_batch_size'])


def process_data(row):
    y, sr = librosa.load(row, duration=2.97)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    if ps.shape != (128, 128):
        print("Error in File: " + row)
        return []
    return ps


def postprocess(results, output_names, batch_size):
    option_dict = ['male', 'female']
    output_dict = {}
    for output_name in output_names:
        output_dict[output_name] = results.as_numpy(output_name)
        if len(output_dict[output_name]) != 1:
            raise Exception("expected {} results for output {}, got {}".format(
                batch_size, output_name, len(output_dict[output_name])))

    output = {}
    for output_name in output_names:
        for result in output_dict[output_name][0]:
            if output_dict[output_name][0].dtype.type == np.object_:
                cls = "".join(chr(x) for x in result).split(':')
            else:
                cls = result.split(':')
            output[output_name] = [float(cls[0]), option_dict[int(cls[1])]]
    return output


def process_results():
    try:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000",
                                                         verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        return {}

    try:
        model_metadata = triton_client.get_model_metadata(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        return {}

    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        return {}

    input_name, output_metadata, batch_size = parse_model_http(
        model_metadata, model_config)

    test_data = process_data("G:/Dokumente/dl/test_m.wav")
    test_data = np.array([test_data.reshape((128, 128, 1))])

    input_data = [httpclient.InferInput(input_name, test_data.shape, "FP32")]
    input_data[0].set_data_from_numpy(test_data, binary_data=True)
    output_names = [output['name'] for output in output_metadata]

    outputs = []
    for output_name in output_names:
        outputs.append(httpclient.InferRequestedOutput(output_name,
                                                       binary_data=True,
                                                       class_count=1))

    result = triton_client.infer(model_name, input_data, outputs=outputs)

    return postprocess(result, output_names, batch_size)
