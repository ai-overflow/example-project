import argparse
import sys

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import librosa
import numpy as np
from PIL import Image
import pickle


def parse_model_http(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
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

def postprocess_image(results, output_names, fac):
    output_dict = {}
    for output_name in output_names:
        output_dict[output_name] = results.as_numpy(output_name)

    output = {}
    for output_name in output_names:
        for result in output_dict[output_name]:
            pred = np.argsort(-result)
            labels = []
            for i in pred:
                labels.append("{} - {}%".format(fac[i], round(result[i] * 100, 2)))
            output[output_name] = labels
    return output


def triton_process(data, model_name):
    try:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000",
                                                         verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)
    triton_client.unload_model(model_name=model_name)
    if not triton_client.is_model_ready(model_name=model_name):
        try:
            triton_client.load_model(model_name=model_name)
        except InferenceServerException as e:
            print("failed to load model: " + str(e))
            sys.exit(1)
    try:
        model_metadata = triton_client.get_model_metadata(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the metadata: " + str(e))
        sys.exit(1)
    try:
        model_config = triton_client.get_model_config(model_name=model_name)
    except InferenceServerException as e:
        print("failed to retrieve the config: " + str(e))
        sys.exit(1)
    input_name, output_metadata, batch_size = parse_model_http(
        model_metadata, model_config)
    input_data = [httpclient.InferInput(input_name, data.shape, "FP32")]
    input_data[0].set_data_from_numpy(data, binary_data=True)
    output_names = [output['name'] for output in output_metadata]
    outputs = []
    for output_name in output_names:
        outputs.append(httpclient.InferRequestedOutput(output_name,
                                                       binary_data=True))
    result = triton_client.infer(model_name, input_data, outputs=outputs)

    return result, output_names


def get_results():
    name = 'image_label'
    img = np.asarray([np.array(
        Image.open("D:/Seafile/Main/Main2/Bilder/tmp/IMG_20200925_100253.jpg")
            .convert('RGB')
            .resize((32, 32), Image.ANTIALIAS)
    )], dtype="float32")
    result, output_names = triton_process(img, name)

    fac = pickle.load(open("data/img_classification.pickle", "rb"))
    return postprocess_image(result, output_names, fac)
