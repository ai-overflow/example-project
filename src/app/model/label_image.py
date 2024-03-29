import pickle
import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from tritonclient.utils import InferenceServerException
from io import BytesIO
import os


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


def postprocess_image(results, output_names, fac):
    """
    Post process a Triton Server label output into a json output.

    Parameter
    ---------
    result:
        Triton results from server request
    output_names: list
        List of all outputs from the model
    fac: dict
        Lookup table to fit an output int to a label

    Returns
    -------
    dict:
        Dictionary containing the label name and the percentage to fit the given label
    """
    output_dict = {}
    for output_name in output_names:
        output_dict[output_name] = results.as_numpy(output_name)

    output = {}
    for output_name in output_names:
        for result in output_dict[output_name]:
            pred = np.argsort(-result)
            labels = []
            for i in pred:
                labels.append({'name': fac[i], 'certainty': result[i] * 100})
            output[output_name] = labels
    return output


def triton_process(data, model_name):
    """
    This method will send data to the Triton model model_name.\n
    The Request will be made via HTTP.

    Parameters
    ----------
    data: ndarray
        File data which will be passed to triton. Must be FP32 numpy array
    model_name: str
        Triton model name, e.g. if your model is in the folder 'models/abc' this should be 'abc'

    Returns
    -------
    Any, list:
        result: Triton results from server request.
        output_names: List of all outputs from the model
    """
    try:
        triton_client = httpclient.InferenceServerClient(url="triton:8000",
                                                         verbose=False)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit(1)

    # if triton is not in default mode load models:
    #
    # UNCOMMENT
    # triton_client.unload_model(model_name=model_name)
    # if not triton_client.is_model_ready(model_name=model_name):
    #    try:
    #        triton_client.load_model(model_name=model_name)
    #    except InferenceServerException as e:
    #        print("failed to load model: " + str(e))
    #        sys.exit(1)
    # END UNCOMMENT

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
    input_name, output_metadata, batch_size = parse_model_http(model_metadata, model_config)
    input_data = [httpclient.InferInput(input_name, data.shape, "FP32")]
    input_data[0].set_data_from_numpy(data, binary_data=True)
    output_names = [output['name'] for output in output_metadata]
    outputs = []
    for output_name in output_names:
        outputs.append(httpclient.InferRequestedOutput(output_name,
                                                       binary_data=True))
    result = triton_client.infer(model_name, input_data, outputs=outputs)

    return result, output_names


def get_results(image_data):
    """
    This method will request labels from triton for a given image.\n
    It will down sample the image to 32x32px float32 RGB.\n
    This method needs a label classification file (data/img_classification.pickle)

    Parameters
    ----------
    image_data: bytes
        Generate labels for this image
    """
    name = 'image_label'
    img = np.asarray([np.array(
        Image.open(BytesIO(image_data))
            .convert('RGB')
            .resize((32, 32), Image.ANTIALIAS)
    )], dtype="float32")

    result, output_names = triton_process(img, name)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fac = pickle.load(open(str(os.path.join(dir_path, "data/img_classification.pickle")), "rb"))
    return postprocess_image(result, output_names, fac)
