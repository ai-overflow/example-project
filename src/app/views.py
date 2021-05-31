import json
from flask import request, abort
from app import app
from app.model.label_image import get_results
import logging


@app.route('/')
def home():
    """ This is the default route which will block any access with http 403 (DENIED)"""
    abort(403)


@app.route('/process/algo/<algo>/', methods=['POST'])
def algo_test(algo):
    """
    This method will request Labels from the Triton Server for a given image.
    If the image is not of type jpg, png or gif it will fail with http status 400 (Bad Request)
    If the Image does not contain algo in its path it will fail with http 400 (Bad Request)
    """
    if request.headers["Content-Type"] not in ["image/jpg", "image/jpeg", "image/png", "image/gif"]:
        logging.warning("Wrong content type: %s" % request.headers["Content-Type"])
        abort(400)
    if algo.lower() not in ["fast", "accurate"]:
        logging.warning("Wrong algo: %s" % algo.lower())
        abort(400)

    results = get_results(request.get_data())

    return json.dumps(results)
