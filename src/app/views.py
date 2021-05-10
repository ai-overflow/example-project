import json
from flask import request, abort
from app import app
from app.model.label_image import get_results
import logging


@app.route('/')
def home():
    abort(403)


@app.route('/process/algo/<algo>/', methods=['POST'])
def algo_test(algo):
    if request.headers["Content-Type"] not in ["image/jpg", "image/jpeg", "image/png", "image/gif"]:
        logging.warning("Wrong content type: %s" % request.headers["Content-Type"])
        abort(400)
    if algo.lower() not in ["fast", "accurate"]:
        logging.warning("Wrong algo: %s" % algo.lower())
        abort(400)

    get_results(request.get_data())


    return "You have requested this page and expected {algo:}\nHeaders: {headers:}".format(algo=algo,
                                                                                           headers=json.dumps(
                                                                                               dict(request.headers)))
