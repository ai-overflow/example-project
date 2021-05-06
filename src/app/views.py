import json
from flask import request, abort
from app import app


@app.route('/')
def home():
    abort(403)


@app.route('/process/algo/<algo>/', methods=['GET'])
def algo_test(algo):
    if request.headers["Content-Type"] not in ["image/jpg", "image/png", "image/gif"]:
        abort(400)
    if algo not in ["Fast", "Accurate"]:
        abort(400)

    data = request.get_data()
    return "You have requested this page and expected {algo:}\nHeaders: {headers:}".format(algo=algo,
                                                                                           headers=json.dumps(
                                                                                               dict(request.headers)))
