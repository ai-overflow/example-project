from app import app
from flask import request
import json

@app.route('/')
def home():
   return "hello world!"


@app.route('/image/test/<algo>/', methods=['POST'])
def algo_test(algo):
   return "You have requested this page and expected {algo:}\nHeaders: {headers:}".format(algo = algo, headers = json.dumps(dict(request.headers)))