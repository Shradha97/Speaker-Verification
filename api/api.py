"""
Simple backend post endpoint to get model inferences
"""

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
import tensorflow as tf
from config import get_config

import test.preprocess as ppr
import test.verification as vs
import test.audio_record_enroll as enr
import test.audio_record_enroll_record_infer as inf

import flask
from flask import request

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['POST'])
def home():
	assert(request.method == 'POST')
	return request.form

app.run()
