"""
Simple backend post endpoint to get model inferences
"""

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# import test.preprocess as ppr
# import test.verification as vs
# import test.audio_record_enroll as enr
# import test.audio_record_enroll_record_infer as inf
# from test.config import get_config

import os
import flask
import tensorflow as tf
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context, request



####################
#    APP
####################
app = Flask(__name__)
socketio = SocketIO(app) #turn the flask app into a socketio app

@app.route('/')
def index():
    return render_template('index.html')



####################
# Interface
####################

@socketio.on('connect', namespace='/test')
def test_connect():
    print('Client connected')

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')

@socketio.on('req',namespace='/test')
def handleReq(data):
    reqType = data['type']
    print('[req] Got request {}'.format(reqType))


####################
# Main
####################

if __name__ == '__main__':
    socketio.run(app,host='0.0.0.0',port=8081, debug=True)
