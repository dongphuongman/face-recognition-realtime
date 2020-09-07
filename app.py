# import necessary lib
import base64
import datetime
import json
import pickle
import time

import cv2
import imutils
import numpy as np
import requests
import tensorflow as tf
#import flask
from flask import Flask, Response, request
# imutils
from imutils.video import FPS, FileVideoStream

from tf_serving.main import collect, recog, train_classifier

app = Flask(__name__)



@app.route('/test', methods=['GET', 'POST'])
@app.route('/test/<path><id>', methods=['GET', 'POST'])
def TestApi(path=None):
    return 'Hello World' + path


@app.route('/face-collect/', methods=['GET'])
def CollectApi():
    # return Response(gen(request.args.get('path'), 'collect'), mimetype='multipart/x-mixed-replace; boundary=frame')
    if request.method == 'GET':
        #path = request.args.get('path')
        path = "VID_20200906_181822.mp4" #"http://192.168.1.23:8888/video"
        employ_id = request.args.get('id')
    employ_id = int(employ_id)
    return Response(collect(path, employ_id), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/face-recog/', methods=['GET', 'POST'])
def RecogApi():
    if request.method == 'GET':
        path = request.args.get('path')
    if request.method == 'POST':
        path = request.form['path']
    return Response(recog(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train-classifier', methods=['GET', 'POST'])
def TrainApi():
    return train_classifier()

if __name__ == '__main__':


    app.run(debug=True, host='localhost', port=5000)
    #train_classifier()
