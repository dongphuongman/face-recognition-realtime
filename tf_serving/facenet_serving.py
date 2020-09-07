import cv2
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.python.framework.tensor_util import MakeNdarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import time

from face_attendance.utils import prewhiten

def facenet_predict(image):

    options = [('grpc.max_send_message_length', 1000331268),('grpc.max_receive_message_length', 555558316)]
    channel = grpc.insecure_channel("0.0.0.0:8500", options = options)

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = "facenet"
    request.model_spec.signature_name = "serving_default"

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
        image = prewhiten(image)
    
    tp = tf.make_tensor_proto(image, dtype=tf.float32, shape=image.shape)

    request.inputs["input_1"].CopyFrom(tp)
    result = stub.Predict(request, 10.0)

    return MakeNdarray(result.outputs['normalize'])

if __name__ == '__main__':
    image = cv2.imread('FaceRecog/collect/135202020-08-07 16-21-25.018098.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    emb_array = facenet_predict(image)
    print(emb_array.shape)