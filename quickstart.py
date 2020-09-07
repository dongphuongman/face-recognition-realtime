"""
This code is used to batch detect images in a folder.
"""
import argparse
import os
import pickle
import sys
import time

import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imutils.video import FPS, FileVideoStream

from feature_extraction import extract_feature
from models.torchmodel.model_irse import IR_50
from vision.ssd.config.fd_config import define_img_size
from vision.ssd.mb_tiny_fd import (create_mb_tiny_fd,
                                   create_mb_tiny_fd_predictor)
from vision.ssd.mb_tiny_RFB_fd import (create_Mb_Tiny_RFB_fd,
                                       create_Mb_Tiny_RFB_fd_predictor)

CLASSIFIER_PATH = 'models/svm/face_classifier_torch.pkl'

parser = argparse.ArgumentParser(
    description='detect_imgs')
parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=640, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.8, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1500, type=int,
                    help='nms candidate size')
parser.add_argument('--path', default="http://192.168.43.1:8888/video", type=str,
                    help='video dir')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--classifier_path', default='models/svm/face_classifier_torch.pkl', type=str,
                    help='path to svm classifier')
parser.add_argument('--emb_model', default='models/torchmodel/backbone_ir50_asia.pth', type=str,
                    help='dir pth model')
args = parser.parse_args()
# must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'
define_img_size(args.input_size)

# Face Detect
label_path = "./models/voc-model-labels.txt"
test_device = args.test_device
class_names = [name.strip() for name in open(label_path).readlines()]
if args.net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    # model_path = "models/pretrained/version-slim-640.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    face_detect = create_mb_tiny_fd_predictor(
        net, candidate_size=args.candidate_size, device=test_device)
elif args.net_type == 'RFB':
    #model_path = "models/pretrained/version-RFB-320.pth"
    model_path = "models/pretrained/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(
        len(class_names), is_test=True, device=test_device)
    face_detect = create_Mb_Tiny_RFB_fd_predictor(
        net, candidate_size=args.candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)
net.load(model_path)

# Face Embedding
face_emb = IR_50((112, 112))
face_emb.eval()
face_emb.load_state_dict(torch.load(args.emb_model))
if torch.cuda.is_available():
    face_emb.cuda()
    torch.cuda.empty_cache()
# SVM load
with open(args.classifier_path, 'rb') as file:
    svm_model, class_names = pickle.load(file)
time.sleep(2.0)
print("loaded")

cap = FileVideoStream(args.path).start()
fps = FPS().start()

while cap.more():
    t0 = time.time()
    frame = cap.read()
    if frame is None:
        continue
    frame = imutils.resize(frame, width=720)
    boxes, labels, probs = face_detect.predict(
        frame, args.candidate_size / 2, args.threshold)
    print(boxes.size(0))
    for i in range(boxes.size(0)):
        try:
            box = boxes[i, :]
            box = box.numpy()
            box = box.astype(int)
            cv2.rectangle(frame, (box[0], box[1]),
                        (box[2], box[3]), (0, 0, 255), 2)

            cropped_face = frame[box[1]:box[3], box[0]:box[2]]

            emb_array = extract_feature(cropped_face, face_emb)
            predictions = svm_model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]

            if best_class_probabilities > 0.50:
                name = class_names[best_class_indices[0]]
            else:
                name = "Unknown"
        
            cv2.putText(frame, "ID: " + name, (box[0], box[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

            fpstext = "Inference speed: " + str(1/(time.time() - t0))[:5] + " FPS"
            cv2.putText(frame, fpstext,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 180, 255), 2)
        except:
            continue
    fps.update()
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.stop()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] FPS: {:.2f}".format(fps.fps()))
