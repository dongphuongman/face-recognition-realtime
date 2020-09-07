# Helper function for extracting features from pre-trained models
import torch
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
from torch.autograd import Variable

def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def extract_feature(img, model, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), tta = False):
    # resize image to [128, 128]
    resized = cv2.resize(img, (128, 128))

    # center crop image
    a=int((128-112)/2) # x start
    b=int((128-112)/2+112) # x end
    c=int((128-112)/2) # y start
    d=int((128-112)/2+112) # y end
    ccropped = resized[a:b, c:d] # center crop the image

    # flip image horizontally
    flipped = cv2.flip(ccropped, 1)

    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped).float().to(device)
    ccropped = Variable(ccropped)

    flipped = flipped.swapaxes(1, 2).swapaxes(0, 1)
    flipped = np.reshape(flipped, [1, 3, 112, 112])
    flipped = np.array(flipped, dtype = np.float32)
    flipped = (flipped - 127.5) / 128.0
    flipped = torch.from_numpy(flipped).float().to(device)


    with torch.no_grad():
        if tta:
            emb_batch = model(ccropped.to(device)).cpu() + model(flipped.to(device)).cpu()
            features = l2_norm(emb_batch)
        else:
            features = l2_norm(model(ccropped.to(device)).cpu())
            
#     np.save("features.npy", features) 
#     features = np.load("features.npy")
    return features