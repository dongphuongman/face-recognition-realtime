import json
import math
import os
import pickle

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from torch.autograd import Variable
from torchvision.transforms import ToTensor

from models.torchmodel.model_irse import IR_50
from feature_extraction import extract_feature

DATASET_PATH = 'Dataset/processed'
CLASSIFIER_PATH = 'face_classifier_torch2.pkl'
INPUT_IMAGE_SIZE = 112

trained_model = IR_50((112, 112))
trained_model.load_state_dict(torch.load(
    'models/torchmodel/backbone_ir50_asia.pth'))
trained_model.eval()

if torch.cuda.is_available():
    trained_model.cuda()
    torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def resnet50(images):
    images = torch.from_numpy(images).float().to(device)
    images = images.permute(0, 3, 1, 2)
    images = Variable(images)
    features = l2_norm(trained_model(images.to(device)).cpu())
    return features.detach().numpy()


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def train_classifier():
    try:
        dataset = get_dataset(DATASET_PATH)
        paths, labels = get_image_paths_and_labels(dataset)
        paths, labels = shuffle(paths, labels)
        print('Number of labels: %d' % len(set(labels)))
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Calculating features for images')
        embedding_size = 512
        batch_size = 32
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = load_data(paths_batch)
            emb_array[start_index:end_index, :] = resnet50(images)

        classifier_filename_exp = os.path.expanduser(CLASSIFIER_PATH)

        # Train classifier
        print('Training classifier')
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        svm = SVC(kernel='linear', probability=True)
        #knn.fit(emb_array, labels)
        svm.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((svm, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    except Exception as e:
        print(str(e))

    return 'OK'


def test_recog():
    with open(CLASSIFIER_PATH, 'rb') as file:
        svm_model, class_names = pickle.load(file)
    dataset = get_dataset('Dataset/test')
    paths, labels = get_image_paths_and_labels(dataset)
    for i in range(len(paths)):
        img = cv2.imread(paths[i])
        img = img[...,::-1]
        emb_array = extract_feature(img, trained_model)
        predictions = svm_model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)[0]
        best_name = class_names[best_class_indices]
        print(best_name, labels[i])

def transfrom(name):

    punctuation = [',', '\"', '\'', '?', ':', '!', ';']
    for p in punctuation:
        name = name.replace(p, '-')

    return name


def index_largestbb(result):
    return np.argmax((result[:, 3] - result[:, 1]) * (result[:, 2] - result[:, 0]))


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def load_data(image_paths, image_size=INPUT_IMAGE_SIZE, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = cv2.imread(image_paths[i])
        resized = cv2.resize(img, (128, 128))
        # center crop image
        a=int((128-112)/2) # x start
        b=int((128-112)/2+112) # x end
        c=int((128-112)/2) # y start
        d=int((128-112)/2+112) # y end
        ccropped = resized[a:b, c:d] # center crop the image
        ccropped = ccropped[...,::-1] # BGR to RGB
        #print(str(img.shape) + " " + str(image_paths[i]))
        if do_prewhiten:
            ccropped = prewhiten(ccropped)
        images[i, :, :, :] = ccropped
    return images


if __name__ == '__main__':
    #train_classifier()
    test_recog()
