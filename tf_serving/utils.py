import numpy as np
import os
import cv2
import requests
import json

INPUT_IMAGE_SIZE = 160


def postrequest(p_dict):
    try:
        messages = []
        checkInTime = ''
        for person in p_dict:
            #checkInTime = str(p_dict[person]['time']).split('.')[0]
            if p_dict[person]['count'] > 10:
                message = {
                    'employee': person,
                    'event': 'checkIn',
                    'time': p_dict[person]['time']
                }
                messages.append(message)
            p_dict[person]['count'] = 0
        print(messages)
        headers = {'api-key': '1b06ea1c-008f-4835-a386-3aaa653f40bb'}

        r = requests.post('http://api.helisoft.com.vn/event/attendance',
                          data=json.dumps(messages), headers=headers)
    except:
        x = 1


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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(str(img.shape) + " " + str(image_paths[i]))
        if do_prewhiten:
            img = prewhiten(img)
        images[i, :, :, :] = img
    return images
