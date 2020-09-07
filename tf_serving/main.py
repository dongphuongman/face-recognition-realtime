import math
import os
import pickle
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, FileVideoStream
from sklearn.svm import SVC

from tf_serving.align_custom import AlignCustom
from tf_serving.facenet_serving import facenet_predict, prewhiten
from tf_serving.mtcnn_serving import detect_face
from tf_serving.utils import *

INPUT_IMAGE_SIZE = 160

CLASSIFIER_PATH = 'models/svm/face_classifier.pkl'

DATASET_PATH = 'Dataset/processed/'

aligner = AlignCustom()


def collect(path, id):
    folder_path = DATASET_PATH + str(id)
    try:
        os.mkdir(folder_path)
    except OSError:
        print("Creation of the directory %s failed" % folder_path)
    else:
        print("Successfully created the directory %s " % folder_path)

    cap = FileVideoStream(path).start()
    while cap.more():
        frame = cap.read()
        if frame is None:
            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')
            continue
        t0 = time.time()
        
        #frame_resized = imutils.resize(frame, width=640)
        area_detect = 900

        top_x = int((frame.shape[0] - area_detect)/2)
        top_y = int((frame.shape[1] - area_detect)/2)
        bottom_x = top_x + area_detect
        bottom_y = top_y + area_detect
        cropped_frame = frame[top_x:bottom_x, top_y:bottom_y, :]
        cv2.rectangle(frame, (top_y, top_x), (bottom_y, bottom_x), (196, 190, 255), 2)

        results, points = detect_face(cropped_frame)
        faces_found = len(results)
        print(faces_found)
        # if found more two face in a frame -> Warning.
        try:
            if faces_found > 1:
                mess = "MORE THAN 2 FACE IN CAM"
                cv2.putText(frame, mess, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (190, 196, 255), 2, cv2.LINE_AA)
            elif faces_found > 0:
                index = index_largestbb(results)
                bbox = results[index]
                confidence = bbox[4] * 100.0
                bbox = bbox.astype(int)
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                if confidence < 99 or h < 160 or w < 160:
                    frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')
                    continue

                cropped_face = cropped_frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                #aligned_frame, pos = aligner.align(INPUT_IMAGE_SIZE, frame, points[:,index])

                scaled = cv2.resize(cropped_face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(
                    frame, (top_y+bbox[0], top_x+bbox[1]), (top_y+bbox[2], top_x+bbox[3]), (0, 255, 0), 2)
                filename = DATASET_PATH + \
                    str(id) + time.strftime("/%d%m%y-") + \
                    str(time.time())[-4:] + ".jpg"
                if cv2.Laplacian(scaled, cv2.CV_64F).var() > 400:
                    print(filename)
                    cv2.imwrite(filename, scaled)
            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()

        except Exception as e:
            print(str(e))
            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()
        print(time.time() - t0)    
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')


def recog(path):

    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        svm_model, class_names = pickle.load(file)
    # create empty nested dict
    p_dict = {}
    for name in class_names:
        p_dict[name] = {}
        p_dict[name]['count'] = 0

    cap = FileVideoStream(path).start()

    t0 = time.time()
    while cap.more():
        frame = cap.read()
        
        if frame is None:
            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')
            continue
        
               
        frame_resized = imutils.resize(frame, width=720)
        result, points = detect_face(frame_resized)
        faces_found = len(result)
        ratio = frame.shape[0]/frame_resized.shape[0]
        result *= ratio
        points *= ratio
        try:
            if faces_found > 0:
                for index, bbox in enumerate(result):
                    confidence = bbox[4] * 100.0
                    bbox = bbox.astype(int)
                    cropped_face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                    if confidence < 99 or cv2.Laplacian(cropped_face, cv2.CV_64F).var() < 250:
                        frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()
                        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')
                        continue

                    #aligned_face, pos = aligner.align(INPUT_IMAGE_SIZE,frame, points[:,index])
                    scaled = cv2.resize(cropped_face, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)

                    emb_array = facenet_predict(scaled)


                    # SVM classifier
                    predictions = svm_model.predict_proba(emb_array)

                    # print(predictions)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]

                    # Get highest probability with name
                    best_name = class_names[best_class_indices[0]]
                    time_attend = time.time()

                    # Draw bounding box
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                    # print(cropped.shape)
                    # If probability > 0.4 then save img and write log
                    if best_class_probabilities > 0.50:
                        name = class_names[best_class_indices[0]]
                        p_dict[name]['proba'] = best_class_probabilities
                        p_dict[name]['time'] = time_attend
                        p_dict[name]['count'] += 1
                    else:
                        name = "Unknown"

                    text_x = bbox[0]
                    text_y = bbox[3] + 20
                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                                0.8, (0, 0, 255), thickness=1, lineType=2)
                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 20),
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), thickness=1, lineType=1)

            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()

        except Exception as e:
            print(str(e))
            frameEncode = cv2.imencode('.jpg', frame)[1].tobytes()

        if time.time() - t0 > 10:
            postrequest(p_dict)
        print(time.time() - t0)
        t0 = time.time()

        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frameEncode + b'\r\n')


def train_classifier():
    from sklearn.utils import shuffle
    try:
        dataset = get_dataset(DATASET_PATH)
        for clas in dataset:
            assert(len(clas.image_paths) > 0,
                   'There must be at least one image for each class in the dataset')

        paths, labels = get_image_paths_and_labels(dataset)
        paths, labels = shuffle(paths, labels)
        print('Number of labels: %d' % len(set(labels)))
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        print('Calculating features for images')
        embedding_size = 512
        batch_size = 256
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = load_data(paths_batch)
            emb_array[start_index:end_index, :] = facenet_predict(images)

        classifier_filename_exp = os.path.expanduser(CLASSIFIER_PATH)

        # Train classifier
        print('Training classifier')
        model = SVC(kernel='linear', probability=True, gamma='scale')
        model.fit(emb_array, labels)

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    except Exception as e:
        print(str(e))

    return 'OK'


