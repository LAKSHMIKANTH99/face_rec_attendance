import os
import cv2
import dlib
import numpy as np
import json


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def compute_face_encoding(image):
    faces = detector(image, 1)
    if len(faces) > 0:
        shape = predictor(image, faces[0])
        return np.array(face_rec_model.compute_face_descriptor(image, shape))
    else:
        return None


dataset_path = 'pics'

known_faces = []


for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        encoding = compute_face_encoding(rgb_image)

        if encoding is not None:
            name = os.path.splitext(filename)[0]
            known_faces.append({"name": name, "encoding": encoding.tolist()})


with open('known_faces.json', 'w') as file:
    json.dump(known_faces, file)
