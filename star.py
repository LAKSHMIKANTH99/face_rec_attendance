import cv2
import dlib
import numpy as np
import json
from datetime import datetime
import csv


with open('known_faces.json', 'r') as file:
    known_faces = json.load(file)

known_face_encodings = [np.array(entry["encoding"]) for entry in known_faces]
known_face_names = [entry["name"] for entry in known_faces]


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


def compute_face_encoding(image, face):
    shape = predictor(image, face)
    return np.array(face_rec_model.compute_face_descriptor(image, shape))


def find_match(face_encoding):
    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
    best_match_index = np.argmin(distances)
    if distances[best_match_index] < 0.5:
        return known_face_names[best_match_index]
    else:
        global stranger_counter
        stranger_name = f"Stranger {stranger_counter}"
        stranger_counter += 1
        return stranger_name


def log_attendance(name):
    try:
        with open('attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            current_time = datetime.now().strftime("%H:%M:%S")
            writer.writerow([name, "present", current_time])
            print(f"Logged attendance for {name} at {current_time}")
    except Exception as e:
        print(f"Error logging attendance: {e}")


stranger_counter = 1
logged_names = set()


vid = cv2.VideoCapture(0)

while True:
    success, frame = vid.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    for face in faces:
        face_encoding = compute_face_encoding(rgb_frame, face)
        name = find_match(face_encoding)


        if name not in logged_names:
            log_attendance(name)
            logged_names.add(name)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
        cv2.putText(frame, name, (face.left() + 6, face.bottom() - 6), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
