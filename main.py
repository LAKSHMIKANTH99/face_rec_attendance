import cv2
import face_recognition as fr
import numpy as np

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

variable1 = fr.load_image_file('path to your image file')
variable2 = fr.load_image_file('path to your image file')

variable1 = cv2.cvtColor(variable1, cv2.COLOR_BGR2RGB)
variable2 = cv2.cvtColor(variable2, cv2.COLOR_BGR2RGB)

variable1 = resize(variable1, 0.5)
variable2 = resize(variable2, 0.5)



face_locations_1 = fr.face_locations(variable1, number_of_times_to_upsample=1, model="cnn")
if face_locations_1:
    encode_variable1 = fr.face_encodings(variable1, known_face_locations=face_locations_1)
    if encode_variable1:
        loc_1 = face_locations_1[0]
        cv2.rectangle(variable1, (loc_1[3], loc_1[0]), (loc_1[1], loc_1[2]), (255, 0, 255), 3)


face_locations_2 = fr.face_locations(variable2, number_of_times_to_upsample=1, model="cnn")
if face_locations_2:
    encode_variable2 = fr.face_encodings(variable2, known_face_locations=face_locations_2)
    if encode_variable2:
        loc_2 = face_locations_2[0]
        cv2.rectangle(variable2, (loc_2[3], loc_2[0]), (loc_2[1], loc_2[2]), (255, 0, 255), 3)


if 'variable1' in locals() and 'variable2' in locals():
    results = fr.compare_faces([encode_variable1[0]], encode_variable2[0])
    print(results)
    cv2.putText(variable2, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
else:
    print("Face not encoded in one or both images")

# Display images
cv2.imshow('main', variable1)
cv2.imshow('test', variable2)
cv2.waitKey(0)
cv2.destroyAllWindows()
