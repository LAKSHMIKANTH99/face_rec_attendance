import cv2
import face_recognition as fr
import numpy as np

def resize(img, size):
    width = int(img.shape[1] * size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation=cv2.INTER_AREA)

reddy1 = fr.load_image_file('pics/REDDY.jpg')
reddy_test1 = fr.load_image_file('pics/')

reddy1 = cv2.cvtColor(reddy1, cv2.COLOR_BGR2RGB)
reddy_test1 = cv2.cvtColor(reddy_test1, cv2.COLOR_BGR2RGB)

reddy1 = resize(reddy1, 0.5)
reddy_test1 = resize(reddy_test1, 0.5)



face_locations_1 = fr.face_locations(reddy1, number_of_times_to_upsample=1, model="cnn")
if face_locations_1:
    encode_reddy1 = fr.face_encodings(reddy1, known_face_locations=face_locations_1)
    if encode_reddy1:
        loc_1 = face_locations_1[0]
        cv2.rectangle(reddy1, (loc_1[3], loc_1[0]), (loc_1[1], loc_1[2]), (255, 0, 255), 3)


face_locations_2 = fr.face_locations(reddy_test1, number_of_times_to_upsample=1, model="cnn")
if face_locations_2:
    encode_reddytest1 = fr.face_encodings(reddy_test1, known_face_locations=face_locations_2)
    if encode_reddytest1:
        loc_2 = face_locations_2[0]
        cv2.rectangle(reddy_test1, (loc_2[3], loc_2[0]), (loc_2[1], loc_2[2]), (255, 0, 255), 3)


if 'encode_reddy1' in locals() and 'encode_reddytest1' in locals():
    results = fr.compare_faces([encode_reddy1[0]], encode_reddytest1[0])
    print(results)
    cv2.putText(reddy_test1, f'{results}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
else:
    print("Face not encoded in one or both images")

# Display images
cv2.imshow('main', reddy1)
cv2.imshow('test', reddy_test1)
cv2.waitKey(0)
cv2.destroyAllWindows()
