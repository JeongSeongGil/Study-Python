import cv2
import numpy as np
import pathlib

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
model = cv2.face.LBPHFaceRecognizer_create()

training_data, labels = [], []

data_dir = pathlib.Path("my_face")
print("data_dir", data_dir)

image_count = len(list(data_dir.glob("*.jpg")))
print("image_count : " + str(image_count))

for image_path in data_dir.glob("*.jpg"):
    my_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (20, 20))
    face_cnt = len(faces)

    if face_cnt == 1:
        x, y, w, h = faces[0]
        face_image = gray[y:y + h, x: x + w]

        training_data.append(face_image)
        labels.append(0)

        print(training_data)
        print(labels)

        model.train(training_data, np.array(labels))
        model.save("model/face-trainner2.yml")

        cv2.rectangle(my_image, faces[0], (255, 0, 0), 4)

cv2.destroyAllWindows()
