import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# 성별예측을 위한 학습된 모델데이터와 모델데이터 구조 설명데이터
# deploy_gender.prototxt : 모델구조 설명 데이터
# gender_net.caffemodel : 학습된 모델데이터
gender_net = cv2.dnn.readNetFromCaffe(
    "model/deploy_gender.prototxt", "model/gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

gender_list = ["Male", "Female"]

image = cv2.imread("image/image1.png", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.equalizeHist(gray)

faces = face_cascade.detectMultiScale(gray, 1.5, 5, 0, (100, 100))

for face in faces:
    x, y, w, h = face

    face_image = image[y:y + h, x:x + w]

    blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)

    gender_preds = gender_net.forward()

    gender = gender_preds.argmax()

    cv2.rectangle(image, face, (255, 0, 0), 4)

    result = "Gender : "+ gender_list[gender]

    print(result)

    cv2.putText(image, result, (x, y - 15), 0, 1, (255, 0, 0), 2)

# 이미지 출력
cv2.imshow("myFace", image)

cv2.waitKey(0)
