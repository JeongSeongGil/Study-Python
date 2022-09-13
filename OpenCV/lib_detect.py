from util.CommUtils import *


def preprocessing():
    image = cv2.imread("image/image2.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (700, 700))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    return image, gray


face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

image, gray = preprocessing()  # 전처리

if image is None: raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

if faces.any():

    x, y, w, h = faces[0]

    face_image = image[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    if len(eyes) == 2:
        face_center = (int(x + w // 2), int(y + h // 2))

        eye_centers = [[int(x + ex + ew // 2), int(y + ey + eh // 2)] for ex, ey, ew, eh in eyes]

        correction_image, correction_center = do_correction_image(image, face_center, eye_centers)

        rois = doDetectObject(faces[0], face_center)

        cv2.rectangle(correction_image, rois[0], (255, 0, 255), 2)  # 윗머리 영역
        cv2.rectangle(correction_image, rois[1], (255, 0, 255), 2)  # 귀밑머리 영역

        cv2.rectangle(correction_image, rois[2], (255, 0, 0), 2)

        cv2.circle(correction_image, tuple(correction_center[0]), 5, (0, 255, 0), 2)  # 보정 눈 좌표
        cv2.circle(correction_image, tuple(correction_center[1]), 5, (0, 255, 0), 2)  # 보정 눈 좌표

        cv2.circle(correction_image, face_center, 3, (0, 0, 255), 2)  # 얼굴 중심 좌표

        cv2.imshow("MyFace_Edit", correction_image)

    else:
        print("눈 미검출")

else:
    print("얼굴 미검출")

cv2.waitKey(0)
