from util.CommUtils import *
import numpy as np
import cv2


def preprocessing():
    image = cv2.imread("image/image2.jpg", cv2.IMREAD_COLOR)

    if image is None: return None, None

    image = cv2.resize(image, (700, 700))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray - cv2.equalizeHist(gray)

    return image, gray


face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("data/haarcascade_eye_tree_eyeglasses.xml")

image, gray = preprocessing()

if image is None: raise Exception("영상 파일 읽기 에러")

faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

if faces.any():
    x, y, w, h = faces[0]

    face_image = image[y: y + h, x: x + w]
    eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

    if len(eyes) == 2:
        face_center = (int(x + w // 2), int(y + h // 2),)
        eye_centers = [[int(x + ex + ew // 2), int(y + ey + eh // 2)] for ex, ey, ew, eh in eyes]

        correction_image, correction_center = do_correction_image(image, face_center, eye_centers)

        rois = doDetectObject(faces[0], face_center)
        base_mask = np.full(correction_image.shape[:2], 255, np.uint8)
        face_mask = draw_ellipse(base_mask, rois[3], 0, -1)
        lip_mask = draw_ellipse(np.copy(base_mask), rois[2], 255)

        masks = [face_mask, face_mask, lip_mask, ~lip_mask]
        masks = [mask[y:y + h, x: x + w] for mask, (x, y, w, h) in zip(masks, rois)]

        for i, mask in enumerate(masks):
            cv2.imshow('mask' + str(i), mask)

            subs = [correction_image[y: y + h, x: x + w] for x, y, w, h in rois]

        for i, sub in enumerate(subs):
            cv2.imshow('sub' + str(i), sub)

    else:
        print("cannot find eyes")

else:
    print("cannot find face")

cv2.waitKey(0)
