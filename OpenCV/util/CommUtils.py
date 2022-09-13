import numpy as np, cv2


# 이미지 기울기 보정을 위한 함수
# 인자값1 : 원본이미지 / 인자값2 : 얼굴 중심 좌표 / 인자값3 : 양쪽 눈 중심좌표
# 결과값 : 보정된 이미지, 보정된 얼굴 중심 좌표
def do_correction_image(image, face_center, eye_centers):
    # 양쪽 눈 좌표
    pt0, pt1 = eye_centers

    if pt0[0] > pt1[0]: pt0, pt1 = pt1, pt0

    # 두 좌표간 차분 계산
    dx, dy = np.subtract(pt1, pt0).astype(float)

    # 역탄젠트로 기울기 계산
    angle = cv2.fastAtan2(dy, dx)

    # 계산된 기울기만큼 이미지 회전하기
    rot = cv2.getRotationMatrix2D(face_center, angle, 1)

    # 회전된 이미지를 원래 이미지 크기로 자르기
    size = image.shape[1::-1]

    # 보정된 이미지 생성
    correction_image = cv2.warpAffine(image, rot, size, cv2.INTER_CUBIC)

    # 눈 위치 보정
    eye_centers = np.expand_dims(eye_centers, axis=0)
    correction_centers = cv2.transform(eye_centers, rot)
    correction_centers = np.squeeze(correction_centers, axis=0)

    return correction_image, correction_centers