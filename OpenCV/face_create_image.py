import cv2

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

vcp = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0



while True:
    ret, my_image = vcp.read()

    if ret is True:

        gray = cv2.cvtColor(my_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, 0, (20, 20))

        face_cnt = len(faces)

        if face_cnt == 1:
            count += 1
            cv2.imwrite("my_face/" + str(count) + ".jpg", my_image)

    if cv2.waitKey(1) == 13 or count == 100:
        break

vcp.release()
cv2.destroyAllWindows()
