import numpy as np
import dlib
import cv2


video_capture = cv2.VideoCapture("huge.mp4")
face_cascade = cv2.CascadeClassifier(r'E:/OpenCV2.4.9/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# image_file ='images/face.jpg'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    ret, frame = video_capture.read()
    ret2, frame2 = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(50, 50)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        for (x, y) in shape:
                cv2.circle(frame2, (x, y), 2, (0, 0, 255), 1)

    cv2.imshow("Output", frame2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
