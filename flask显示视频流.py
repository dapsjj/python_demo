
from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if success:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(img, size=640)
            results.display(render=True)#图片加框
            im_rgb = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB)  # Because of OpenCV reading images as BGR
            # df = results.pandas().xyxy[0] #坐标和类别信息  xmin    ymin    xmax   ymax  confidence  class    name
            ret, buffer = cv2.imencode('.jpg', im_rgb)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5s.pt')  # local model
    app.run(host='127.0.0.1', port=8081)



'''
import torch
from PIL import Image
import cv2
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/yolov5m.pt')  # local model

# Images
img2 = cv2.imread('data/images/bus.jpg')[:, :, ::-1]  # OpenCV image (BGR to RGB)

# Inference
results = model(img2, size=640)  # includes NMS
results.display(render=True)

im_rgb = cv2.cvtColor(results.imgs[0], cv2.COLOR_BGR2RGB) # Because of OpenCV reading images as BGR
cv2.imshow('Detection Screen', im_rgb)
cv2.waitKey(0)

df = results.pandas().xyxy[0]
for index, row in df.iterrows():
    print(int(row['xmin']), row['name'])
'''
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
