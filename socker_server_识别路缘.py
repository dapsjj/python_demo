#! /usr/bin/env python
# coding=utf-8

import cv2
import time
import numpy as np
import core.utils as utils
import tensorflow as tf
from PIL import Image
import socket
import pickle
import struct

HOST = '127.0.0.1'
PORT = 8485

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
# pb_file         = "./yolov3_coco.pb"
pb_file         = "./yolov3_curb_1218pic_17epoch_test_loss=3.8328_train_loss=2.6367.pb"
# video_path      = 0
video_path      = "./docs/images/road.mp4"
# num_classes     = 80
num_classes     = 1
input_size      = 416
graph           = tf.Graph()
return_tensors  = utils.read_pb_return_tensors(graph, pb_file, return_elements)

with tf.Session(graph=graph) as sess:

    socketserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print('Socket created')
    socketserver.bind((HOST, PORT))
    print('Socket bind complete')
    socketserver.listen(10)
    print('Socket now listening')
    clientsocket, addr = socketserver.accept()
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += clientsocket.recv(4096)
        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += clientsocket.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        frame_size = frame.shape[:2]
        image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...]
        prev_time = time.time()

        pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
            [return_tensors[1], return_tensors[2], return_tensors[3]],
                    feed_dict={ return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        # bboxes :xmin, ymin, xmax, ymax, score, class
        strPositio = ''
        if bboxes:
            for row in bboxes:
                xmin, ymin, xmax, ymax = row[:4]
                strPositio += str(int(xmin)) + ',' + str(int(ymin)) + ',' + str(int(xmax)) + ',' + str(int(ymax)) + ';'
            print(strPositio[:-1])  # 打印路缘坐标，为socket通信用

        image = utils.draw_bbox(frame, bboxes)
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        msg = strPositio
        if not msg:
            msg='None'
        # 对要发送的数据进行编码
        print('消息是：'+msg)
        clientsocket.send(msg.encode("utf-8"))




