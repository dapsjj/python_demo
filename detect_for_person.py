"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import ctypes
import numpy as np
import statistics

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync


@torch.no_grad()
def run(weights='yolov5s.pt',  # model.pt path(s)
        source='data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    #初始化海康威视摄像头
    mydll = ctypes.cdll.LoadLibrary(r'./VR_Control_DLL.dll')
    result1 = mydll.VR_Init() #成功返回 0
    ip = '192.168.1.199'
    user = 'admin'
    password = 'eseen2015'
    lUserID = mydll.VR_Login(ip.encode("utf-8"), user.encode("utf-8"), password.encode("utf-8")) # 成功返回 设备ID号
    count = 0
    for path, img, im0s, vid_cap in dataset:

        # 20210812,是否有人标志位，False代表没有人
        person_flag = False
        #计算画面中的所有人的位置的中心
        person_coordinate_center = [] #存放每个人中心点坐标[[x1,y1],[x2,y2],[x3,y3]]
        person_coordinate_leftTop_and_rightBottom = [] #存放每个人左上角和右下角坐标[[x1,y1,x2,y2],[x3,y3,x4,y4]]

        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # 20210812添加,如果是person(index是0)
                        if c == 0:
                            person_flag = True
                            person_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]  # 截取人位置
                            person_name = os.path.join(r'D:/person_in_image',str(count) + '.jpg')  # 图片名称
                            # cv2.imwrite(person_name, person_image)
                            person_center = [int(int(xyxy[0])+int(xyxy[2])/2),(int(xyxy[1])+int(xyxy[3])/2)]#x,y
                            #向person_coordinate_center中添加每个人的中心点坐标
                            person_coordinate_center.append(person_center)

                            person_leftTop_and_rightBottom = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]#左上角和右下角
                            # 向person_coordinate_leftTop_and_rightBottom中添加每个人的左上角和右下角坐标
                            person_coordinate_leftTop_and_rightBottom.append(person_leftTop_and_rightBottom)

                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)#画框
                            if save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            count += 1

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            sleep_time = 0.05  # 转动摄像头的动作持续多久
            distance_threshold = 200  # 人物中心点坐标与屏幕中心差的像素
            proportion_of_the_largest_rectangle = 5  # 最大的人的矩形框面积占据屏幕的几分之一

            if person_coordinate_center:#如果识别到人，则控制摄像头转动，把人放到屏幕中间
                person_coordinate_list = np.array(person_coordinate_center)
                person_x_center = statistics.mean([ x[0] for x in person_coordinate_list ])  # 行均值
                person_y_center = statistics.mean([ x[1] for x in person_coordinate_list ])  # 列均值
                height,width = im0.shape[:2]
                image_x_center = int(width/2)
                image_y_center = int(height/2)

                #取所有人的坐标框中面积最大的一个矩形框
                max_area = 0
                for item in person_coordinate_leftTop_and_rightBottom:
                    rect_width = item[2] - item[0]
                    rect_height = item[3] - item[1]
                    if rect_height * rect_width > max_area:
                        max_area = rect_height * rect_width

                if person_x_center - image_x_center >= distance_threshold:#人偏右，需要右移摄像头，让人出现在屏幕中央
                    right = mydll.VR_Control(lUserID, 1, 24, 0)
                    time.sleep(sleep_time)
                    right = mydll.VR_Control(lUserID, 1, 24, 1)

                    if max_area <= (height * width)/proportion_of_the_largest_rectangle: #如果最大的人的矩形框的面积小于等于屏幕五分之一,就缩放
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 0) #焦距变大
                        time.sleep(sleep_time)
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 1)

                elif person_x_center - image_x_center <= -distance_threshold:#人偏左，需要左移摄像头，让人出现在屏幕中央
                    left = mydll.VR_Control(lUserID, 1, 23, 0)
                    time.sleep(sleep_time)
                    left = mydll.VR_Control(lUserID, 1, 23, 1)

                    if max_area <= (height * width) / proportion_of_the_largest_rectangle:  # 如果最大的人的矩形框的面积小于等于屏幕五分之一,就缩放
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 0)  # 焦距变大
                        time.sleep(sleep_time)
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 1)

                elif person_y_center - image_y_center >= distance_threshold:  # 人靠上，需要上移摄像头，让人出现在屏幕中央
                    up = mydll.VR_Control(lUserID, 1, 21, 0)
                    time.sleep(sleep_time)
                    up = mydll.VR_Control(lUserID, 1, 21, 1)

                    if max_area <= (height * width) / proportion_of_the_largest_rectangle:  # 如果最大的人的矩形框的面积小于等于屏幕五分之一,就缩放
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 0)  # 焦距变大
                        time.sleep(sleep_time)
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 1)

                elif person_y_center - image_y_center <= -distance_threshold:  # 人靠下，需要下移摄像头，让人出现在屏幕中央
                    down = mydll.VR_Control(lUserID, 1, 22, 0)
                    time.sleep(sleep_time)
                    down = mydll.VR_Control(lUserID, 1, 22, 1)

                    if max_area <= (height * width) / proportion_of_the_largest_rectangle:  # 如果最大的人的矩形框的面积小于等于屏幕五分之一,就缩放
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 0)  # 焦距变大
                        time.sleep(sleep_time)
                        ZOOM_IN = mydll.VR_Control(lUserID, 1, 11, 1)

            else:#没人把焦点还原回去5秒
                ZOOM_OUT = mydll.VR_Control(lUserID, 1, 12, 0) #焦距变小
                time.sleep(sleep_time)#焦距变小5秒
                ZOOM_OUT = mydll.VR_Control(lUserID, 1, 12, 1)


            if person_flag == True:
                # 20210812保存person图片
                image_name = os.path.join(r'D:/person_in_image',str(count) + '.jpg')
                cv2.imwrite(image_name, im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 10, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            break

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5m.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default=r'D:/pythonProjects/yolov5-master/dataset/images/train_yellow',help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default=r'rtsp://admin:eseen2015@192.168.1.199:554/h264/ch1/main/av_stream', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
