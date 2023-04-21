import glob
import numpy as np
import json

labelList = ['Algae', 'Clean', 'Color', 'Rainbow']

def convert_json_label_to_yolov_seg_label():
    json_path = r"C:/Users/Desktop/haha" #labelme的json的路径
    json_files = glob.glob(json_path + "/*.json")
    for json_file in json_files:
        print(json_file)
        f = open(json_file)
        json_info = json.load(f)
        height = json_info['imageHeight']
        width = json_info['imageWidth']
        np_w_h = np.array([[width, height]], np.int32)
        txt_file = json_file.replace(".json", ".txt")
        f = open(txt_file, "a")
        for point_json in json_info["shapes"]:
            txt_content = ""
            np_points = np.array(point_json["points"], np.int32)
            norm_points = np_points / np_w_h
            norm_points_list = norm_points.tolist()
            label = labelList.index(point_json["label"])
            txt_content += str(label)+" " + " ".join([" ".join([str(cell[0]), str(cell[1])]) for cell in norm_points_list]) + "\n"
            f.write(txt_content)

convert_json_label_to_yolov_seg_label()
