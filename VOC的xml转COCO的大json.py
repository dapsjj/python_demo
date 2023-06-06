import os
import xml.etree.ElementTree as ET
import json

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_path = root.find('path').text

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    annotations = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        label = obj.find('name').text
        annotations.append({
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'label': label
        })

    return image_path, image_width, image_height, annotations

def build_coco_structure(xml_files, labels):
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    image_id = 1
    annotation_id = 1
    category_id = 1

    category_id_map = {}
    for label in labels:
        coco_data['categories'].append({
            'id': category_id,
            'name': label,
            'supercategory': 'none'
        })
        category_id_map[label] = category_id
        category_id += 1

    for xml_file in xml_files:
        image_path, image_width, image_height, annotations = parse_xml(xml_file)

        image_name = os.path.basename(image_path)
        coco_data['images'].append({
            'id': image_id,
            'file_name': image_name,
            'width': image_width,
            'height': image_height
        })

        for annotation in annotations:
            bbox = annotation['bbox']
            label = annotation['label']
            coco_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': category_id_map[label],
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0
            })
            annotation_id += 1

        image_id += 1

    return coco_data

def save_coco_json(coco_data, json_file):
    with open(json_file, 'w') as f:
        json.dump(coco_data, f)

def convert_xml_to_coco(xml_folder, labels, json_file):
    xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]
    coco_data = build_coco_structure(xml_files, labels)
    save_coco_json(coco_data, json_file)

# 使用示例
xml_folder = r'C:/Users/aa/Desktop/新建文件夹'  # 替换为实际的XML文件夹路径
labels = ['algae', 'clean', 'color', 'floating', 'foam', 'rainbow']  # 标签列表
json_file = 'C:/Users/aa/Desktop/instances_train2017.json'  # 替换为输出的JSON文件路径
convert_xml_to_coco(xml_folder, labels, json_file)
