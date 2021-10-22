
import xml.etree.ElementTree as ET
import os

anno_path = r'C:/Users/eseen_new/Desktop/all' #labelimg的xml路径
xml_list = os.listdir(anno_path)
xml_list = [i for i in xml_list if i.endswith('xml')]
for axml in xml_list:
    try:
        path_xml = os.path.join(anno_path, axml)
        tree = ET.parse(path_xml)
        root = tree.getroot()
        object = root.findall('object')
        if not object:
            print(axml)
            image_name_xml = os.path.join(anno_path, axml)
            image_name_jpg = os.path.join(anno_path, axml.split('.')[0] + '.jpg')
            image_name_png = os.path.join(anno_path, axml.split('.')[0] + '.png')
            if os.path.exists(image_name_jpg):
                os.remove(image_name_jpg)
            if os.path.exists(image_name_png):
                os.remove(image_name_png)
            if os.path.exists(image_name_xml):
                os.remove(image_name_xml)
            continue
    except Exception as ex:
        print(axml)
