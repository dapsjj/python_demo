import xml.etree.ElementTree as ET
import os

anno_path = r'C:\Users\aaa\Desktop\test'

xml_list = os.listdir(anno_path)
xml_list = [ i for i in xml_list if i.endswith('xml')]
for axml in xml_list:
    path_xml = os.path.join(anno_path, axml)
    tree = ET.parse(path_xml)
    root = tree.getroot()
    for child in root.findall('object'):
        xmin = int(child[1][0].text)
        ymin = int(child[1][1].text)
        xmax = int(child[1][2].text)
        ymax = int(child[1][3].text)
        min_area = (xmax-xmin)*(ymax-ymin)
        if min_area < 250:
            root.remove(child)
    tree.write(os.path.join(anno_path, axml))
