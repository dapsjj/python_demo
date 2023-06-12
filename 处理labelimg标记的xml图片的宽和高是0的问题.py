import os
import cv2
import xml.etree.ElementTree as ET

# 输入路径
xml_folder = '/a'  # XML文件夹路径
image_folder = '/a'  # 图片文件夹路径

# 遍历XML文件夹中的所有文件
for xml_file in os.listdir(xml_folder):
    if xml_file.endswith('.xml'):
        # 获取XML文件的路径和文件名（不带扩展名）
        xml_path = os.path.join(xml_folder, xml_file)
        file_name = os.path.splitext(xml_file)[0]
        
        # 读取对应的图片文件
        image_path = os.path.join(image_folder, file_name + '.jpg')
        img = cv2.imread(image_path)
        
        # 获取图片的宽度和高度
        height, width, _ = img.shape
        
        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 更新XML中的宽度和高度信息
        size_elem = root.find('size')
        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')
        width_elem.text = str(width)
        height_elem.text = str(height)
        
        # 保存更新后的XML文件
        tree.write(xml_path)

print("宽度和高度已更新！")
