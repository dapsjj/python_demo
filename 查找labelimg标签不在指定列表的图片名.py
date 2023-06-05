import os
import xml.etree.ElementTree as ET

label_list = ['algae', 'clean', 'color', 'floating', 'foam', 'rainbow']#标签列表
dir_a = r"D:/image"  # xml路径

# 遍历目录下的所有xml文件
for xml_file in os.listdir(dir_a):
    if xml_file.endswith(".xml"):
        xml_path = os.path.join(dir_a, xml_file)

        # 解析xml文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 查找标签名
        label_names = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            label_names.append(label)

        # 判断标签名是否在指定列表中
        if not set(label_names).issubset(set(label_list)):
            # 获取对应的图片名
            image_name = xml_file.replace(".xml", ".jpg")
            print(f"图片名: {image_name}，标签名: {label_names}")
