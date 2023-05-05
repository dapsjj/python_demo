import os
import xml.etree.ElementTree as ET

folder_path = r"C:/Users/user1/Desktop/aa"
label_name= 'water'

# 遍历文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".xml"):
            # 打开XML文件并解析
            xml_path = os.path.join(root, file)
            tree = ET.parse(xml_path)
            xml_root = tree.getroot()

            # 找到并删除标签名称为label_name的节点
            for obj in xml_root.findall('object'):
                name = obj.find('name').text
                if name == label_name:
                    xml_root.remove(obj)

            # 将更新后的XML文件保存
            tree.write(xml_path)
