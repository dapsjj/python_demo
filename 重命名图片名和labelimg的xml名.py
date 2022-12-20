import os
import xml.etree.ElementTree as ET


image_and_xml_path = r'C:/Users/dapsj/Desktop/test'
image_list = os.listdir(image_and_xml_path)
image_list = [i for i in image_list if not i.endswith('.xml')]
i = 1
pre = '20221220_' + 'water_'
for name in image_list:
    old_image_name = os.path.join(image_and_xml_path,name)
    new_image_name_tmp = pre + str(i)+'.jpg'
    new_image_name = os.path.join(image_and_xml_path,new_image_name_tmp) #图片重命名
    os.rename(old_image_name,new_image_name)
    old_xml_name = os.path.join(image_and_xml_path,name.split('\\')[-1].split('.')[0]+'.xml')
    tree = ET.parse(old_xml_name)
    root = tree.getroot()
    filename = root[1].text
    path = root[2].text
    root[1].text = new_image_name_tmp
    root[2].text = new_image_name_tmp
    tree.write(old_xml_name, encoding='utf-8', xml_declaration=False)
    new_xml_name_tmp = pre + str(i)+'.xml'
    new_xml_name = os.path.join(image_and_xml_path,new_xml_name_tmp) #xml重命名
    os.rename(old_xml_name, new_xml_name)
    i+=1

