import cv2
import os
import xml.etree.ElementTree as ET
imagePath = r'C:\Users\Administrator\Desktop\test' #此路径是含有原始大图片和xml的路径
savePath = r'C:\Users\Administrator\Desktop\auto_labelimg' #要保存截取后图片和xml的路径
xml_folder = r'zhinengwanglianbu_20200727'

objstr = """\
<annotation>
    <folder>%s</folder>
    <filename>%s</filename>
    <path>%s</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
</annotation>
"""

def saveRegionAndXml(name):
    xml_file = os.path.join(imagePath,name+'.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    count=1
    for member in root.findall('object'):
        xmin_100 = int(member[4][0].text)-100 if int(member[4][0].text)-100>0 else 1
        ymin_100 = int(member[4][1].text)-100 if int(member[4][1].text)-100>0 else 1
        xmax_100 = int(root.find('size')[0].text) if int(member[4][2].text)+100>int(root.find('size')[0].text) else int(member[4][2].text)+100
        ymax_100 = int(root.find('size')[1].text) if int(member[4][3].text)+100>int(root.find('size')[1].text) else int(member[4][3].text)+100
        class_name = member[0].text

        # int(root.find('size')[0].text)#图片宽
        # int(root.find('size')[1].text)#图片高
        img_org = cv2.imread(os.path.join(imagePath,name+'.jpg'))  # imread后面括号中的为图片的具体地址
        img_2 = img_org[ymin_100:ymax_100, xmin_100:xmax_100]
        small_image_path = os.path.join(savePath,name+'_' + str(count) + '.jpg')
        small_image_name = name+'_' + str(count) + '.jpg'
        cv2.imwrite(small_image_path, img_2)
        small_xml_name = os.path.join(savePath,name+'_' + str(count) + '.xml')
        small_image_width = xmax_100 - xmin_100 #截取出来的小图的宽度
        small_image_height = ymax_100 - ymin_100 #截取出来的小图的高度

        # 计算labelimg与边框的实际距离
        xmin_distance = int(member[4][0].text) - xmin_100
        ymin_distance = int(member[4][1].text) - ymin_100
        xmax_distance = xmax_100 - int(member[4][2].text)
        ymax_distance = ymax_100 - int(member[4][3].text)

        # 计算小图坐标
        small_xmin = xmin_distance
        small_ymin = ymin_distance
        small_xmax = small_image_width - xmax_distance
        small_ymax = small_image_height - ymax_distance


        with open(small_xml_name, 'w', encoding='utf-8') as f:
            f.write(objstr %(xml_folder,small_image_name,small_image_name,small_image_width,small_image_height,class_name,small_xmin,small_ymin,small_xmax,small_ymax))
        count+=1

fileList = os.listdir(imagePath)
fileList = [item.split('.')[0] for item in fileList if item.endswith('xml')]
for item in fileList:
    saveRegionAndXml(item)
