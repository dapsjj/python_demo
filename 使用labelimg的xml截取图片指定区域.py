import cv2
import os
import xml.etree.ElementTree as ET
imagePath = r'C:\Users\Administrator\Desktop\train' #此路径是含有原始大图片和xml的路径
savePath = r'C:\Users\Administrator\Desktop\regionSaveTrain' #要保存截取后图片的路径

def saveRegion(name):
    xml_file = os.path.join(imagePath,name+'.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    count=1
    for member in root.findall('object'):
        xmin = int(member[4][0].text)-100 if int(member[4][0].text)-100>0 else 1
        ymin = int(member[4][1].text)-100 if int(member[4][1].text)-100>0 else 1
        xmax = int(root.find('size')[0].text) if int(member[4][2].text)+100>int(root.find('size')[0].text) else int(member[4][2].text)+100
        ymax = int(root.find('size')[1].text) if int(member[4][3].text)+100>int(root.find('size')[1].text) else int(member[4][3].text)+100
        # int(root.find('size')[0].text)#图片宽
        # int(root.find('size')[1].text)#图片高
        img = cv2.imread(os.path.join(imagePath,name+'.jpg'))  # imread后面括号中的为图片的具体地址
        img_2 = img[ymin:ymax, xmin:xmax]
        cv2.imwrite(os.path.join(savePath,name+'_' + str(count) + '.jpg'), img_2)
        count+=1

fileList = os.listdir(imagePath)
fileList = [item.split('.')[0] for item in fileList if item.endswith('xml')]
for item in fileList:
    saveRegion(item)
