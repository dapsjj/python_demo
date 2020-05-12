import os
import cv2
from xml.dom import minidom

testImagesPath = r'E:\test_opencv\tensorflow-yolov3-master_for_12_garbage\data\dataset\train'
testList=os.listdir(testImagesPath)
for test in testList:
    if test.endswith('.jpg'):
        img=cv2.imread(testImagesPath+'/'+test)
        image_height, image_weight = img.shape[0:2]
        xmlName=test.split('.')[0]+'.xml'
        doc = minidom.parse(testImagesPath+'/'+xmlName)
        xml_width_node = doc.getElementsByTagName('width')
        xml_height_node = doc.getElementsByTagName('height')
        xml_width=int(doc.getElementsByTagName('width')[0].firstChild.nodeValue)
        xml_height=int(doc.getElementsByTagName('height')[0].firstChild.nodeValue)
        if image_height!=xml_height or image_weight!=xml_width:
            print(testImagesPath+'\\'+test,xml_width,xml_height)
