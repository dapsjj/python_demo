import os
import glob
import csv
import xml.etree.ElementTree as ET

os.chdir(r'C:/Users/Administrator/Desktop/xml转txt')
path = r'C:/Users/Administrator/Desktop/xml转txt'

#image_path x_min,y_min,x_max,y_max,class_id  x_min,y_min,x_max,y_max,class_id ……

#txt格式:E:/VOC2007/JPEGImages/000005.jpg 263,211,324,339,car 165,264,253,372,truck 241,194,295,299,person

#我这里的class_id是英语，先生成这样的英语类别也没关系，生成txt后再打开txt全局替换car成0，truck成1，person成2，这里的0,1,2是根据voc.names中的内容来sheHi的
#比如voc.names内容如下：
#car
#truck
#person
#……



def xml_to_txt(path):
    txt_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        everyrow_xml_list = []
        tree = ET.parse(xml_file)
        root = tree.getroot()
        everyrow_xml_list.append(path + '/' + root.find('filename').text)
        # everyrow_xml_list.append(int(root.find('size')[0].text))#图片宽
        # everyrow_xml_list.append(int(root.find('size')[1].text))#图片高
        for member in root.findall('object'):
            value = str(int(member[4][0].text))+','+str(int(member[4][1].text))+','+str(int(member[4][2].text))+','+str(int(member[4][3].text))+','+member[0].text
            everyrow_xml_list.append(value)
        txt_list.append(everyrow_xml_list)#image_path x_min,y_min,x_max,y_max,class_id  x_min,y_min,x_max,y_max,class_id ……
    return txt_list



def main():
    image_path = path
    xml2txt_list = xml_to_txt(image_path)
    with open(r'D:/aaa.txt', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f,delimiter=' ')
        # writer.writerows(title)
        writer.writerows(xml2txt_list)
    print('Successfully converted xml to txt.')

if __name__ == '__main__':
    main()
