# 此程序用于修改图形文件的文件名为以数字来命名
# encoding:utf-8
import os
import xml.etree.ElementTree as ET
from natsort import natsorted


currentPath= r'C:/Users/123/Desktop/新建文件夹' #路径名

print('currentPath:'+currentPath)
pre = 'water_20230425_'
fileno = 200
listfile=os.listdir(currentPath)    # 得到进程当前工作目录中的所有文件名称列表
listfile = natsorted(listfile)   # 排序按 1.jpg,2.jpg,3.jpg顺序排，不是1.jpg.10.jpg,11.jpg顺序排

for fileName in listfile:  # 获取文件列表中的文件
    thisPath = currentPath + '\\' + fileName  # 当前文件的路径
    if os.path.isfile(thisPath):
        if fileName.endswith("jpg"):
            jpgFileName = fileName.split('.')[0]
        elif  fileName.endswith("xml"):
            xmlFileName = fileName.split('.')[0] # 取当前文件名,去掉扩展名

            try:
                if xmlFileName == jpgFileName:
                    fileno += 1
                    # 修改图形文件名
                    os.rename(os.path.join(currentPath, jpgFileName + ".jpg"),
                              os.path.join(currentPath, pre+str(fileno) + ".jpg"))

                    # 处理描述文件内容
                    tree = ET.parse(thisPath)
                    root = tree.getroot()
                    for child in root:
                        if child.tag == "filename":
                            child.text = pre+str(fileno) + ".jpg"
                        if child.tag == "path":
                            child.text =  pre+str(fileno) + ".jpg"
                    tree.write(thisPath)  # 保存修改后的XML文件
                    # 修改xml文件名
                    os.rename(os.path.join(currentPath, fileName), os.path.join(currentPath,pre+ str(fileno) + ".xml"))
                else:
                    print('图形文件 ' + jpgFileName+'.jpg 不存在对应的 xml 描述文件,请检查！')
                    continue
            except:
                print(str(fileno) + '.jpg 已存在名字, ' + fileName + '未改名！')
                continue

    elif os.path.isdir(thisPath):
        print("'" + thisPath + "' is a directory,can not rename it!")
    else:
        print("'" + thisPath + "' is a special file(socket,FIFO,device file)")
