
import os

filepath = r'E:\test_opencv\models-master\research\object_detection\images\train'
for i in os.listdir(filepath):
    xml=i.split('.')[0]
    file=filepath+'/'+i
    if file.endswith('.xml'):
        if not os.path.exists(filepath+'/'+xml+'.jpg'):
            print(i)
