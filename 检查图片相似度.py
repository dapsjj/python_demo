import numpy as np
import os
import cv2
from skimage.measure import compare_ssim

img_path = r'C:/Users/Administrator/Desktop/what/'
repeatList = []
img_files = os.listdir(img_path)
img_files=[img_path + i for i in img_files]
notNeedList= []
for thisImage in img_files:
    notNeedList.append(thisImage)
    try:
        copyImg_files = img_files.copy()
        copyImg_files.remove(thisImage)
        # img = cv2.imread(thisImage)
        img = cv2.imdecode(np.fromfile(thisImage, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (600, 400), cv2.INTER_AREA)
        for item in copyImg_files:
            if item in notNeedList:
                continue
            # print(item)
            # img1 = cv2.imread(item)
            img1 = cv2.imdecode(np.fromfile(item, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            img1 = cv2.resize(img1, (600, 400), cv2.INTER_AREA)
            ssim = compare_ssim(img, img1, multichannel=True)
            # print(thisImage,item,ssim)
            if ssim > 0.9:
                repeatList.append([item,thisImage])
                info=item+ '与' + thisImage + '可能重复'
                # repeatList.append(info+'\n')
                print(info)
    except:
        continue
a = set(tuple(sorted(l)) for l in repeatList)
b = [list(i) for i in a]
b.sort()
result=[]
for row in b:
    message = row[0] + '与' + row[1] + '可能重复'
    result.append(message+'\n')

with open(r'D:/eee.txt', 'w', newline='', encoding='utf-8') as f:
    f.writelines(result)

