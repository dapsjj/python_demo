import csv
import os
import cv2
from skimage.measure import compare_ssim

img_path = r'C:/Users/Administrator/Desktop/cardboard/'
repeatList = []
img_files = os.listdir(img_path)
img_files=[img_path + i for i in img_files]
for thisImage in img_files:
    copyImg_files = img_files.copy()
    copyImg_files.remove(thisImage)
    img = cv2.imread(thisImage)
    img = cv2.resize(img, (600, 400), cv2.INTER_AREA)
    for item in copyImg_files:
        img1 = cv2.imread(item)
        img1 = cv2.resize(img1, (600, 400), cv2.INTER_AREA)
        ssim = compare_ssim(img, img1, multichannel=True)
        # print(thisImage,item,ssim)
        if ssim > 0.9:
            repeatList.append(item)
            info=item+ '与' + thisImage + '可能重复'
            print(info)
print(repeatList)
with open(r'D:/bbb.txt', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(repeatList)
