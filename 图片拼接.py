import os
import cv2

#图像拼接
left_dir = r'C:/Users/XXX/Desktop/left' #路径不能含有中文
right_dir = r'C:/Users/XXX/Desktop/algae/right' #路径不能含有中文
lake_river_algae = r'C:/Users/XXX/Desktop/aaa'


left_images = [os.path.join(left_dir,i) for i in os.listdir(left_dir)]
right_images = [os.path.join(right_dir,i) for i in os.listdir(right_dir) if not i.endswith('.xml')]
concat_images = left_images[:len(right_images)]
print(len(concat_images))
print(len(right_images))
for i in range(len(concat_images)):
    left = cv2.imread(concat_images[i])
    right = cv2.imread(right_images[i])
    height, width = right.shape[0], right.shape[1]
    left = cv2.resize(left, (width, height))
    hconcat = cv2.hconcat([left, right])
    cv2.imwrite(os.path.join(lake_river_algae,'lake_river_algae_'+str(i)+'.jpg'),hconcat)
