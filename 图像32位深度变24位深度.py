import cv2
import os

src_path = r'C:/Users/Administrator/Desktop/1'
dst_path = r'C:/Users/Administrator/Desktop/2'
images = [item for item in os.listdir(src_path)]
for item in images:
    try:
        bgra_image = cv2.imread(os.path.join(src_path,item))
        bgr_image = cv2.cvtColor(bgra_image,cv2.COLOR_BGRA2BGR)
        img_name = os.path.join(dst_path,item)
        cv2.imwrite(img_name,bgr_image)
    except Exception as ex:
        print(item)
        continue
