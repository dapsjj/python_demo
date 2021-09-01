import cv2

src_image = r'D:/20210901120800.jpg'
img_rgb = cv2.imread(src_image)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_blur = cv2.GaussianBlur(img_gray, ksize=(11, 11), sigmaX=0, sigmaY=0)
img_edge = cv2.divide(img_gray, img_blur, scale=255)
dest_image = r"D:/result.jpg"
cv2.imwrite(dest_image, img_edge)
