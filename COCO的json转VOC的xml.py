# -*- coding: UTF-8 -*-
# import cv2

#识别图片中的空货架

# img = cv2.imread('images/short_supply.jpg')
# # imgcopy = img.copy()
# imgcopy = img
# cv2.imshow('img',img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.rectangle(imgcopy, (10, 30), (100,200), (0, 255, 0), 2)
# imgcopy[:,:,2] = 0
# cv2.imshow('img',img)
# cv2.imshow('imgcopy',imgcopy)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# img1 = cv2.imread( r'E:/test_opencv/images/n03461385_74155.JPEG' )
# operator = np.ones( ( 3, 3 ), np.uint8 )
# img_erode = cv2.erode( img1, operator, iterations=1 )
# img_dilate = cv2.dilate( img_erode, operator, iterations=3 )
# gray = cv2.cvtColor(img_dilate,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img_dilate, contours, -1, (0, 0, 255), 3)
# cv2.imshow( "img",img_dilate )
# cv2.waitKey( 0 )
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
# img1 = cv2.imread( r'E:/test_opencv/images/n03461385_74155.JPEG' )
# gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(gray,127,255,0)
# operator = np.ones( ( 3, 3 ), np.uint8 )
# img_erode = cv2.erode( thresh, operator, iterations=10 )
# img_dilate = cv2.dilate( img_erode, operator, iterations=20 )
# image, contours, hierarchy = cv2.findContours(img_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# for i in range(1, len(contours)):
#     x, y, w, h = cv2.boundingRect(contours[i])
#     cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
# cv2.imshow("img1", img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>

README FIRST:
    Two windows will show up, one for input and one for output.

    At first, in input window, draw a rectangle around the object using
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on
the areas you want. Then again press 'n' for updating the output.

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

'''
import numpy as np
import cv2

img = cv2.imread(r'E:/test_opencv/images/n03461385_74155.JPEG')
src_img = img.copy()
mask = np.zeros(img.shape[:2],np.uint8)
h,w = img.shape[0:2]
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (1,1,w,h)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
# for x in range(w):
#     for y in range(h):
#         if img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 0:
#             img[y][x][0]=255
#             img[y][x][1]=255
#             img[y][x][2]=255
cv2.imshow('src_img',src_img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
import cv2
import numpy as np
img = cv2.imread(r'E:/test_opencv/images/IMG_20171212_154743.jpg')
img = cv2.resize(img,(600,400))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ada_th0 = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5 )
ada_th1 = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5 )

kernel = np.ones( (5, 5), dtype = np.int )
morph = cv2.erode( ada_th0, kernel, iterations = 1 )
morph = cv2.dilate( morph, kernel, iterations = 2 )

cv2.imshow( 'morph0', cv2.resize( morph, (360, 640) ) )

copyimg = img.copy()
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,5)
blured = cv2.blur(img, (5, 5))
cv2.imshow('th2', th2)
morph_kernel1 = np.ones((10, 10), dtype=np.int)
morph_th2 = cv2.erode(th2, morph_kernel1, iterations=10)
morph_th2 = cv2.dilate(morph_th2, morph_kernel1, iterations=12)
cv2.imshow('th',morph_th2)
im, contours2, hierarchy = cv2.findContours(morph_th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


mask = np.zeros(img.shape[:2],np.uint8)
h,w = img.shape[0:2]
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (1,1,w,h)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
for i in range(1, len(contours2)):
    x1, y1, w1, h1 = cv2.boundingRect(contours2[i])
    print(x1,y1,x1+w1,y1+h1)
    # epsilon = 0.01 * cv2.arcLength(contours2[i], True)
    # approx = cv2.approxPolyDP(contours2[i], epsilon, True)
    # cv2.drawContours(th2, [approx], -1, (0, 255, 0), 3)
    # cv2.rectangle(copyimg, (x1-20, y1-20), (x1 + w1+20, y1 + h1+20), (0, 0, 255), 2)
    cv2.grabCut(copyimg, mask, (x1-20,y1-20,x1+w1+20,y1+h1+20), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = img * mask2[:, :, np.newaxis]
cv2.imshow('frame', copyimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''



# import logging
# import logging.handlers
# import datetime
#
# now = datetime.datetime.now()
# now=now.strftime('%Y%m%d')
#
# def Err_log_Write(path,file_name,error):
#     LOG_FILE_Err = path+ now +'_Err.log'
#     handler = logging.handlers.RotatingFileHandler(LOG_FILE_Err, maxBytes = 1024*1024, backupCount = 5) # 实例化handler
#     fmt = '%(asctime)s -%(lineno)s:%(levelname)s - %(name)s - %(message)s'
#     print(fmt)
#     formatter = logging.Formatter(fmt)       # 实例化formatter
#     handler.setFormatter(formatter)          # 为handler添加formatter
#     logger = logging.getLogger(file_name)    # 获取名为tst的logger
#     logger.addHandler(handler)               # 为logger添加handler
#     logger.setLevel(logging.DEBUG)
#     logger.debug(error)
#
# def Info_log_Write(path,file_name,Info):
#     LOG_FILE_Info = path + now + '_Info.log'
#     handler = logging.handlers.RotatingFileHandler(LOG_FILE_Info, maxBytes = 1024*1024, backupCount = 5)
#     fmt = '%(asctime)s -%(lineno)s:%(levelname)s - %(name)s - %(message)s'
#
#     formatter = logging.Formatter(fmt)
#     handler.setFormatter(formatter)
#     logger = logging.getLogger(file_name)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)
#     logger.info(Info)
#
# def Warning_log_Write(path,file_name,Warning):
#     LOG_FILE_Warning = path + now + '_Err.log'
#     handler = logging.handlers.RotatingFileHandler(LOG_FILE_Warning, maxBytes=1024 * 1024, backupCount=5)
#     fmt = '%(asctime)s -%(lineno)s:%(levelname)s - %(name)s - %(message)s'
#
#     formatter = logging.Formatter(fmt)
#     handler.setFormatter(formatter)
#     logger = logging.getLogger(file_name)
#     logger.addHandler(handler)
#     logger.setLevel(logging.WARNING)
#     logger.warning(Warning)
#
# if __name__=="__main__":
#     Err_log_Write(r'E:/','test.txt','cuowu')


'''
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
# img1 = cv2.imread( r'E:/test_opencv/images/graf1.png',0 )
# img2 = cv2.imread( r'E:/test_opencv/images/graf3.png',0 )
# img2_2 = cv2.imread( r'E:/test_opencv/images/graf3.png')
img1 = cv2.imread( r'C:/Users/Shinelon/Desktop/SRC/0000.jpg',0 )
img2 = cv2.imread( r'C:/Users/Shinelon/Desktop/SRC/0899.jpg',0 )
img2_2 = cv2.imread( r'C:/Users/Shinelon/Desktop/SRC/0899.jpg')

copy2_img = img2_2.copy()
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute( img1,None )
kp2, des2 = sift.detectAndCompute( img2,None )
FLANN_INDEX_KDTREE = 0
index_params = dict( algorithm = FLANN_INDEX_KDTREE, trees = 10 )
search_params = dict( checks = 100 )
flann = cv2.FlannBasedMatcher( index_params, search_params )
matches = flann.knnMatch( des1,des2,k=2 )
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32( [ kp1[m.queryIdx].pt for m in good ] ).reshape( -1,1,2 )
    dst_pts = np.float32( [ kp2[m.trainIdx].pt for m in good ] ).reshape( -1,1,2 )
    M, mask = cv2.findHomography( src_pts, dst_pts, cv2.RANSAC,5.0 )
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32( [ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ] ).reshape( -1,1,2 )
    dst = cv2.perspectiveTransform( pts,M )
    img2 = cv2.polylines( img2,[np.int32( dst )],True,255,3, cv2.LINE_AA )
    # print(dst_pts)
    # print(dst_pts[0][0])
    # print(dst_pts[len(dst_pts)-1][0])
    for i in range(len(dst_pts)):
        copy2_img[int(dst_pts[i][0][1]),int(dst_pts[i][0][0])] = [255,255,255]
else:
    matchesMask = None
gray = cv2.cvtColor(copy2_img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur( gray, (5, 5), 0 )
ada_th0 = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 20, 5 )
operator = np.ones( ( 5, 5 ), np.uint8 )
img_dilate = cv2.dilate( gray, operator, iterations=30 )
img_erode = cv2.erode( img_dilate, operator, iterations=25 )


draw_params = dict( matchColor = ( 0,255,0 ),
                   singlePointColor = None,
                   matchesMask = matchesMask,
                   flags = 2 )
img3 = cv2.drawMatches( img1,kp1,img2,kp2,good,None,**draw_params )
plt.imshow( img3, 'gray' )
plt.show()
cv2.imwrite('E:/jieguo.jpg',img_erode)
'''



'''
from sub import *

if __name__=="__main__":
    image_names = [r'E:/test_opencv/images/tree1.jpg',
                   r'E:/test_opencv/images/tree2.jpg',
                   r'E:/test_opencv/images/tree3.jpg',
                   r'E:/test_opencv/images/tree4.jpg',
                   r'E:/test_opencv/images/tree5.jpg',
                   r'E:/test_opencv/images/tree6.jpg',
                   r'E:/test_opencv/images/tree7.jpg',
                   r'E:/test_opencv/images/tree8.jpg',
                   r'E:/test_opencv/images/tree9.jpg']
    images = []
    panorama = []
    for i in range( 0,len( image_names ) ):
        img = resize_image( cv2.imread( image_names[i],cv2.IMREAD_COLOR ) )
        images.append( Image( str( i ), img ) )
    panorama.append( Image( images[0].name, images[0].image ) )
    for i in range( 0,len( images )-1 ):
        panorama.append( Image( str( i+1 ),make_panorama( panorama[i],images[i+1] ) ) )
    cv2.imwrite( r"E:/panorama.png",panorama[-1].image )
    cv2.imshow( 'test',panorama[-1].image )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread( r'C:/Users/Shinelon/Desktop/SRC/0000.jpg' )
img2 = cv2.imread( r'C:/Users/Shinelon/Desktop/SRC/0899.jpg' )


sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute( img1, None )
kp2, des2 = sift.detectAndCompute( img2, None )

FLANN_INDEX_KDTREE = 1

index_params = dict( algorithm = FLANN_INDEX_KDTREE, tree = 5 )
search_params = dict( checks = 50 )

flann = cv2.FlannBasedMatcher( index_params, search_params )
matches = flann.knnMatch( des1, des2, k = 2 )

good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append( m )


if len( good ) > MIN_MATCH_COUNT:

    src_pts = np.int32( [kp1[m.queryIdx].pt for m in good] )
    dst_pts = np.int32( [kp2[m.trainIdx].pt for m in good] )

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    mask_1 = np.zeros( ( h1, w1 ) )
    mask_2 = np.zeros( ( h2, w2 ) )


    mask_1[src_pts[:, 1], src_pts[:, 0]] = 255
    mask_2[dst_pts[:, 1], dst_pts[:, 0]] = 255


    # cv2.imwrite( 'mask_1_feature.jpg', cv2.cvtColor( np.uint8( mask_1 ), cv2.COLOR_GRAY2BGR ) )
    # cv2.imwrite( 'mask_2_feature.jpg', cv2.cvtColor( np.uint8( mask_2 ), cv2.COLOR_GRAY2BGR ) )

    mask_1 = cv2.GaussianBlur( mask_1, ( 81, 81 ), 0 )
    mask_2 = cv2.GaussianBlur( mask_2, ( 81, 81 ), 0 )

    mask_1[mask_1 > 0] = 255
    mask_2[mask_2 > 0] = 255


    # cv2.imwrite( 'mask_1_low_filter_pass.jpg', cv2.cvtColor( np.uint8( mask_1 ), cv2.COLOR_GRAY2BGR ) )
    # cv2.imwrite( 'mask_2_low_filter_pass.jpg', cv2.cvtColor( np.uint8( mask_2 ), cv2.COLOR_GRAY2BGR ) )
    cv2.imshow('11', cv2.cvtColor( np.uint8( mask_1 ), cv2.COLOR_GRAY2BGR ) )
    cv2.imshow('22', cv2.cvtColor( np.uint8(mask_2), cv2.COLOR_GRAY2BGR))

    mask_or = cv2.bitwise_or( np.uint8( mask_1 ), np.uint8( mask_2 ) )

    # cv2.imwrite( 'mask_or.jpg', cv2.cvtColor( np.uint8( mask_or ), cv2.COLOR_GRAY2BGR ) )
    cv2.imshow('33', cv2.cvtColor(np.uint8(mask_2), cv2.COLOR_GRAY2BGR))
    mask = cv2.bitwise_not( mask_or )
    cv2.imshow('44', cv2.cvtColor(np.uint8(mask_2), cv2.COLOR_GRAY2BGR))
    # cv2.imwrite( 'mask.jpg', cv2.cvtColor( np.uint8( mask ), cv2.COLOR_GRAY2BGR ) )
    cv2.imshow('55', cv2.cvtColor( np.uint8( mask ), cv2.COLOR_GRAY2BGR ))
    frame = cv2.bitwise_and( img1, img1, mask = mask )

    # cv2.imwrite( 'frame.jpg', frame )

    cv2.waitKey(0)
'''

'''
import os
import time
import cv2
import numpy as np

def main():
    img_src_dirpath = r'C:/Users/Shinelon/Desktop/SRC' + os.sep
    dir = r'D:/deal_pics/' + time.strftime('%Y-%m-%d') + os.sep
    if not os.path.exists(dir):
        os.makedirs(dir)
    img_dst_dirpath = dir
    img0 = None
    detector = None
    kpt0, des0 = None, None
    matcher = None
    for f in os.listdir( img_src_dirpath ):
        if f.endswith( '.jpg' ):
            if detector is None:
                img0 = cv2.imread( img_src_dirpath + f )
                detector = cv2.xfeatures2d.SIFT_create()
                kpt0, des0 = detector.detectAndCompute( img0, None )
                FLANN_INDEX_KDTREE = 1
                index_params = dict( algorithm = FLANN_INDEX_KDTREE, trees = 5 )
                search_params = dict( checks = 50 )
                matcher = cv2.FlannBasedMatcher( index_params, search_params )
            else:
                img1 = cv2.imread( img_src_dirpath + f )
                kpt1, des1 = detector.detectAndCompute( img1, None )
                matches = matcher.knnMatch( des0, des1, k = 2 )
                matchesMask = [[0, 0] for i in range( len( matches ) )]
                good = []
                for i, (m, n) in enumerate( matches ):
                    if m.distance < 0.7 * n.distance:
                        matchesMask[i] = [1, 0]
                        good.append( m )
                if len( good ) > 10:
                    src_pts = np.float32( [kpt0[m.queryIdx].pt for m in good] ).reshape( -1, 1, 2 )
                    dst_pts = np.float32( [kpt1[m.trainIdx].pt for m in good] ).reshape( -1, 1, 2 )
                    M, mask = cv2.findHomography( dst_pts, src_pts, cv2.RANSAC, 5.0 )
                    h, w, _ = img0.shape
                    img3 = cv2.warpPerspective( img1, M, (w, h) )
                    cv2.imwrite( img_dst_dirpath + f, img3 )
                else:
                    print( 'Not enough good matches. length of good matches: %d' % len( good ) )


if __name__ == '__main__':
    main()
'''


'''
import os
import cv2
import numpy as np


def get_diff( img0, img1 ):
    gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
    # gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2YCrCb )[:,:,0]
    gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
    diff = cv2.absdiff( gray0, gray1 )
    _, th = cv2.threshold( diff, 70, 255, cv2.THRESH_BINARY )

    morph_kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5, 5) )
    img_bin = cv2.erode( th, morph_kernel, iterations = 2 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 25 )
    img_bin = cv2.erode( img_bin, morph_kernel, iterations = 30 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 40 )

    cv2.imshow( 'th', cv2.resize( th, (640, 360) ) )
    cv2.imshow( 'img_bin', cv2.resize( img_bin, (640, 360) ) )

    return th, img_bin

def get_diff_02( img0, img1 ):
    gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
    gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
    diff = cv2.absdiff( gray0, gray1 )
    _, th = cv2.threshold( diff, 70, 255, cv2.THRESH_BINARY )

    morph_kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5, 5) )
    img_bin = cv2.erode( th, morph_kernel, iterations = 2 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 25 )
    img_bin = cv2.erode( img_bin, morph_kernel, iterations = 30 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 40 )

    cv2.imshow( 'th', cv2.resize( th, (640, 360) ) )
    cv2.imshow( 'img_bin', cv2.resize( img_bin, (640, 360) ) )
    return th, img_bin


def main1():
    img_src_dirpath0 = r'D:/deal_pics/2018-01-08/'
    video_dst = r'E:/20180109.mp4'

    files = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files[0] )
    height0, width0, _ = img0.shape
    width1, height1 = int( width0 / 2 ), int( height0 / 2 )
    fourcc = cv2.VideoWriter_fourcc( *'DIVX' )
    fps = 30.0
    vwriter = cv2.VideoWriter( video_dst, fourcc, fps, (width0, height1) )

    img_num = len( files )
    diff_step = 100
    waitSec = 1
    for i in range( img_num ):
        f = files[i]
        img1 = cv2.imread( img_src_dirpath0 + f )

        if i >= diff_step:
            img0 = cv2.imread( img_src_dirpath0 + files[i - diff_step] )
        else:
            img0 = img1.copy()

        th, img_bin = get_diff( img0, img1 )
        img3 = cv2.bitwise_and( img1, img1, mask = img_bin )
        _, contours, hierarchy = cv2.findContours( img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        for ct in contours:
            x, y, w, h = cv2.boundingRect( ct )
            cv2.rectangle( img1, (x, y), (x + w, y + h), (0, 255, 255), 5 )
        img = np.zeros( (height1, width0, 3), dtype = np.uint8 )
        img[:height1, :width1, :] = cv2.resize( img1, (width1, height1) )
        img[:height1, width1:width0, :] = cv2.resize( img3, (width1, height1) )
        vwriter.write( img )
        cv2.putText( img3, f, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
        cv2.imshow( 'img0', cv2.resize( img0, (640, 360) ) )
        cv2.imshow( 'img1', cv2.resize( img1, (640, 360) ) )
        cv2.imshow( 'img3', cv2.resize( img3, (640, 360) ) )
        cv2.imshow( 'img', cv2.resize( img, (1280, 360) ) )

        key = cv2.waitKey( waitSec ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec = 1 - waitSec

    vwriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main1()
'''

'''
import os
import cv2
import numpy as np


def get_diff( img0, img1 ):
    gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
    # gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2YCrCb )[:,:,0]
    gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
    diff = cv2.absdiff( gray0, gray1 )
    _, th = cv2.threshold( diff, 70, 255, cv2.THRESH_BINARY )

    morph_kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5, 5) )
    img_bin = cv2.erode( th, morph_kernel, iterations = 2 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 25 )
    img_bin = cv2.erode( img_bin, morph_kernel, iterations = 30 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 40 )

    cv2.imshow( 'th', cv2.resize( th, (640, 360) ) )
    cv2.imshow( 'img_bin', cv2.resize( img_bin, (640, 360) ) )

    return th, img_bin

def get_diff_02( img0, img1 ):
    gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
    gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
    diff = cv2.absdiff( gray0, gray1 )
    _, th = cv2.threshold( diff, 70, 255, cv2.THRESH_BINARY )

    morph_kernel = cv2.getStructuringElement( cv2.MORPH_RECT, (5, 5) )
    img_bin = cv2.erode( th, morph_kernel, iterations = 2 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 25 )
    img_bin = cv2.erode( img_bin, morph_kernel, iterations = 30 )
    img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 40 )
    cv2.imshow( 'th', cv2.resize( th, (640, 360) ) )
    cv2.imshow( 'img_bin', cv2.resize( img_bin, (640, 360) ) )
    return th, img_bin


def main1():
    img_src_dirpath0 = r'D:/deal_pics/2018-01-09/'
    img_src_dirpath1 = r'D:/deal_pics/2018-01-08/'
    video_dst = r'E:/20180109.mp4'
    files = [f for f in os.listdir( img_src_dirpath1 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath1 + files[0] )
    height0, width0, _ = img0.shape
    width1, height1 = int( width0 / 2 ), int( height0 / 2 )
    fourcc = cv2.VideoWriter_fourcc( *'DIVX' )
    fps = 30.0
    vwriter = cv2.VideoWriter( video_dst, fourcc, fps, (width0, height1) )
    img_num = len( files )
    diff_step = 100
    waitSec = 1
    for i in range( img_num ):
        f = files[i]
        img1 = cv2.imread( img_src_dirpath1 + f )
        img2 = cv2.imread(img_src_dirpath0 + f)
        if i >= diff_step:
            img0 = cv2.imread( img_src_dirpath1 + files[i - diff_step] )
        else:
            img0 = img1.copy()
        th, img_bin = get_diff( img0, img1 )
        img3 = cv2.bitwise_and( img1, img1, mask = img_bin )
        _, contours, hierarchy = cv2.findContours( img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        for ct in contours:
            x, y, w, h = cv2.boundingRect( ct )
            cv2.rectangle( img2, (x, y), (x + w, y + h), (0, 255, 255), 5 )
        img = np.zeros( (height1, width0, 3), dtype = np.uint8 )
        img[:height1, :width1, :] = cv2.resize( img2, (width1, height1) )
        img[:height1, width1:width0, :] = cv2.resize( img3, (width1, height1) )
        vwriter.write( img )
        cv2.putText( img3, f, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
        cv2.imshow( 'img0', cv2.resize( img0, (640, 360) ) )
        cv2.imshow( 'img1', cv2.resize( img1, (640, 360) ) )
        cv2.imshow( 'img3', cv2.resize( img3, (640, 360) ) )
        cv2.imshow( 'img', cv2.resize( img, (1280, 360) ) )
        key = cv2.waitKey( waitSec ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec = 1 - waitSec
    vwriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main1()

'''

'''
import cv2

import numpy as np


def main():
    video_dst = r'E:/test.mp4'

    fourcc = cv2.VideoWriter_fourcc( *'DIVX' )
    fps = 30.0
    width, height = 1280, 720
    vwriter = cv2.VideoWriter( video_dst, fourcc, fps, (width, height) )

    frame_num = 900
    color_bg = 128
    color_prod = 50
    color_person = 200
    pos_proda = (300, 200)
    pos_prodb = (780, 200)
    pos_person = (-200, 200)
    size_proda = (200, 200)
    size_prodb = (200, 200)
    size_person = (200, 200)
    person_step = 1.8

    proda_right = pos_proda[0] + size_proda[0]
    prodb_right = pos_prodb[0] + size_prodb[0]
    for i in range( frame_num ):
        frame = np.full( (height, width), color_bg, dtype = np.uint8 )

        person_left = int( pos_person[0] + i * person_step + 0.5 )
        person_right = person_left + size_person[0]
        person_left = min( max( person_left, 0 ), width )
        person_right = max( min( person_right, width ), 0 )
        person_top = pos_person[1]
        person_bottom = pos_person[1] + size_person[1]

        if person_right < proda_right:
            cv2.rectangle( frame, pos_proda, (pos_proda[0] + size_proda[0], pos_proda[1] + size_proda[1]), color_prod,
                           -1 )
        if person_right < prodb_right:
            cv2.rectangle( frame, pos_prodb, (pos_prodb[0] + size_prodb[0], pos_prodb[1] + size_prodb[1]), color_prod,
                           -1 )

        cv2.rectangle( frame, (person_left, person_top), (person_right, person_bottom), color_person, -1 )

        frame = cv2.cvtColor( frame, cv2.COLOR_GRAY2BGR )
        vwriter.write( frame )

        cv2.imshow( 'frame', frame )
        key = cv2.waitKey( 1 ) & 0xff
        if key == 27:
            break

    cv2.destroyAllWindows()
    vwriter.release()


if __name__ == '__main__':
    main()

'''

'''

import cv2
import os
import  time

print(cv2.ocl.haveOpenCL())#True
cv2.ocl.setUseOpenCL(True)
print(cv2.ocl.useOpenCL())#True
img_root = r'D:/deal_pics/2018-01-09/'
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
im_names = os.listdir(img_root)
for im_name in im_names:
    img = cv2.imread(img_root+im_name)
    UmatImg = cv2.UMat(img)
    img_1 = UmatImg.get()
    detector = cv2.ORB_create()
    keypoints = detector.detect(img_1)
    out = cv2.drawKeypoints(img_1, keypoints, None)
    cv2.imshow('10',out)
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()))
'''



'''
import numpy as np
import cv2

cap = cv2.VideoCapture('E:/20171204a.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret,frame = cap.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    else:
        break
cv2.destroyAllWindows()
cap.release()
'''

'''
import cv2
import numpy as np
cap = cv2.VideoCapture(r"E:\OpenCV\opencv\sources\samples\data\vtest.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while True:
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2',rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png',frame2)
            cv2.imwrite('opticalhsv.png',rgb)
        prvs = next
    else:
        break

cap.release()
cv2.destroyAllWindows()
'''


'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

VIDEO_PATH = r"E:\OpenCV\opencv\sources\samples\data\domo_running.mp4"

# params for ShiTomasi corner detection
FEATURE_COUNT = 100
FEATURE_PARAMS = dict(
    maxCorners=FEATURE_COUNT,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for lucas kanade optical flow
LK_PARAMS = dict(
    winSize  = (15,15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)
)

# color for drawing (create FEATURE_COUNT colors each of these is RGB color)
COLOR = np.random.randint(0, 255,(FEATURE_COUNT, 3))


def frames():
    """
    Get frames from video
    """

    video = cv2.VideoCapture(VIDEO_PATH)
    success = True

    while (success):
        success, current = video.read()
        if not success:
            break
        yield current
    else:
        video.release()


def optical_flow():
    """
    Detect and Draw Optical Flow
    """

    get_features = lambda f: cv2.goodFeaturesToTrack(f, mask=None, **FEATURE_PARAMS)
    to_grayscale = lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    previous_is_set = False
    p_frame = None
    p_features = []
    mask = []

    for frame in frames():
        glayed = to_grayscale(frame)

        if not previous_is_set:
            p_frame = glayed
            p_features = get_features(p_frame)
            mask = np.zeros_like(frame)

            if p_features is not None:
                previous_is_set = True
            else:
                continue

        else:
            # calculate optical flow
            c_features, st, err = cv2.calcOpticalFlowPyrLK(p_frame, glayed, p_features, None, **LK_PARAMS)

            if c_features is None:
                continue
            else:
                # select good points (st = 1 if tracking is successed)
                tracked = c_features[st == 1]
                tracked_in_previous = p_features[st == 1]

                # draw line
                for i, (current, previous) in enumerate(zip(tracked, tracked_in_previous)):
                    x1, y1 = current.ravel()
                    x0, y0 = previous.ravel()
                    mask = cv2.line(mask, (x1, y1), (x0, y0), COLOR[i].tolist(), 2)
                    frame = cv2.circle(frame, (x1, y1), 5, COLOR[i].tolist(), -1)

                img = cv2.add(frame, mask)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV has color BGR
                yield img

                p_frame = glayed.copy()
                p_features = c_features.reshape(-1, 1, 2)

def draw(flow_func):
    fig = plt.figure()
    ims = []

    for f in flow_func():
        ims.append([plt.imshow(f)])

    ani = animation.ArtistAnimation(fig, ims)
    plt.axis("off")
    plt.show()

draw(optical_flow)
'''



'''

import os

import cv2
import numpy as np


def merge_video( img_src_dirpath0, img_src_dirpath1, img_dst_dirpath1, video_dst0, draw_step0 ):
    if not os.path.exists( img_dst_dirpath1 ):
        os.makedirs( img_dst_dirpath1 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape
    width1, height1 = int( width0 / 2 ), int( height0 / 2 )
    fourcc0 = cv2.VideoWriter_fourcc( *'DIVX' )
    fps0 = 30.0

    vwriter0 = cv2.VideoWriter( video_dst0, fourcc0, fps0, (width0, height1) )
    img_num0 = len( files0 )
    # draw_step0 = 0
    waitSec0 = 1
    for i in range( img_num0 ):
        idx0 = i + draw_step0
        img0 = cv2.imread( img_src_dirpath0 + files0[i] )
        img0 = cv2.resize( img0, (width1, height1) )
        img_dst0 = np.zeros( (height1, width0, 3), dtype = np.uint8 )
        img_dst0[:height1, :width1, :] = img0

        if idx0 < 0:
            pass
        elif idx0 < img_num0:
            filename = img_src_dirpath1 + '%04d.jpg' % idx0
            if os.path.exists( filename ):
                img1 = cv2.imread( filename )
                img1 = cv2.resize( img1, (width1, height1) )
                img_dst0[:height1, width1:width0, :] = img1

                cv2.imwrite( img_dst_dirpath1 + '%04d.jpg' % i, img_dst0 )
        else:
            pass

        vwriter0.write( img_dst0 )

        cv2.imshow( 'img_dst0', cv2.resize( img_dst0, (1280, 360) ) )
        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0

    vwriter0.release()
    cv2.destroyAllWindows()


def bg_dense( img_src_dirpath0, img_dst_dirpath0, diff_step0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape

    hsv0 = np.zeros_like( img0 )
    hsv0[..., 1] = 255

    img_num0 = len( files0 )
    # diff_step0 = 60
    waitSec0 = 1
    for i in range( img_num0 ):
        if i >= diff_step0:
            idx0 = i - diff_step0
            img0 = cv2.imread( img_src_dirpath0 + files0[i] )
            img1 = cv2.imread( img_src_dirpath0 + files0[idx0] )

            gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
            gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )

            flow = cv2.calcOpticalFlowFarneback( gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
            mag, ang = cv2.cartToPolar( flow[..., 0], flow[..., 1] )
            hsv0[..., 0] = ang * 180 / np.pi / 2
            hsv0[..., 2] = cv2.normalize( mag, None, 0, 255, cv2.NORM_MINMAX )
            bgr = cv2.cvtColor( hsv0, cv2.COLOR_HSV2BGR )

            cv2.imwrite( img_dst_dirpath0 + '%04d.jpg' % i, bgr )

            cv2.putText( img0, str( i ), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
            cv2.imshow( 'img0', cv2.resize( img0, (640, 360) ) )
            cv2.imshow( 'img1', cv2.resize( img1, (640, 360) ) )
            cv2.imshow( 'bgr', cv2.resize( bgr, (640, 360) ) )

        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-08' + os.sep
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-16_dense' + os.sep
    diff_step0 = 60
    bg_dense( img_src_dirpath0, img_dst_dirpath0, diff_step0 )

    img_src_dirpath1 = r'D:\deal_pics\2018-01-09' + os.sep
    img_dst_dirpath1 = r'D:\deal_pics\2018-01-16_dense1' + os.sep
    video_dst0 = r'E:/20180116.mp4'
    draw_step0 = 60
    merge_video( img_src_dirpath1, img_dst_dirpath0, img_dst_dirpath1, video_dst0, draw_step0 )


if __name__ == '__main__':
    main()

'''


'''
import os

import cv2
import numpy as np


def merge_video( img_src_dirpath0, img_src_dirpath1, img_dst_dirpath0, video_dst0 ):

    # :param img_src_dirpath0: bg
    # :param img_src_dirpath1: img_diff0
    # :param video_dst0:
    # :return: bg + img_diff0

    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape
    fourcc0 = cv2.VideoWriter_fourcc( *'DIVX' )
    fps0 = 30.0

    vwriter0 = cv2.VideoWriter( video_dst0, fourcc0, fps0, (width0, height0) )
    img_num0 = len( files0 )
    threshval0 = 10
    threshval1 = 60
    draw_step0 = 60
    waitSec0 = 1
    for i in range( img_num0 ):
        idx0 = i + draw_step0
        img0 = cv2.imread( img_src_dirpath0 + files0[i] )
        img_dst0 = img0.copy()

        if idx0 < 0:
            pass
        elif idx0 < img_num0:
            filename = img_src_dirpath1 + '%04d.jpg' % idx0
            if os.path.exists( filename ):
                img_diff0 = cv2.imread( filename, 0 )

                # 0-10
                img_diff1 = np.zeros( (height0, width0), dtype = np.uint8 )
                img_diff1[np.where( img_diff0 <= threshval0 )] = 255
                img1 = cv2.bitwise_and( img0, img0, mask = img_diff1 )

                # 11-60
                img_diff2 = np.full( (height0, width0), 255, dtype = np.uint8 )
                img_diff2[np.where( img_diff0 <= threshval0 )] = 0
                img_diff2[np.where( img_diff0 > threshval1 )] = 0
                img2 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img2[:, :, 0] = img_diff2

                # 61-255
                img_diff3 = np.zeros( (height0, width0), dtype = np.uint8 )
                img_diff3[np.where( img_diff0 > threshval1 )] = 255
                img3 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img3[:, :, 2] = img_diff3

                img_dst0 = cv2.add( img1, img2 )
                img_dst0 = cv2.add( img_dst0, img3 )

        else:
            pass

        cv2.imwrite( img_dst_dirpath0 + '%04d.jpg' % i, img_dst0 )
        vwriter0.write( img_dst0 )

        cv2.putText( img_dst0, str( i ), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
        cv2.imshow( 'img_dst0', cv2.resize( img_dst0, (640, 360) ) )
        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0

    vwriter0.release()
    cv2.destroyAllWindows()


def img_diff( img_src_dirpath0, img_dst_dirpath0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape

    img_num0 = len( files0 )
    diff_step0 = 100
    waitSec0 = 1
    for i in range( img_num0 ):
        if i >= diff_step0:
            idx0 = i - diff_step0
            img0 = cv2.imread( img_src_dirpath0 + files0[i] )
            img1 = cv2.imread( img_src_dirpath0 + files0[idx0] )

            img_gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
            img_gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
            img_diff0 = cv2.absdiff( img_gray0, img_gray1 )

            cv2.imwrite( img_dst_dirpath0 + '%04d.jpg' % i, img_diff0 )

        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-08' + os.sep #动体去除后的照片
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-17_diff' + os.sep

    img_diff( img_src_dirpath0, img_dst_dirpath0 )

    img_src_dirpath1 = r'D:\deal_pics\2018-01-09' + os.sep #有人的照片
    img_dst_dirpath1 = r'D:\deal_pics\2018-01-17_merge' + os.sep
    video_dst0 = r'E:/20180117.mp4'
    merge_video( img_src_dirpath1, img_dst_dirpath0, img_dst_dirpath1, video_dst0 )


if __name__ == '__main__':
    main()

'''

'''
import cv2
import time

def img_multi_thresh( img_gray, thresholds, colors ):
    time_start = time.time()
    if len( img_gray.shape ) != 2:
        print( 'Input image is not gray image. ' )
    if len( thresholds ) != len( colors ):
        print( 'Thresholds and colors must be same size.' )

    loop_count = len( thresholds )

    img_exec = img_gray.copy()

    for counter in range( loop_count ):
        min = thresholds[counter][0]
        max = thresholds[counter][1]
        min_bool = img_gray >= min
        max_bool = img_gray <= max
        range_bool = min_bool * max_bool
        img_exec[ range_bool ] = counter

    img_bgr = cv2.cvtColor( img_exec, cv2.COLOR_GRAY2BGR )
    for counter in range( loop_count ):
        print(img_bgr[ :, :, 0 ])
        img_bgr[ img_bgr[ :, :, 0 ] == counter ] = colors[ counter ]


    time_end = time.time()
    print( 'Method img_thresh_exec time: %fs' % (time_end-time_start) )

    return img_bgr

if __name__ == '__main__':
    img = cv2.imread( r'D:\deal_pics\2018-01-09\0032.jpg' )
    img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    thresholds = [ [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 255] ]
    black = [ 0, 0, 0 ]
    blue = [ 0, 0, 255 ]
    red = [ 255, 0, 0 ]
    green = [ 0, 255, 0 ]
    cyan = [ 0, 255, 255 ]
    yellow = [ 255, 255, 0 ]
    magenta = [ 255, 0, 255 ]
    white = [ 255, 255, 255 ]
    colors = [ black, blue, red, green, cyan, yellow, magenta, white ]
    img_thresh = img_multi_thresh( img_gray, thresholds, colors )
    cv2.imshow( 'img_thresh', img_thresh )
    cv2.waitKey( 0 )

'''

'''
import cv2

img1 = cv2.imread( r'D:\deal_pics\2018-01-17\0001.jpg' )
img2 = cv2.imread( r'D:\deal_pics\2018-01-17\S0392_0001_Capture_20171223103626464.jpg' )
img_gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
img_gray2 = cv2.cvtColor( img2, cv2.COLOR_BGR2GRAY )
img_diff = cv2.absdiff( img_gray1, img_gray2 )

testImg = img1 = cv2.imread( r'D:\deal_pics\2018-01-17\video_gray.jpg' )
_, th = cv2.threshold( testImg, 15, 255, cv2.THRESH_BINARY )
morph_kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5, 5) )
img_bin = cv2.erode( th, morph_kernel, iterations = 3 )
img_bin = cv2.dilate( img_bin, morph_kernel, iterations = 3 )

maxValue = 0
height,width,_=img1.shape
# for i in range(height):
#     for j in range(width):
#         # print(img1[i,j])
#         if int(img_diff[i,j]) > maxValue:
#             max = int(img_diff[i,j])
# print(max)
cv2.imshow('1',cv2.resize(img_bin,(800,600)))
# cv2.imwrite(r'D:\deal_pics\2018-01-17\videv_diff_pic.jpg',img_diff)
cv2.waitKey(0)

'''


'''
import os
import cv2
import numpy as np


def merge_video( img_src_dirpath0, img_src_dirpath1, img_dst_dirpath1, video_dst0, draw_step0 ):
    if not os.path.exists( img_dst_dirpath1 ):
        os.makedirs( img_dst_dirpath1 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img_num0 = len( files0 )
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape
    fourcc0 = cv2.VideoWriter_fourcc( *'DIVX' )
    fps0 = 30.0

    vwriter0 = cv2.VideoWriter( video_dst0, fourcc0, fps0, (width0, height0) )
    threshval0 = 1
    threshval1 = 3
    threshval2 = 5
    threshval3 = 7
    threshval4 = 9
    threshval5 = 11
    threshval6 = 13
    # draw_step0 = 0
    waitSec0 = 1
    for i in range( img_num0 ):
        idx0 = i + draw_step0
        img0 = cv2.imread( img_src_dirpath0 + files0[i] )
        img_dst0 = img0.copy()

        if idx0 < 0:
            pass
        elif idx0 < img_num0:
            filename = img_src_dirpath1 + '%04d.jpg' % idx0
            if os.path.exists( filename ):
                dense0 = cv2.imread( filename, 0 )

                # 0-1
                dense1 = np.zeros( (height0, width0), dtype = np.uint8 )
                dense1[np.where( dense0 <= threshval0 )] = 255
                img1 = cv2.bitwise_and( img0, img0, mask = dense1 )

                # 2-3
                dense2 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense2[np.where( dense0 <= threshval0 )] = 0
                dense2[np.where( dense0 > threshval1 )] = 0
                img2 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img2[:, :, 0] = dense2

                # 4-5
                dense3 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense3[np.where( dense0 <= threshval1 )] = 0
                dense3[np.where( dense0 > threshval2 )] = 0
                img3 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img3[:, :, 2] = dense3

                # 6-7
                dense4 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense4[np.where( dense0 <= threshval2 )] = 0
                dense4[np.where( dense0 > threshval3 )] = 0
                img4 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img4[:, :, 1] = dense4

                # 8-9
                dense5 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense5[np.where( dense0 <= threshval3 )] = 0
                dense5[np.where( dense0 > threshval4 )] = 0
                img5 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img5[:, :, 0] = dense5
                img5[:, :, 1] = dense5

                # 10-11
                dense6 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense6[np.where( dense0 <= threshval4 )] = 0
                dense6[np.where( dense0 > threshval5 )] = 0
                img6 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img6[:, :, 1] = dense6
                img6[:, :, 2] = dense6

                # 12-13
                dense7 = np.full( (height0, width0), 255, dtype = np.uint8 )
                dense7[np.where( dense0 <= threshval5 )] = 0
                dense7[np.where( dense0 > threshval6 )] = 0
                img7 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
                img7[:, :, 0] = dense7
                img7[:, :, 2] = dense7

                # 14-
                dense8 = np.zeros( (height0, width0), dtype = np.uint8 )
                dense8[np.where( dense0 > threshval6 )] = 255
                img8 = cv2.cvtColor( dense8, cv2.COLOR_GRAY2BGR )

                img_dst0 = cv2.add( img1, img2 )
                img_dst0 = cv2.add( img_dst0, img3 )
                img_dst0 = cv2.add( img_dst0, img4 )
                img_dst0 = cv2.add( img_dst0, img5 )
                img_dst0 = cv2.add( img_dst0, img6 )
                img_dst0 = cv2.add( img_dst0, img7 )
                img_dst0 = cv2.add( img_dst0, img8 )
                cv2.imwrite( img_dst_dirpath1 + '%04d.jpg' % i, img_dst0 )
        else:
            pass

        vwriter0.write( img_dst0 )

        cv2.putText( img_dst0, str( i ), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
        cv2.imshow( 'img_dst0', cv2.resize( img_dst0, (640, 360) ) )
        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0

    vwriter0.release()
    cv2.destroyAllWindows()


def bg_dense( img_src_dirpath0, img_dst_dirpath0, diff_step0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    height0, width0, _ = img0.shape

    hsv0 = np.zeros_like( img0 )
    hsv0[..., 1] = 255

    img_num0 = len( files0 )
    # diff_step0 = 60
    waitSec0 = 1
    for i in range( img_num0 ):
        if i >= diff_step0:
            idx0 = i - diff_step0
            img0 = cv2.imread( img_src_dirpath0 + files0[i] )
            img1 = cv2.imread( img_src_dirpath0 + files0[idx0] )

            gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
            gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )

            flow = cv2.calcOpticalFlowFarneback( gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
            mag, ang = cv2.cartToPolar( flow[..., 0], flow[..., 1] )
            hsv0[..., 0] = ang * 180 / np.pi / 2
            hsv0[..., 2] = cv2.normalize( mag, None, 0, 255, cv2.NORM_MINMAX )
            bgr = cv2.cvtColor( hsv0, cv2.COLOR_HSV2BGR )

            cv2.imwrite( img_dst_dirpath0 + '%04d.jpg' % i, bgr )

            cv2.putText( img0, str( i ), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 255, 5 )
            cv2.imshow( 'img0', cv2.resize( img0, (640, 360) ) )
            cv2.imshow( 'img1', cv2.resize( img1, (640, 360) ) )
            cv2.imshow( 'bgr', cv2.resize( bgr, (640, 360) ) )

        key = cv2.waitKey( waitSec0 ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec0 = 1 - waitSec0


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-08' + os.sep  # 动体去除后的照片
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-17_diff' + os.sep
    diff_step0 = 100
    # bg_dense( img_src_dirpath0, img_dst_dirpath0, diff_step0 )

    img_src_dirpath1 = r'D:\deal_pics\2018-01-09' + os.sep  # 有人的照片
    img_dst_dirpath1 = r'D:\deal_pics\2018-01-17_merge' + os.sep
    video_dst0 = r'E:/20180118.mp4'
    draw_step0 = 66
    merge_video( img_src_dirpath1, img_dst_dirpath0, img_dst_dirpath1, video_dst0, draw_step0 )


if __name__ == '__main__':
    main()

'''


'''
import cv2
import numpy as np

threshval0 = 1
threshval1 = 3
threshval2 = 5
threshval3 = 7
threshval4 = 9
threshval5 = 11
threshval6 = 13

img0 = cv2.imread( r'D:/deal_pics/2018-01-08_BG/0146.jpg' ) #动体去除后的照片
img1 = cv2.imread( r'D:/deal_pics/2018-01-08_BG/0246.jpg' ) #动体去除后的照片
gray0 = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
gray1 = cv2.cvtColor( img1, cv2.COLOR_BGR2GRAY )
height0, width0, _ = img0.shape
hsv0 = np.zeros_like( img0 )
hsv0[..., 1] = 255

flow = cv2.calcOpticalFlowFarneback( gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0 )
mag, ang = cv2.cartToPolar( flow[..., 0], flow[..., 1] )
hsv0[..., 0] = ang * 180 / np.pi / 2
hsv0[..., 2] = cv2.normalize( mag, None, 0, 255, cv2.NORM_MINMAX )
bgr = cv2.cvtColor( hsv0, cv2.COLOR_HSV2BGR )

dense0 = cv2.cvtColor( bgr, cv2.COLOR_BGR2GRAY )

# 2-3
dense2 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense2[np.where( dense0 <= threshval0 )] = 0
dense2[np.where( dense0 > threshval1 )] = 0
img2 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img2[:, :, 0] = dense2

# 4-5
dense3 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense3[np.where( dense0 <= threshval1 )] = 0
dense3[np.where( dense0 > threshval2 )] = 0
img3 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img3[:, :, 2] = dense3

# 6-7
dense4 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense4[np.where( dense0 <= threshval2 )] = 0
dense4[np.where( dense0 > threshval3 )] = 0
img4 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img4[:, :, 1] = dense4

# 8-9
dense5 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense5[np.where( dense0 <= threshval3 )] = 0
dense5[np.where( dense0 > threshval4 )] = 0
img5 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img5[:, :, 0] = dense5
img5[:, :, 1] = dense5

# 10-11
dense6 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense6[np.where( dense0 <= threshval4 )] = 0
dense6[np.where( dense0 > threshval5 )] = 0
img6 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img6[:, :, 1] = dense6
img6[:, :, 2] = dense6

# 12-13
dense7 = np.full( (height0, width0), 255, dtype = np.uint8 )
dense7[np.where( dense0 <= threshval5 )] = 0
dense7[np.where( dense0 > threshval6 )] = 0
img7 = np.zeros( (height0, width0, 3), dtype = np.uint8 )
img7[:, :, 0] = dense7
img7[:, :, 2] = dense7

# 14-
dense8 = np.zeros( (height0, width0), dtype = np.uint8 )
dense8[np.where( dense0 > threshval6 )] = 255
img8 = cv2.cvtColor( dense8, cv2.COLOR_GRAY2BGR )

# img_dst0 = cv2.add( img1, img2 )
img_dst0 = cv2.add( img2, img3 )
img_dst0 = cv2.add( img_dst0, img4 )
img_dst0 = cv2.add( img_dst0, img5 )
img_dst0 = cv2.add( img_dst0, img6 )
img_dst0 = cv2.add( img_dst0, img7 )
img_dst0 = cv2.add( img_dst0, img8 )


cv2.imshow( 'img0', cv2.resize( img0, (640, 360) ) )
cv2.imshow( 'img1', cv2.resize( img1, (640, 360) ) )
cv2.imshow( 'bgr', cv2.resize( bgr, (640, 360) ) )
cv2.imshow( 'img_dst0', cv2.resize( img_dst0, (640, 360) ) )

cv2.waitKey( 0 )
cv2.destroyAllWindows()

'''


'''
import os

import cv2
import numpy as np


def img_time_guass( img_src_dirpath0, pts ):
    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    img_num0 = len( files0 )

    win0 = 59
    win0_half0 = int( (win0 - 1) / 2 )
    loop_num0 = img_num0 + win0 - 1
    pts0 = np.array( pts )
    pts0_num = len( pts0 )

    img0 = cv2.imread( img_src_dirpath0 + files0[0], 0 )
    height0, width0 = img0.shape
    img_empty0 = np.zeros_like( img0 )

    dump_src0 = np.zeros( (pts0_num, img_num0), dtype = np.uint8 )
    dump_dst0 = np.zeros( (pts0_num, img_num0), dtype = np.uint8 )
    imgs0 = np.zeros( (win0, height0, width0), dtype = np.uint8 )
    for i in range( loop_num0 ):
        idx0 = i - win0_half0
        idx1 = i - win0 + 1
        idx2 = i % win0
        if idx0 < 0:
            imgs0[idx2] = img_empty0.copy()
        elif idx0 < img_num0:
            img0 = cv2.imread( img_src_dirpath0 + files0[idx0], 0 )
            imgs0[idx2] = img0
            dump_src0[:, idx0] = img0[pts0[:, 1], pts0[:, 0]]
        else:
            imgs0[idx2] = img_empty0.copy()

        if 0 <= idx1:
            src0 = imgs0[:, pts0[:, 1], pts0[:, 0]]
            dump_dst0[:, idx1] = np.uint8( np.average( src0, axis = 0 ) + 0.5 )

    for i in range( pts0_num ):
        with open( r'E:/gauss_dump_%d_%d.csv' % pts[i], 'w' ) as file:
            file.write( 'dumpXY: %d_%d\n' % pts[i] )
            print( 'dumpXY: %d_%d' % pts[i] )
            file.write( 'src\n' )
            print( 'src' )
            for j in range( img_num0 ):
                file.write( '%d\n' % dump_src0[i, j] )
                print( dump_src0[i, j] )
            file.write( 'dst\n' )
            print( '\ndst' )
            for j in range( img_num0 ):
                file.write( '%d\n' % dump_dst0[i, j] )
                print( dump_dst0[i, j] )


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-22_dense_gauss_bin1' + os.sep

    pts = [(873, 1401), (1905, 913), (2165, 849), (3093, 1285), (2949, 889), (865, 897)]
    img_time_guass( img_src_dirpath0, pts )


if __name__ == '__main__':
    main()
'''



'''
import os
import cv2

def draw_rect( img_src_dirpath0, img_dst_dirpath0, rect_file_src0, draw_step0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    rects = []

    with open( rect_file_src0, 'r' ) as file:
        rect_lines = file.readlines()
    for line in rect_lines:
        if line != '':
            idx, x, y, w, h = line.split( ',' )
            rects.append( (int( idx ), int( x ), int( y ), int( w ), int( h )) )

    rect_idx0 = 0
    rect_num0 = len( rects )
    idx1 = rects[0][0]
    for i in range( len( files0 ) ):
        idx0 = i + draw_step0
        f = files0[i]
        img0 = cv2.imread( img_src_dirpath0 + f )
        while idx1 <= idx0:
            if idx0 == idx1:
                _, x, y, w, h = rects[rect_idx0]
                cv2.rectangle( img0, (x, y), (x + w, y + h), (0, 255, 0), 4 )

            if rect_idx0 < (rect_num0 - 1):
                rect_idx0 += 1
                idx1 = rects[rect_idx0][0]
            else:
                break
        cv2.imshow('2', cv2.resize(img0,(800,600)))
        cv2.waitKey(1)
        cv2.imwrite( img_dst_dirpath0 + f, img0 )


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-22_dense_gauss_bin1' + os.sep
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-23_area' + os.sep
    rect_file_dst0 = r'E:/ct_rect.csv'
    thresharea0 = 10000

    # img_ctarea( img_src_dirpath0, img_dst_dirpath0, rect_file_dst0, thresharea0 )

    img_src_dirpath0 = r'D:\deal_pics\2018-01-09_SRC' + os.sep
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-23' + os.sep
    rect_file_src0 = rect_file_dst0
    draw_step0 = 66

    draw_rect( img_src_dirpath0, img_dst_dirpath0, rect_file_src0, draw_step0 )


if __name__ == '__main__':
    main()

'''

'''
import os
import cv2
import numpy as np

def img_ctarea( img_src_dirpath0, img_dst_dirpath0, rect_file_dst0, thresharea0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    rects = []
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (5, 5) )

    img0 = cv2.imread( img_src_dirpath0 + files0[0] )
    img_empty0 = np.zeros_like( img0 )

    for i in range( len( files0 ) ):
        f = files0[i]
        img0 = cv2.imread( img_src_dirpath0 + f, 0 )
        _, img0 = cv2.threshold( img0, 100, 255, cv2.THRESH_BINARY )
        img1 = img_empty0.copy()

        img0 = cv2.erode( img0, kernel, iterations = 5 )
        img0 = cv2.dilate( img0, kernel, iterations = 5 )

        _, contours, hierarchy = cv2.findContours( img0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        for ct in contours:
            area = cv2.contourArea( ct )
            if thresharea0 <= area:
                color = (0, 0, 255)
                cv2.drawContours( img1, [ct], -1, color, -1 )
                x, y, w, h = cv2.boundingRect( ct )
                rects.append( (i, x, y, w, h) )
                cv2.rectangle( img1, (x, y), (x + w, y + h), (0, 255, 0), 4 )
        cv2.imwrite( img_dst_dirpath0 + f, img1 )
    with open( rect_file_dst0, 'w' ) as file:
        for idx, x, y, w, h in rects:
            file.write( '%d,%d,%d,%d,%d\n' % (idx, x, y, w, h) )


def draw_rect( img_src_dirpath0, img_dst_dirpath0, rect_file_src0, draw_step0 ):
    if not os.path.exists( img_dst_dirpath0 ):
        os.makedirs( img_dst_dirpath0 )

    files0 = [f for f in os.listdir( img_src_dirpath0 ) if f.endswith( '.jpg' )]
    rects = []

    with open( rect_file_src0, 'r' ) as file:
        rect_lines = file.readlines()
    for line in rect_lines:
        if line != '':
            idx, x, y, w, h = line.split( ',' )
            rects.append( (int( idx ), int( x ), int( y ), int( w ), int( h )) )

    rect_idx0 = 0
    rect_num0 = len( rects )
    idx1 = rects[0][0]
    for i in range( len( files0 ) ):
        idx0 = i + draw_step0
        f = files0[i]
        img0 = cv2.imread( img_src_dirpath0 + f )
        while idx1 <= idx0:
            if idx0 == idx1:
                _, x, y, w, h = rects[rect_idx0]
                cv2.rectangle( img0, (x, y), (x + w, y + h), (0, 255, 0), 4 )

            if rect_idx0 < (rect_num0 - 1):
                rect_idx0 += 1
                idx1 = rects[rect_idx0][0]
            else:
                break

        cv2.imwrite( img_dst_dirpath0 + f, img0 )


def main():
    img_src_dirpath0 = r'D:\deal_pics\2018-01-08_BG' + os.sep
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-24_getGoods' + os.sep
    rect_file_dst0 = r'E:/ct_rect.csv'
    thresharea0 = 10000

    # img_ctarea( img_src_dirpath0, img_dst_dirpath0, rect_file_dst0, thresharea0 )

    img_src_dirpath0 = r'D:\deal_pics\2018-01-08_BG' + os.sep
    img_dst_dirpath0 = r'D:\deal_pics\2018-01-24_getGoods' + os.sep
    rect_file_src0 = rect_file_dst0
    draw_step0 = 66

    draw_rect( img_src_dirpath0, img_dst_dirpath0, rect_file_src0, draw_step0 )


if __name__ == '__main__':
    main()
'''

'''
import  cv2
import os
import numpy as np

src_imgs = [src_f for src_f in os.listdir(r'D:/deal_pics/2018-01-09_SRC')]
img0 = cv2.imread( r'D:/deal_pics/2018-01-09_SRC/0000.jpg')
hsv0 = np.zeros_like( img0 )
hsv0[..., 1] = 255
for i in range(840):
    step = 59
    f1 = r'D:/deal_pics/2018-01-09_SRC/'+src_imgs[i]
    f_step = r'D:/deal_pics/2018-01-09_SRC/'+src_imgs[i+step]
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f_step)
    # diff = cv2.absdiff(img1,img2)
    # gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    # _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    gray0 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray0, gray1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv0[..., 0] = ang * 180 / np.pi / 2
    hsv0[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv0, cv2.COLOR_HSV2BGR)
    # cv2.imshow('d',cv2.resize(bgr,(800,600)))
    cv2.imwrite(r'D:/deal_pics/2018-01-24_getGoods/' + '%04d.jpg' % i, bgr)
    key = cv2.waitKey(1) & 0xff
    if key == 27:
        break

'''

'''
import cv2
import numpy as np
import os

def count_pixel( count_array, img_gray, thresh_val = 10 ):
    img_gray_copy = img_gray.copy()
    count_array_copy = count_array.copy()
    img_gray_copy[ img_gray_copy <= thresh_val ] = 0
    img_gray_copy[ img_gray_copy > thresh_val] = 1
    count_array_copy = count_array_copy + img_gray_copy
    return count_array_copy

def count_pixel_zero( count_array, img_gray, thresh_val = 10 ):
    img_gray_copy = img_gray.copy()
    count_array_copy = count_array.copy()
    img_gray_copy[ img_gray_copy <= thresh_val ] = 0
    img_gray_copy[ img_gray_copy > thresh_val] = 1
    count_array_copy = count_array_copy * img_gray_copy + img_gray_copy
    return count_array_copy

def heatmap( src0, src1, dst0, dst1, thresh_val = 10, alpha = 0.4 ):
    imgs_src0 = [os.path.join( src0, f ) for f in os.listdir( src0 ) if f.lower().endswith( '.jpg' )]
    imgs_src1 = [os.path.join( src1, f ) for f in os.listdir( src1 ) if f.lower().endswith( '.jpg' )]

    if not os.path.exists( dst0 ):
        os.makedirs( dst0 )

    if not os.path.exists( dst1 ):
        os.makedirs( dst1 )

    # count_arrays = [ None for i in range( len( imgs_src0 ) ) ]
    count_array = None

    for i in range( len( imgs_src0 ) ):
        print( i )
        img0_gray = cv2.imread( imgs_src0[i], 0 )
        if i == 0:
            count_array = np.zeros_like( img0_gray )
        count_array = count_pixel( count_array, img0_gray, thresh_val )

    max_count = count_array.max()

    count_pixel_zero_array = None
    for i in range( len( imgs_src0 ) ):
        print( i )
        img0 = cv2.imread( imgs_src0[i] )

        img0_gray = cv2.cvtColor( img0, cv2.COLOR_BGR2GRAY )
        img1 = cv2.imread( imgs_src1[i] )
        img1_copy = img1.copy()
        if i == 0:
            count_pixel_zero_array = np.zeros_like( img0_gray )
        count_pixel_zero_array = count_pixel_zero( count_pixel_zero_array, img0_gray, thresh_val )
        count_pixel_zero_normalize = ( count_pixel_zero_array * ( 255 / max_count ) ).astype( np.uint8 )

        count_pixel_zero_normalize_bgr = cv2.cvtColor( count_pixel_zero_normalize, cv2.COLOR_GRAY2BGR )
        blank_color = cv2.applyColorMap( count_pixel_zero_normalize_bgr, cv2.COLORMAP_JET )
        cv2.imshow('blank_color', cv2.resize(blank_color, (640, 360)))

        cv2.addWeighted( blank_color, alpha, count_pixel_zero_normalize_bgr, 1 - alpha, 0, count_pixel_zero_normalize_bgr )
        cv2.imshow('add_icount_pixel_zero_normalize_bgr', cv2.resize(count_pixel_zero_normalize_bgr, (640, 360)))

        cv2.addWeighted( blank_color, alpha, img1, 1 - alpha, 0, img1 )
        cv2.imshow('add_img1', cv2.resize(img1, (640, 360)))
        img1[img0_gray <= thresh_val] = img1_copy[img0_gray <= thresh_val]

        cv2.imwrite( os.path.join( dst0, '%04d.jpg' % i ),  count_pixel_zero_normalize_bgr )
        cv2.imwrite( os.path.join( dst1, '%04d.jpg' % i ), img1 )

        # cv2.imshow( 'count_pixel_zero_normalize_bgr', cv2.resize( count_pixel_zero_normalize_bgr, ( 640, 360 ) ) )
        cv2.imshow( 'img1', cv2.resize( img1, ( 640, 360 ) ) )
        cv2.waitKey( 1 )


if __name__ == '__main__':
    heatmap( r'D:/deal_pics/2018-01-24_bin' ,#二值化图像
             r'D:/deal_pics/2018-01-09_SRC',#有人的图像
             r'D:/deal_pics/2018-01-25_a',#要保存的图像1
             r'D:/deal_pics/2018-01-25_b')#要保存的图像2

'''



'''
import os
import time
import cv2
import numpy as np


def main():
    img_src_dirpath = r'D:/deal_pics/2018-01-25_SRC' + os.sep
    dir = r'D:/deal_pics/' + time.strftime('%Y-%m-%d')+'_DST' + os.sep
    if not os.path.exists(dir):
        os.makedirs(dir)
    img_dst_dirpath = dir
    img0 = None
    detector = None
    kpt0, des0 = None, None
    matcher = None
    for f in os.listdir( img_src_dirpath ):
        if f.endswith( '.JPG' ):
            if detector is None:
                img0 = cv2.imread( img_src_dirpath + f )
                detector = cv2.xfeatures2d.SIFT_create()
                kpt0, des0 = detector.detectAndCompute( img0, None )
                FLANN_INDEX_KDTREE = 1
                index_params = dict( algorithm = FLANN_INDEX_KDTREE, trees = 5 )
                search_params = dict( checks = 50 )
                matcher = cv2.FlannBasedMatcher( index_params, search_params )
            else:
                img1 = cv2.imread( img_src_dirpath + f )
                kpt1, des1 = detector.detectAndCompute( img1, None )
                matches = matcher.knnMatch( des0, des1, k = 2 )
                matchesMask = [[0, 0] for i in range( len( matches ) )]
                good = []
                for i, (m, n) in enumerate( matches ):
                    if m.distance < 0.7 * n.distance:
                        matchesMask[i] = [1, 0]
                        good.append( m )
                if len( good ) > 10:
                    src_pts = np.float32( [kpt0[m.queryIdx].pt for m in good] ).reshape( -1, 1, 2 )
                    dst_pts = np.float32( [kpt1[m.trainIdx].pt for m in good] ).reshape( -1, 1, 2 )
                    M, mask = cv2.findHomography( dst_pts, src_pts, cv2.RANSAC, 5.0 )
                    h, w, _ = img0.shape
                    img3 = cv2.warpPerspective( img1, M, (w, h) )
                    cv2.imwrite( img_dst_dirpath + f, img3 )
                else:
                    print( 'Not enough good matches. length of good matches: %d' % len( good ) )


if __name__ == '__main__':
    main()

'''


# import cv2
# import numpy as np
#
# img = cv2.imread( r'E:/image/S0392_0030_Capture_20180125113010608.JPG' )
# gray = cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
# edges = cv2.Canny( gray,50,250,apertureSize = 3  )
# lines = cv2.HoughLines( edges,1,np.pi/180,255 )
# for line in lines:
#     for rho,theta in line:
#         a = np.cos( theta )
#         b = np.sin( theta )
#         x0 = a*rho
#         y0 = b*rho
#         x1 = int( x0 + 1000*( -b ) )
#         y1 = int( y0 + 1000*( a ) )
#         x2 = int( x0 - 1000*( -b ) )
#         y2 = int( y0 - 1000*( a ) )
#         cv2.line( img,( x1,y1 ),( x2,y2 ),( 0,0,255 ),2 )
# cv2.imwrite( r'E:/image/20180130_30.jpg',img )
#
#
# cv2.imshow('1',img)
# cv2.waitKey(0)



# import cv2
# import numpy as np
#
# img = cv2.imread( 'E:/image/S0392_0030_Capture_20180125113010608.JPG' )
# gray = cv2.cvtColor( img,cv2.COLOR_BGR2GRAY )
# edges = cv2.Canny( gray,70,180,apertureSize = 3 )
# minLineLength = 10
# maxLineGap = 400
# lines = cv2.HoughLinesP( edges,1,np.pi/180,255,minLineLength=minLineLength,maxLineGap=maxLineGap )
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.line( img,( x1,y1 ),( x2,y2 ),( 0,255,0 ),2 )
# cv2.imwrite( 'E:/image/20180130_30p.jpg',img )
#
# cv2.imshow( '1',img )
# cv2.waitKey(0)

'''
import cv2
import numpy as np

img1 = cv2.imread( r'E:/cola1.jpg' )
img2 = cv2.imread( r'E:/cola2.jpg' )
diff = cv2.absdiff( img1, img2 )
operator = np.ones( ( 3, 3 ), np.uint8 )
img_erode = cv2.erode( diff, operator, iterations=1 )
img_dilate = cv2.dilate( img_erode, operator, iterations=3 )
gray = cv2.cvtColor(img_dilate,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_,contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_dilate, contours, -1, (0, 0, 255), 3)
cv2.imshow( "img",img_dilate )
cv2.waitKey( 0 )
cv2.destroyAllWindows()
'''

'''
import cv2
import os

def get_mask( video_homo_src0, video_bg_subtract_src0,video_dst0,image_dst0 ):
    cap0 = cv2.VideoCapture( video_homo_src0 )
    height0 = int( cap0.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    width0 = int( cap0.get( cv2.CAP_PROP_FRAME_WIDTH ) )
    fps0 = cap0.get( cv2.CAP_PROP_FPS )
    total0 = cap0.get( cv2.CAP_PROP_FRAME_COUNT )
    cap1 = cv2.VideoCapture( video_bg_subtract_src0 )
    height1 = int( cap1.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
    width1 = int( cap1.get(cv2.CAP_PROP_FRAME_WIDTH ) )
    fps1 = cap1.get( cv2.CAP_PROP_FPS )
    total1 = cap1.get( cv2.CAP_PROP_FRAME_COUNT )
    fourcc = cv2.VideoWriter_fourcc( *'DIVX' )
    if fps0==fps1 and height0==height1 and width0==width1 and total0==total1:
        vwriter = cv2.VideoWriter( video_dst0, fourcc, fps0, ( width0, height0 ) )
        if not os.path.exists( image_dst0 ):
            os.makedirs( image_dst0 )

        cap00 = cv2.VideoCapture()
        cap00.open(video_homo_src0)
        cap01 = cv2.VideoCapture()
        cap01.open(video_bg_subtract_src0)
        c = 0
        while 1:
            ret, frame00 = cap00.read()
            ret, frame01 = cap01.read()
            if ret:
                frame00 = cv2.cvtColor( frame00, cv2.COLOR_BGR2GRAY )
                frame01 = cv2.cvtColor( frame01, cv2.COLOR_BGR2GRAY )
                frame_diff = cv2.absdiff( frame00, frame01 )
                ret, bin_image = cv2.threshold( frame_diff, 100, 255, cv2.THRESH_BINARY )
                bin_image = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2BGR)
                vwriter.write( bin_image )
                cv2.imwrite(os.path.join( image_dst0, '%04d.jpg' % c), bin_image)
                c += 1
            else:
                break
        cap0.release()
        cap1.release()
        vwriter.release()
    else:
        # heatmap_logger.info( str( video_homo_src0 )+' and ' + str( video_bg_subtract_src0 ) +' size mismatch!' )
        pass

get_mask( r'E:/video/S0400_0043_homo.mp4',r'E:/video/S0400_0043_bg.mp4',r'E:/video/S0400_0043_mask.mp4',r'D:/deal_pics/2018-02-09' )

'''

'''
import cv2
import numpy as np

video_arr = ['E:/video/b1.mp4', 'E:/video/b2.mp4',
             'E:/video/b3.mp4']
video_dst = 'E:/ttt.mp4'

width, height = 480, 270

fps = 30.0
fourcc = cv2.VideoWriter_fourcc( *'DIVX' )
video_writer = cv2.VideoWriter( video_dst, fourcc, fps, (width*3, height) )

cap0 = cv2.VideoCapture()
cap0.open( video_arr[0] )
cap1 = cv2.VideoCapture()
cap1.open( video_arr[1] )
cap2 = cv2.VideoCapture()
cap2.open( video_arr[2] )

waitSec = 1
while 1:
    frame = np.zeros( (height, width * 3, 3), dtype = np.uint8 )
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    ret, frame2 = cap2.read()
    if ret:
        frame0 = cv2.resize( frame0, (width, height) )
        frame1 = cv2.resize( frame1, (width, height) )
        frame2 = cv2.resize( frame2, (width, height) )

        frame[:,0:width,:] = frame0
        frame[:,width:width*2, :] = frame1
        frame[:,width*2:width*3,:] = frame2

        video_writer.write( frame )
        cv2.imshow( 'frame', frame )

        key = cv2.waitKey( waitSec ) & 0xff
        if key == 27:
            break
        elif key == 32:
            waitSec = 1 - waitSec
    else:
        break

cap0.release()
cap1.release()
cap2.release()
video_writer.release()
'''




'''
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy import optimize as opt

kion = pd.read_csv(r'D:/a.csv')
kion.head()

Px = np.arange(0, len(kion), 1)
Py = kion['temp']
plt.plot(Px, Py)
plt.show()


def fit_func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

res = opt.curve_fit(fit_func, Px, Py)
a = res[0][0]
b = res[0][1]


def fit_func(x, a, b):
    return a * x+ b

res = opt.curve_fit(fit_func, Px, Py)
a = res[0][0]
b = res[0][1]

print("a = %s" % (a))
print("b = %s" % (b))

Px2 = []
for x in Px:
    Px2.append(a * x + b )

plt.plot(Px, Py)
plt.plot(Px, np.array(Px2))
plt.show()
'''

'''
# 拟合曲线
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize as opt

kion = pd.read_csv(r'D:/a.csv')
kion.head()

Px = np.arange(0, len(kion), 1)
Py = kion['temp']
plt.plot(Px, Py)
# plt.show()


def fit_func(x, a, b, c, d):
    # return a * x**3 + b * x**2 + c * x + d
    return a * x + b

res = opt.curve_fit(fit_func, Px, Py)

a = res[0][0]
b = res[0][1]
c = res[0][2]
d = res[0][3]
# print("a = %s" % a)
# print("b = %s" % b)
# print("c = %s" % c)
# print("d = %s" % d)

Px2 = []
for x in Px:
    # Px2.append(a * x**3 + b * x**2 + c * x + d)
    Px2.append(a * x + b )

plt.plot(Px, Py)
plt.plot(Px, np.array(Px2))
plt.show()

'''




'''

import pandas as pd
import csv
#计算相关系数
# rs = pd.DataFrame.from_csv(r'D:/ttt.csv',encoding='utf-8')
rs = pd.read_csv(r'D:/Clustering_TOP.csv',encoding='utf-8')

with open('D:/Clustering_TOP.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
csv_title = rows[0]
csv_title = csv_title[1:]
len_csv_title = len(csv_title)
for i in range(len_csv_title):
    for j in range(i+1):
        print(str(csv_title[j])+'_'+str(csv_title[i]) + " = " + str(rs[csv_title[i]].corr(rs[csv_title[j]])), end='\t')
    print()
'''


#azure表情分析
'''
import requests
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
from io import BytesIO

subscription_key = "de4384c8a8254f399ad8854384d36e91"
assert subscription_key
face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
image_url = 'https://how-old.net/Images/faces2/main007.jpg'
headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,' +
    'emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
}
response = requests.post(
    face_api_url, params=params, headers=headers, json={"url": image_url})
faces = response.json()
print(faces)
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
plt.figure(figsize=(8, 8))
ax = plt.imshow(image, alpha=0.6)
for face in faces:
    fr = face["faceRectangle"]
    fa = face["faceAttributes"]
    origin = (fr["left"], fr["top"])
    p = patches.Rectangle(
        origin, fr["width"], fr["height"], fill=False, linewidth=2, color='b')
    ax.axes.add_patch(p)
    plt.text(origin[0], origin[1], "%s, %d"%(fa["gender"].capitalize(), fa["age"]),
             fontsize=20, weight="bold", va="bottom")
plt.axis("off")
plt.savefig('D:/test.jpg')
plt.show()
'''


'''
import json
from watson_developer_cloud import PersonalityInsightsV2


personality_insights = PersonalityInsightsV2(
    username='9898ee12-f4f6-4f3e-a008-00b262cc6ca5',
    password='dwiNBc2EKddX')

with open(r'D:/1.txt',encoding='utf-8') as personality_text:
    print(json.dumps(personality_insights.profile(text=personality_text.read()), indent=2))

'''

'''
import json
from watson_developer_cloud import PersonalityInsightsV3

personality_insights = PersonalityInsightsV3(
    version='2016-10-20',
    username='9898ee12-f4f6-4f3e-a008-00b262cc6ca5',
    password='dwiNBc2EKddX')

with open(r'D:/v3.json',encoding='utf-8') as profile_json:
    profile = personality_insights.profile(
        profile_json.read(), content_type='application/json',raw_scores=True, consumption_preferences=True)
    print(json.dumps(profile, indent=2))

'''



'''
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os

print('Credendtials from environ: {}'.format(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))
# Instantiates a client
client = language.LanguageServiceClient()
# The text to analyze
text = u'Hello, world!'
document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)
# Detects the sentiment of the text
sentiment = client.analyze_sentiment(document=document).document_sentiment
print('Text: {}'.format(text))
print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

'''




'''
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os

def print_result(sentiment):
    score = sentiment.document_sentiment.score
    magnitude = sentiment.document_sentiment.magnitude
    for index, sentence in enumerate(sentiment.sentences):
        every_score = sentence.sentiment.score
        every_magnitude = sentence.sentiment.magnitude
        print(index, sentence.text.content,every_score,every_magnitude)

client = language.LanguageServiceClient()
text = u'Half the people on our streets look as though life was a sorry business. It is hard to find a happy looking man or woman.'
document = types.Document(
    content=text,
    type=enums.Document.Type.PLAIN_TEXT)
sentiment = client.analyze_sentiment(document=document)
print_result(sentiment)
'''

'''
def move(n, a, b, c):
    if n==1:
        print(a,'-->',c)
        return
    else:
        move(n-1,a,c,b)  #首先需要把 (N-1) 个圆盘移动到 b
        move(1,a,b,c)    #将a的最后一个圆盘移动到c
        move(n-1,b,a,c)  #再将b的(N-1)个圆盘移动到c
move(4, 'A', 'B', 'C')
'''


'''
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(100,2)
estimator=KMeans(n_clusters=3)
res=estimator.fit_predict(data)
lable_pred=estimator.labels_
centroids=estimator.cluster_centers_
inertia=estimator.inertia_
#print res
print(lable_pred)
print(centroids)
print(inertia)

for i in range(len(data)):
    if int(lable_pred[i])==0:
        plt.scatter(data[i][0],data[i][1],color='red')
    if int(lable_pred[i])==1:
        plt.scatter(data[i][0],data[i][1],color='black')
    if int(lable_pred[i])==2:
        plt.scatter(data[i][0],data[i][1],color='blue')
plt.show()
'''

'''
#每4位删除一个元素
myList = [i for i in range(1,101)]
lenMyLisy = len(myList)
while len(myList)>3:
    for i in range(lenMyLisy):
        if i>=len(myList):
                break
        if (i+1)%4 == 0:
            del myList[i]
            print(myList)

'''

'''
import numpy as np
import cv2
#车辆行动轨迹
cap = cv2.VideoCapture(r'E:\OpenCV\opencv\sources\samples\data\slow_traffic_small.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()

'''


'''
#人脸关键点检测
import numpy as np
import dlib
import cv2


video_capture = cv2.VideoCapture("huge.mp4")
face_cascade = cv2.CascadeClassifier(r'E:/OpenCV2.4.9/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def resize(image, width=1200):
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

# image_file ='images/face.jpg'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    ret, frame = video_capture.read()
    ret2, frame2 = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(50, 50)
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        (x, y, w, h) = rect_to_bb(rect)
        for (x, y) in shape:
                cv2.circle(frame2, (x, y), 2, (0, 0, 255), 1)

    cv2.imshow("Output", frame2)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
'''



"""
Usage:
  # From tensorflow/models/
  # Create train data:
python generate_TFR.py --csv_input=data/train.csv  --output_path=data/train.record
  # Create test data:
python generate_TFR.py --csv_input=data/test.csv  --output_path=data/test.record
  需要修改三处
  os.chdir('D:\\python3\\models-master\\research\\object_detection\\')
  path = os.path.join(os.getcwd(), 'images/train')
  def class_text_to_int(row_label): #对应的标签返回一个整数，后面会有文件用到
    if row_label == 'ZJL':
        return 1
    elif row_label == 'CYX':
        return 2
    else:
        None
"""

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw

#the path you want to save your results for coco to voc
savepath="E:/datasets/COCO/result/"
# img_dir=savepath+'images/'
img_dir=savepath+'images/'
anno_dir=savepath+'Annotations/'
# datasets_list=['train2014', 'val2014']
datasets_list=['train2017']

classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
#Store annotations and train2014/val2014/... in this folder
dataDir= 'E:/datasets/COCO/'
imgDir='resule/images'

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

#if the dir is not exists,make it,else delete it
def mkr(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
mkr(img_dir)
mkr(anno_dir)
def id2name(coco):
    classes=dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']]=cls['name']
    return classes

def write_xml(anno_path,head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr%(obj[0],obj[1],obj[2],obj[3],obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco,dataset,filename,objs):
    #eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path=anno_dir+filename[:-3]+'xml'
    img_path=dataDir+dataset+'/'+filename
    print(img_path)
    dst_imgpath=img_dir+filename

    img=cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)

    head=headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path,head, objs, tail)


def showimg(coco,dataset,img,classes,cls_id,show=True):
    global dataDir
    I=Image.open('%s/%s/%s'%(dataDir,dataset,img['file_name']))
    #通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name=classes[ann['category_id']]
        if class_name in classes_names:
            print(class_name)
            if 'bbox' in ann:
                bbox=ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs

for dataset in datasets_list:
    #./COCO/annotations/instances_train2014.json
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataset)

    #COCO API for initializing annotated data
    coco = COCO(annFile)
    '''
    COCO 对象创建完毕后会输出如下信息:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
    '''
    #show all classes in coco
    classes = id2name(coco)
    print(classes)
    #[1, 2, 3, 4, 6, 8]
    classes_ids = coco.getCatIds(catNms=classes_names)
    print(classes_ids)
    for cls in classes_names:
        #Get ID number of this class
        cls_id=coco.getCatIds(catNms=[cls])
        img_ids=coco.getImgIds(catIds=cls_id)
        print(cls,len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs=showimg(coco, 'result/train2017', img, classes,classes_ids,show=False)
            print(objs)
            save_annotations_and_imgs(coco, 'result/train2017', filename, objs)
