# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:22:22 2017

@author: Denny
"""

import scipy as sp
import numpy as np
import sys

#import scipy as sp
import matplotlib.pyplot as plt
from sift import find_correspondence

import cv2 # load opencv

def plot_matching_points(img1, img2, match_list1, match_list2):
   
    # #####################################
    # visualization of the matches
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    
    view = sp.ones((max(h1, h2), w1 + w2), sp.uint8)
    view[:h1, :w1] = img1  
    view[:h2, w1:w1 + w2 + 1] = img2
    
#    print "lenght = ", len(match_list1)
    for m in range(len(match_list1)):
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
#        new_list = [x[1]+1 for x in match_list2]
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        pt_a = (int(match_list1[m][1]), int(match_list1[m][0]))
        pt_b = (int(match_list2[m][1] + w1), int(match_list2[m][0]))
        print "ploting = ", pt_a, pt_b
        
        cv2.line(view, pt_a , pt_b, (0,0,128),1)
        cv2.circle(view,pt_a, 5, (0,128,0), 1)
        cv2.circle(view,pt_b, 5, (255,128,0), 1)



    cv2.imshow("dst_rt", view)
    cv2.waitKey()
    

# plot sift
def plot_sift( gray_image, skp):
    for i,elem in enumerate(skp):  
        cv2.circle(gray_image,
                   (int(skp[i].pt[0]),int(skp[i].pt[1])),  
                   int(skp[i].size), 
                   (0,128,0), 
                   1)
    
    cv2.imshow('gray_image',gray_image)
    cv2.waitKey(0)
    
# SIFT compute
def determine_sift( img):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    
    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)
    return skp,sd
    
def apply_ransac(srcPoints, dstPoints):
    M, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)
    print M
    print "and then"
    print mask
    return M,mask

def stitch_images(left_image, right_image):
    h1, w1 = left_image.shape
    h2, w2 = right_image.shape
    
    
    view = sp.ones((max(h1, h2),max(w1, w2)), sp.uint8)
    view[:h1, :w1] = left_image  
    view[:h2, :w2] = right_image
    
    plt.figure(figsize=(10,20))
    plt.subplot(211),plt.imshow(left_image, cmap = 'gray')
    plt.title('left'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(212),plt.imshow(right_image, cmap = 'gray')
    plt.title('right'), plt.xticks([]), plt.yticks([])
    cv2.imwrite('stichedimage.png',view)
    
def main():

#    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread("BK_left.jpg")
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)

    (image_rows, image_columns, image_channels) = img.shape 
    print "channels = ", image_channels;
    if (image_channels > 1):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
     
        
    img = cv2.imread("BK_right.jpg")
    (image_rows, image_columns, image_channels) = img.shape 
    print "channels = ", image_channels;
    if (image_channels > 1):
        gray_image2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image2 = img
        
    # determine first image descr
    skp1,sd1 = determine_sift(gray_image)
    points1 = ([(int(p.pt[1]),int(p.pt[0] )) for p in skp1])
#    plot_sift( gray_image, skp1)
              
    # determine second image descr              
    skp2,sd2 = determine_sift(gray_image2)
    points2 = ([(int(p.pt[1]),int(p.pt[0] )) for p in skp2])
#    plot_sift( gray_image2, skp2)              

    # Finding matching 
    match_list1, match_list2 = find_correspondence( sd1, points1, sd2, points2)
#    np_points1 = np.asarray(match_list1)
#    np_points2 = np.asarray(match_list2)
    plot_matching_points(gray_image, gray_image2, match_list1, match_list2)
        
    float_points1 = ([((p[1]),(p[0])) for p in match_list1])
    float_points2 = ([((p[1]),(p[0])) for p in match_list2])
    float_points1 = np.float32(float_points1)
    float_points2 = np.float32(float_points2)     
    
    M, mask = apply_ransac(float_points1, float_points2)    
    dst = cv2.warpPerspective(gray_image,M,gray_image.shape)
    stitch_images(dst, gray_image2)
    
if __name__== "__main__":
    main()
