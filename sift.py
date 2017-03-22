# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:22:22 2017

@author: Denny
"""
from scipy import signal
from corner_detect import gauss_derivatives
import scipy as sp
import numpy as np
import sys
import scipy.ndimage.filters as fi
from numpy import linalg as LA
import math as math
#import scipy as sp
import matplotlib.pyplot as plt
from corner_detect import find_image_gradient

import cv2 # load opencv


def rotate_image(image):
    angle = 15.0
    image_center = tuple(np.array(image.shape)/2)
    rotated_image = np.zeros(((image.shape[1]),(image.shape[0])))
  
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    rotated_image = cv2.warpAffine(image, rot_mat, 
                                   ((image.shape[1]),(image.shape[0])),flags=cv2.INTER_LINEAR)
    return rotated_image

def plot_matching_points(img1, img2, match_list1, match_list2):
   
    # #####################################
    # visualization of the matches
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    
    view = sp.ones((max(h1, h2), w1 + w2), sp.uint8)
    view[:h1, :w1] = img1  
    view[:h2, w1:w1 + w2 + 1] = img2
    
    print "lenght = ", len(match_list1)
    for m in range(len(match_list1)):
        # draw the keypoints
        # print m.queryIdx, m.trainIdx, m.distance
#        new_list = [x[1]+1 for x in match_list2]
        color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
        pt_a = (int(match_list1[m][1]), int(match_list1[m][0]))
        pt_b = (int(match_list2[m][1] + w1), int(match_list2[m][0]))
        print "ploting = ", pt_a, pt_b
        
        cv2.line(view, pt_a , pt_b, color,1)         
        cv2.circle(view,pt_a, 5, (0,128,0), 1)
        cv2.circle(view,pt_b, 5, (255,128,0), 1)
        
    cv2.imshow("dst_rt", view)
    cv2.waitKey()
    cv2.imwrite('sift_matching.png',view)
    
#patch values varies from 0 to 7;size is n*n
def compute_hog(patch, n):
    hist_points = np.zeros(8)
    for i in range(n*n): 
#        print "patch[0,i] is " , patch[0,i]    
        hist_points[int(patch[0,i])] = hist_points[int(patch[0,i])] + 1
#    print "HoG", hist_points
    return hist_points

def histogram_shift(hist_points, dominant_orientation_bin):    
    ctr = dominant_orientation_bin
    new_hist_points = np.zeros(8)
    for i in range(8): 
        new_hist_points[i] = hist_points[ctr]
        ctr = (ctr + 1) % 8
#    print "Dominant HoG", new_hist_points
    return new_hist_points

#hist points is 8 size array
def compute_dominant_and_shift(hist_points ):
    dominant_orientation_bin = np.argmax(hist_points)
    new_hist_points = np.zeros(8)
    ctr = dominant_orientation_bin
    for i in range(8): 
        new_hist_points[i] = hist_points[ctr]
        ctr = (ctr + 1) % 8
#    print "Dominant HoG", new_hist_points
    return dominant_orientation_bin, new_hist_points

#hist points is list of 16 arrays ,n is isze of each 1d array
def concatenate_normalize_hog(hist_points_arr, n):
    
#    print "length of histogram array = ", len(hist_points_arr), n   
    concatenated = np.array(8*len(hist_points_arr))
    ctr = 0
    for i in range(0,n*len(hist_points_arr) , n):         
        concatenated = np.insert((concatenated),i,hist_points_arr[ctr])
        ctr = ctr + 1

    normalized_hist = LA.norm((concatenated), 2) 
    normalized_hist = concatenated / normalized_hist
    for i in range(128): 
       if (normalized_hist[i] > 0.2): 
           normalized_hist[i] = 0.2
                          
    renormalized_hist = LA.norm((normalized_hist), 2) 
    renormalized_hist = normalized_hist / renormalized_hist      
#    print "sizeof renormalized_hist ", renormalized_hist.shape   
    return renormalized_hist

# Finding matching 
def find_correspondence( sift_array1, c_points1, sift_array2, c_points2):
    
    r = 0.60
    
    match_list1 = []
    match_list2 = []
    for i,elem1 in enumerate(c_points1):
        d1 = 1000000.0;d2 = 1000000000.0 # random large value
        ctr = 0
        for j,elem2 in enumerate(c_points2):
#            print "ind " ,i,j, len(sift_array1)  , len(c_points1) , len(sift_array2)  , len(c_points2)         
            dist = np.linalg.norm(sift_array1[i]-sift_array2[j])
            if (dist < d1):
                d1 = dist
#                if (d1 == 0):
#                    ctr = ctr + 1
                potential_match = elem2
            if (dist < d2 and dist > d1):
                d2 = dist
        if (ctr > 1) :
            print "ctr   = ", ctr, i
        print "distances  " ,d1, d2, d1/d2
        if (d2 != 0 and d1 / d2 < r):
            match_list1.append((elem1))
            match_list2.append((potential_match))
#        else:
#           match_list.append((elem1, "none"))     
#            
    print "MATCH = ", match_list1, match_list2  
    
    return match_list1, match_list2
    
#assuming size by size array
def determine_gradient_and_orientation( imx, imy, size):
    row = size
    col = size
    m = np.zeros((row,col))
    bins = 8
    q = 360 / bins
    theta = np.zeros((row,col))
    x_q = np.zeros((row,col))
    
    for i in range(row):        
        for j in range(col):

            m[i,j] =  imx[i,j]**2 + imy[i,j]**2
            m[i,j] =  np.sqrt(m[i,j])
            if ((imx[i,j]) == 0):
                theta[i,j] = 90
            else:
                theta[i,j] = (imy[i,j] ) / (imx[i,j])
                theta[i,j] = math.degrees(math.atan(theta[i,j]))
            x = theta[i,j]
            x_q[i,j] = np.abs(np.floor((x+q/2)/q));
#            print x, x_q[i,j]
    return m, x_q
    
    
            
# Determin the SIFT Descriptor
def determine_gradients( gray_image,c_points):
    height, width = gray_image.shape
    sigma = 5
    filter_length = int((4 * sigma)) + 1
    #find the derivatives
   
    
    row = gray_image.shape[0]
    col = gray_image.shape[1]
   
    #Create a descriptor
    new_c_points = []
    N = 16
    NbyTwo = N/2
    NbyFour = N/4
    
    hist_points = []
    sift_result = []
    
    # for each point take a 16x16 patch and find the dominant bin in Hog
    for i,elem in enumerate(c_points):
        r = elem[0]
        c = elem[1]
        if ((r-NbyTwo) <= 0 or 
                (r+NbyTwo)>=(row-1) or 
                (c-NbyTwo) <=0 or 
                (c+NbyTwo) >=(col-1)):
                continue
        new_c_points.append((r,c))
        imagepatch = gray_image[r-NbyTwo:r+NbyTwo , c-NbyTwo:c+NbyTwo ]
        sigma = NbyTwo
        filter_length = int((4 * sigma)) + 1
        imx,imy = gauss_derivatives(imagepatch, sigma, filter_length)
        grad_mag, x_q = determine_gradient_and_orientation( imx, imy, N)
        
        grad_mag = fi.gaussian_filter(grad_mag,(np.sqrt(NbyTwo)))#,0, grad_mag, 'reflect')
        
        imx,imy = gauss_derivatives(grad_mag, sigma, filter_length)
        grad_mag, x_q = determine_gradient_and_orientation( imx, imy, N)       
        x_q16x16_vector = x_q.reshape( 1, (N * N))
        histogram = compute_hog(x_q16x16_vector, N)
        dominant_orientation_bin, histogram_full = compute_dominant_and_shift(histogram)
        
#        Calculate each 4x4 and rotate its hog to match dominant HoG
        ctr = 0
        del hist_points[:]
        #get all 4x4 patch
        for r in range(0,15 , NbyFour):        
            for c in range(0,15, NbyFour):
           
                patch4x4 = x_q[r:r+NbyFour,c:c+NbyFour]
                patch4x4_vector = patch4x4.reshape( 1, (NbyFour * NbyFour))

                histogram = compute_hog(patch4x4_vector, NbyFour)
                histogram = histogram_shift(histogram, dominant_orientation_bin)                
                hist_points.append(histogram)                
                ctr = ctr + 1
        # Append the computed sift descriptor
        sift_result.append( concatenate_normalize_hog(hist_points, NbyTwo))


    return sift_result,new_c_points
    
def main():

#    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread("corners.jpg")
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)

    (image_rows, image_columns, image_channels) = img.shape 
    print "channels = ", image_channels;
    if (image_channels > 1):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
       
        
    c_points1 = find_image_gradient(gray_image)   
    sift_descriptor_array1,corresponding_c_points1 = determine_gradients(gray_image, c_points1)
    
    rotated_image = rotate_image(gray_image)
#    rotated_image = gray_image2
    c_points2 = find_image_gradient(rotated_image)
    sift_descriptor_array2,corresponding_c_points2 = determine_gradients(rotated_image, c_points2)
    match_list1, match_list2 = find_correspondence(sift_descriptor_array1, corresponding_c_points1, 
                        sift_descriptor_array2, corresponding_c_points2  )
    plot_matching_points(gray_image, rotated_image, match_list1, match_list2)
#    plot_harris_points(gray_image, c_points)
    
if __name__== "__main__":
    main()
