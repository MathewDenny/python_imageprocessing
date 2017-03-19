# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:22:22 2017

@author: Denny
"""
from scipy import signal
import scipy as sp
import numpy as np
import sys
import scipy.ndimage.filters as fi

#import scipy as sp
import matplotlib.pyplot as plt
import cv2 # load opencv

def generate_1d_gaussiankernel( sigma):
    filter_length = int((4 * sigma)) + 1
    result = np.zeros( filter_length )
    mid = filter_length/2
    result[mid] = 1
    return fi.gaussian_filter1d(result, sigma)

def process_1d_gaussiankernel(guassian, sigma):
    filter_length = int((4 * sigma)) + 1
    for j in range(int(filter_length/2)):
        guassian[j] = guassian[j] * -1
    return guassian

def detect_local_maxima(image,window_size):
    if (image[0,(window_size * window_size/2)] == image.max()
        and image[0,(window_size * window_size/2)] > 0):
        
        return 1
    else:
        return 0
    
def plot_harris_points(image, filtered_coords):
   
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
    plt.axis('off')
    plt.show()
    
def get_harris_points(harrisim, min_distance=10, threshold=0.5):
    """ return corners from a Harris response image
        min_distance is the minimum nbr of pixels separating 
        corners and image boundary"""

    #find top corner candidates above a threshold
    corner_threshold = max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    #get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = [ (candidates[0][c],candidates[1][c]) for c in range(len(candidates[0]))]
    #...and their values
    candidate_values = [harrisim[c[0]][c[1]] for c in coords]
    
    #sort candidates
    index = np.argsort(candidate_values)
    
    #store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,min_distance:-min_distance] = 1
    
    #select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i][0]][coords[i][1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i][0]-min_distance):(coords[i][0]+min_distance),
                (coords[i][1]-min_distance):(coords[i][1]+min_distance)] = 0
                
    return filtered_coords

# To Call the conv2d function
def find_image_gradient( gray_image):
    height, width = gray_image.shape
    sigma = 5
#    sp.ndimage.filters.gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    guassian = generate_1d_gaussiankernel(sigma)
    guassian = process_1d_gaussiankernel(guassian, sigma)
    
    imx = np.zeros(gray_image.shape)
    fi.gaussian_filter(gray_image,(sigma),(0,1), imx)
    
    imy = np.zeros(gray_image.shape)
    fi.gaussian_filter(gray_image,(sigma),(1,0), imy)
   
#    imx = sp.ndimage.convolve1d(gray_image, guassian, 1)
#    imy = sp.ndimage.convolve1d(gray_image, guassian, 0)   

#    print imx, imy
    Ix2 = np.multiply(imx , imx)
    Iy2 = np.multiply(imy , imy)
    IxIy = np.multiply(imx , imy)
    filter_length = int((6 * sigma)) + 1
    print ((imx)) , Ix2
    
    result = np.zeros( filter_length )
    mid = filter_length/2
    result[mid] = 1
    smoothening_filter = fi.gaussian_filter1d(result, sigma)
    
    smoothening_filter = np.outer(smoothening_filter , smoothening_filter.transpose())


#    Ix2 = sp.signal.convolve2d(Ix2, smoothening_filter, 'same', 'symm', 0)
#    Iy2 = sp.signal.convolve2d(Iy2, smoothening_filter, 'same', 'symm', 0)
#    IxIy = sp.signal.convolve2d(IxIy, smoothening_filter, 'same', 'symm', 0)
#    

#    fi.gaussian_filter(Ix2trial,(2 * sigma),0, Ix2, 'reflect',0, filter_length)
#    fi.gaussian_filter(Iy2trial,(2 * sigma),0, Iy2, 'reflect',0, filter_length)
#    fi.gaussian_filter(IxIytrial,(2 * sigma),0, IxIy, 'reflect',0, filter_length)
    
    fi.gaussian_filter(Ix2,(2 * sigma),0, Ix2, 'reflect',0, filter_length)
    fi.gaussian_filter(Iy2,(2 * sigma),0, Iy2, 'reflect',0, filter_length)
    fi.gaussian_filter(IxIy,(2 * sigma),0, IxIy, 'reflect',0, filter_length)
    
   
    Idet =  ((np.multiply(Ix2 , Iy2) - np.multiply(IxIy, IxIy)))    
    Itrace =  Ix2 + Iy2
    H = Idet  - (Itrace**2)* 0.06
    
    
#    offset = 3/2
#    cornerList = []
#    height = gray_image.shape[0]
#    width = gray_image.shape[1]
#    #Loop through image and find our corners
#    print "Finding Corners..."
#    for y in range(offset, height-offset):
#        for x in range(offset, width-offset):
#            #Calculate sum of squares
#            windowIxx = Ix2[y-offset:y+offset+1, x-offset:x+offset+1]
#            windowIxy = IxIy[y-offset:y+offset+1, x-offset:x+offset+1]
#            windowIyy = Iy2[y-offset:y+offset+1, x-offset:x+offset+1]
#            Sxx = windowIxx.sum()
#            Sxy = windowIxy.sum()
#            Syy = windowIyy.sum()
#
#            #Find determinant and trace, use to get corner response
#            det = (Sxx * Syy) - (Sxy**2)
#            trace = Sxx + Syy
#            r = det - 0.06*(trace**2)
##            print r
#
#            #If corner response is over threshold, color the point and add to corner list
#            if r > 150:
##               print x, y, r
#                cornerList.append([x, y, r])
#                
#   
#
#
#
#    plt.figure(figsize=(10,20))
#    plt.subplot(211),plt.imshow(Idet, cmap = 'gray')
#    plt.title('Ix'), plt.xticks([]), plt.yticks([])
#    
#    plt.subplot(212),plt.imshow(Ix2, cmap = 'gray')
#    plt.title('Iy'), plt.xticks([]), plt.yticks([])
    
    
#    print H

    plt.figure(figsize=(10,20))
    plt.subplot(211),plt.imshow(imx, cmap = 'gray')
    plt.title('Ix'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(212),plt.imshow(H, cmap = 'gray')
    plt.title('Iy'), plt.xticks([]), plt.yticks([])

    maxima_found = 0
    window_size = 3
    MAX = 800
    r = H.shape[0]
    c = H.shape[1]
    # Create an empty list to hold our points of interest.
    c_points = []
    exit = 0
    for i in range(r):        
        if (exit == 1):
            break
        for j in range(c):
            if ((i-(window_size/2)) <= 0 or 
                (i+(window_size/2))>=(r-1) or 
                (j-(window_size/2)) <=0 or 
                (j+(window_size/2)) >=(c-1)):
                continue
            window = H[(i-(window_size/2)):(i+(window_size/2) + 1),
                       (j-(window_size/2)):(j+(window_size/2) + 1)]
#            print window
#            print "is", i-(window_size/2), (j-(window_size/2))
            window_to_image_vector = window.reshape( 1, (window_size * window_size) )
          
            is_local_max = detect_local_maxima(window_to_image_vector, window_size)
            if (is_local_max == 1):
                c_points.append((i,j))
                maxima_found = maxima_found + 1 
#                print "Selected",  window, j, i
                if (maxima_found > MAX):
                    print maxima_found
                    exit = 1
                    break
    print c_points
    
    plot_harris_points(gray_image, c_points)
    
#    for i,elem in enumerate(cornerList):  
#        cv2.circle(gray_image,(cornerList[i][0],cornerList[i][1]), 1, (0,128,0), -1)
#    
#    cv2.imshow('image',gray_image)
#    cv2.waitKey(0)
#    Ix2 = sp.ndimage.convolve1d(Ix2, smoothening_filter, 1)
#    Ix2 = fi.gaussian_filter(Ix2, 2*sigma,0)
#    Iy2 = fi.gaussian_filter(Iy2, 2*sigma,0)
#    IxIy = fi.gaussian_filter(IxIy, 2*sigma,0)
#    
##    gaussian = np.array([-1, 0, 1])
##    input = np.array([[1, 2, 3 , 5, 8], 
##                      [3, 4, 5 , 6, 7], 
##                      [3, 4, 5 , 6, 7]])
#    Ix = sp.ndimage.convolve1d(gray_image, guassian, 1)
#    Iy = sp.ndimage.convolve1d(gray_image, guassian, 0)
#    plt.figure(figsize=(10,20))
#    plt.subplot(211),plt.imshow(imx, cmap = 'gray')
#    plt.title('Ix'), plt.xticks([]), plt.yticks([])
#    
#    plt.subplot(212),plt.imshow(imy, cmap = 'gray')
#    plt.title('Iy'), plt.xticks([]), plt.yticks([])
    
#    print guassian, Ix, Iy
    


def main():

    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread(file)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)

    (image_rows, image_columns, image_channels) = img.shape 
    print "channels = ", image_channels;
    if (image_channels > 1):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = img
    #filter details
    find_image_gradient(gray_image)
    
if __name__== "__main__":
    main()
