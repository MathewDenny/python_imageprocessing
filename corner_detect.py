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
import math as math
#import scipy as sp
import matplotlib.pyplot as plt
import cv2 # load opencv

#def generate_1d_gaussiankernel( sigma):
#    filter_length = int((4 * sigma)) + 1
#    result = np.zeros( filter_length )
#    mid = filter_length/2
#    result[mid] = 1
#    return fi.gaussian_filter1d(result, sigma)

def generate_1d_gaussiankernel( sigma, size):
    
    x = np.mgrid[-size:size+1]
    
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    twosigmasquare = 2 * sigma**2
    onebyroottwopi = 1.0/math.sqrt((2*math.pi))
    x = np.mgrid[-size/2:size/2+1]
    
#    kernlen = size
#    interval = (2*sigma+1.)/(kernlen)   
#    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernlen+1)
#    y = np.linspace(-sigma-interval/2., sigma+interval/2., kernlen+1)
    
    
    gx = onebyroottwopi *  np.exp(-(x**2/float(twosigmasquare))) * (1.0/sigma)
    print gx, gx.sum()
    return gx
    
def generate_2d_gaussiankernel(sigma, size, sizey = None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    twosigmasquare = 2 * sigma**2
    onebytwopi = 1.0/((2*math.pi))
    
    x, y = np.mgrid[-size/2:size/2+1, -sizey/2:sizey/2+1]
    g = onebytwopi *  np.exp(-(x**2/float(twosigmasquare)) - 
                               (y**2/float(twosigmasquare))) * (1.0/sigma**2)
    return g / g.sum()

#def generate_2d_gaussiankernel(nsig, kernlen, sizey = None):
#    """Returns a 2D Gaussian kernel array."""
#
#    interval = (2*nsig+1.)/(kernlen)
#    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
#    kern1d = np.diff(sp.stats.norm.cdf(x))
#    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#    kernel = kernel_raw/kernel_raw.sum()
#    return kernel

def gauss_derivative_kernels(sigma, size, sizey=None):
    """ returns x and y derivatives of a 2D 
        gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    y, x = np.mgrid[-size:size+1, -sizey:sizey+1]
    
    #x and y derivatives of a 2D gaussian with standard dev half of size
    # (ignore scale factor)
    twosigmasquare = 2 * sigma**2
    onebyroottwopi = 1.0/math.sqrt((2*math.pi))
    x, y = np.mgrid[-size/2:size/2+1, -sizey/2:sizey/2+1]
    
#    kernlen = size
#    interval = (2*sigma+1.)/(kernlen)   
#    x = np.linspace(-sigma-interval/2., sigma+interval/2., kernlen+1)
#    y = np.linspace(-sigma-interval/2., sigma+interval/2., kernlen+1)
    
    
    gx = onebyroottwopi *  np.exp(-(x**2/float(twosigmasquare))) * (1.0/sigma**3)
    gy = onebyroottwopi *  np.exp(-(y**2/float(twosigmasquare))) * (1.0/sigma**3)
    
    gx = - x * gx# math.exp(-(x**2/float(temp)+y**2/float(temp))) 
    gy = - y * gy# math.exp(-(x**2/float((0.5*size)**2)+y**2/float((0.5*sizey)**2))) 

    return gx,gy

def gauss_derivatives(im,sigma, n, ny=None):
    """ returns x and y derivatives of an image using gaussian 
        derivative filters of size n. The optional argument 
        ny allows for a different size in the y direction."""

    gx,gy = gauss_derivative_kernels(sigma, n, sizey=ny)

    imx = sp.signal.convolve2d(im,gx, mode='same')
    imy = sp.signal.convolve2d(im,gy, mode='same')

    return imx,imy

def detect_local_maxima(image,window_size):
    threshold = 2
    for i in range(window_size*window_size):        
        if (i != (window_size * window_size/2)):
            if ((image[0,(window_size * window_size/2)] - image[0,i]) == 0) :
                return 0
    if (image[0,(window_size * window_size/2)] == image.max()
        and image.max() - image.min() > threshold
        and image[0,(window_size * window_size/2)] > 0):
        
        return 1
    else:
        return 0
    
def plot_harris_points(image, filtered_coords):
   
#    plt.figure()
#    plt.gray()
#    plt.imshow(image)
#    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
#    plt.axis('off')
#    plt.show()
    for i,elem in enumerate(filtered_coords):  
        cv2.circle(image,(filtered_coords[i][0],filtered_coords[i][1]), 10, (0,128,0), -1)
    
    cv2.imshow('image',image)
    cv2.waitKey(0)
    

# To Call the conv2d function
def find_image_gradient( gray_image):
    height, width = gray_image.shape
    sigma = 3
    filter_length = int((4 * sigma)) + 1
#    sp.ndimage.filters.gaussian_filter1d(input, sigma, axis=-1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
#    guassian = generate_1d_gaussiankernel(sigma)
   
    imx,imy = gauss_derivatives(gray_image, sigma, filter_length)
    
    Ixx = np.multiply(imx , imx)
    Iyy = np.multiply(imy , imy)
    Ixy = np.multiply(imx , imy)
    filter_length = int((6 * sigma)) + 1
                       
    #smoothening the gradients
#    gauss2d = generate_2d_gaussiankernel(2*sigma, filter_length)
    
    corners_image = cv2.goodFeaturesToTrack(gray_image, 30, 0.01, 10)
    print "gray_image corners =", corners_image
#    Ix2 = sp.signal.convolve2d(Ix2,gauss2d, mode='same')
#    Iy2 = sp.signal.convolve2d(Iy2,gauss2d, mode='same')
#    IxIy = sp.signal.convolve2d(IxIy,gauss2d, mode='same')
#    print "Test", gauss2d, imx.shape,imy.shape
    
    
#    Ix2 = sp.signal.convolve2d(Ix2, smoothening_filter, 'same', 'symm', 0)
#    Iy2 = sp.signal.convolve2d(Iy2, smoothening_filter, 'same', 'symm', 0)
#    IxIy = sp.signal.convolve2d(IxIy, smoothening_filter, 'same', 'symm', 0)
#    

    Ix2 = fi.gaussian_filter(Ixx,(2 * sigma))#,0, Ix2, 'reflect',0, filter_length)
    Iy2 = fi.gaussian_filter(Iyy,(2 * sigma))#,0, Iy2, 'reflect',0, filter_length)
    IxIy = fi.gaussian_filter(Ixy,(2 * sigma))#,0, IxIy, 'reflect',0, filter_length)
    
   
    Idet =  ((np.multiply(Ix2 , Iy2) - np.multiply(IxIy, IxIy)))    
    Itrace =  Ix2 + Iy2
    H = Idet  - (Itrace**2)* 0.06
   
    plt.figure(figsize=(10,20))
    plt.subplot(211),plt.imshow(Ix2, cmap = 'gray')
    plt.title('Ix'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(212),plt.imshow(Iy2, cmap = 'gray')
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
    plt.figure()
    plt.gray()
    plt.imshow(gray_image)
    points = ([(int(p[0,1]),int(p[0,0])) for p in corners_image]);#,[int(p[0,1]) for p in corners_image],'*')
    plt.plot([int(p[0,0]) for p in corners_image],[int(p[0,1]) for p in corners_image],'*')
    plt.axis('off')
    plt.show()
    print points
    return points
    
   
    
def test_rotate_and_scale(image):
    angle = 45
    image_center = tuple(np.array(image.shape)/2)
    
#    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
#    rotated_image = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
#    c_points = find_image_gradient(rotated_image)
#    plot_harris_points(rotated_image, c_points)
    
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,0.5)
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    c_points = find_image_gradient(rotated_image)
    plot_harris_points(rotated_image, c_points)
    


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
        
#    test_rotate_and_scale(gray_image)
#    filter details
    c_points = find_image_gradient(gray_image)
    plot_harris_points(gray_image, c_points)
    
if __name__== "__main__":
    main()
