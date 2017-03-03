# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:15:15 2017

@author: denny
"""

from conv_fft import conv2d

import cv2 # load opencv
import numpy as np
import sys
import matplotlib.pyplot as plt

import scipy.stats as st


def gkern(kernlen=5, nsig=0.1):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

## 1D Convolution Function definition is here
def add_gaussian_noise(input_image):
    """ Adds gauusian noise with zero mean """
    noise_image = np.random.normal(0.0,30,(input_image.shape))
    rows, columns = input_image.shape
    #noise_image = input_image + noise_image
    norm_noise = np.zeros((rows ,columns ),dtype=np.uint8)
    
    #cv2.normalize(noise_image, norm_noise, 0, 100, cv2.NORM_MINMAX, dtype=cv2.CV_8U )
    print "noise_normalised -" ,norm_noise
    print "input_image -" ,input_image
    
    noise_image = ((input_image + noise_image))
    cv2.normalize(noise_image, norm_noise, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U )
    print "final noise_image added-" ,norm_noise
    return norm_noise

def main():
   
    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread(file)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_image = add_gaussian_noise(gray_image)
#    print "The noisy image is ", noise_image
#    print "The original image is ", gray_image
    #plt.imshow(gray_image)
    
    # creating a gaussian kernel
    gkernel = gkern(5,0.1)
    d = 1
    print "Gaussian Kernel is", gkernel
    grows, gcolumns = gkernel.shape
    print "grows =", grows, " gcols =" , gcolumns
     #filter details
    m = 1
    n = 1
    while True:
        m = int(input('number of rows, m = '))
        n = int(input('number of columns, n = '))
        if (m > 0 and n > 0):
            break;
        
    while True:
        d = int(input('Enter filter division coefficient, d = '))
        if (d != 0):
            break;
    filter = np.zeros((m, n),dtype=np.int8)
    for i in range(m):
        for j in range(n):#           
            #int(input('Enter filter ',i,j))
            filter[i,j] = 1;
                  #int(input('Enter filter[' + str(i) +', ' + str(j) + ']  = '))            


    norm_image = conv2d(noise_image, filter, d) 
    #norm_image = conv2d(noise_image, gkernel, d) 
    plt.figure(figsize=(10,30))
    plt.subplot(311),plt.imshow(gray_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(312),plt.imshow(noise_image, cmap = 'gray')
    plt.title('Noise Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(313),plt.imshow(norm_image, cmap = 'gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    print "The noise image is ", noise_image
    print "The output image is ", noise_image
    print "The input image is ", norm_image

    
    
if __name__== "__main__":
    main()