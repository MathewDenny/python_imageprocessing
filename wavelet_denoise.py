# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 12:15:50 2017

@author: denny
"""

from noise_removal import add_gaussian_noise

import cv2 # load opencv
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pywt
import math
import scipy.stats as st


#this is eq to Hx
def inverse_transform(x):    
    #print "inverse = ", pywt.waverec2(x, 'haar')
    return pywt.waverec2(x, 'db2')

#this is eq to Ht y    
def forward_transform(y):
    #print "forward = ", pywt.wavedec2(y, 'haar', 'symmetric',3)
    return pywt.wavedec2(y, 'db2', 'symmetric',3)
   
## Iterated soft thresholding algorithm 
"""ISTA algorithm.
    
Arguments
---------
y : array-like
    input image.
coeffs :tuple
    2D wavelet transform coefficients 
lambda : scalar
    Proximal operator.
"""
def ista(y,coeffs,lambda_val):
    ctr = 0
    
    #   x = np.zeros(0)
   
    #print "coeffss = ", coeffs
    #h = LA.inv(x) * y
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    print "cH3 -", cH3
    
#    coeff_matrix = np.zeros((3,3))
#    coeff_matrix [:,1] = (cH1, cV1, cD1)
#    coeff_matrix [:,2] = (cH2, cV2, cD2)
#    coeff_matrix [:,3] = (cH3, cV3, cD3)
    alpha_vals1 = LA.eigvals( y)
    alpha_vals2 = LA.eigvals(np.dot(cH2.transpose() , (cH2)))
    alpha_vals3 = LA.eigvals(np.dot(cH1.transpose() , (cH1)))
    
    noiseSigma = 30

    threshold = lambda_val* noiseSigma* math.sqrt(2* np.log2(y.size))
#    
#    x_new = [x *0  for x in coeffs]
#    x_old=  [x *0  for x in coeffs]
##    x_old = np.zeros((rows,1))
#    alpha1 = np.amax(alpha_vals1)
#    alpha2 = np.amax(alpha_vals2)
#    alpha3 = np.amax(alpha_vals3)
#    
#    alpha = [ alpha3, alpha2, alpha1]
#    alpha = np.amax(alpha)
#    print "alpha=", alpha
    old_error = 1
    new_error = 1
    error_ratio = 2
    
    
    ctr = 0
    alpha   = 3.0
    
#    print "scalar = ", scalar
#    x_0 = pywt.threshold(coeffs[0], scalar, 'soft')
#    x_1 = pywt.threshold(coeffs[1], scalar, 'soft')
#    x_2 = pywt.threshold(coeffs[2], scalar, 'soft')
#    x_3 = pywt.threshold(coeffs[3], scalar, 'soft')
#    x_new = [x_0,x_1,x_2,x_3];
#                
    while abs(error_ratio) > 0.01:
        
        old_error = new_error

        x_old = x_new
       
        if (ctr == 0):
            to_norm = y
            ctr = ctr + 1
        else:
            to_norm = y -  inverse_transform(x_old)
        
                                                
        data1 = [x / alpha for x in (forward_transform(to_norm)[0])]
        data2 = [x / alpha for x in (forward_transform(to_norm)[1])]
        data3 = [x / alpha for x in (forward_transform(to_norm)[2])]
        data4 = [x / alpha for x in (forward_transform(to_norm)[3])]
        data = [data1, data2, data3, data4];
        print "forward_transform(to_norm) =  " ,forward_transform(to_norm)
        print "coeffs  =  " ,coeffs
        print "data  =  " ,data
        data = data + x_old
        #print "data =  " ,data,  ;
        #scalar   = lambda_val / 2.0 * alpha ;
        x_0 = pywt.threshold(data[0], threshold, 'soft')
        x_1 = pywt.threshold(data[1], threshold, 'soft')
        x_2 = pywt.threshold(data[2], threshold, 'soft')
        x_3 = pywt.threshold(data[3], threshold, 'soft')
        x_new = [x_0,x_1,x_2,x_3];

        new_error = LA.norm((y -  inverse_transform(x_new)), 2)
        error_ratio = (old_error - new_error)/ old_error
        ctr = ctr + 1   
        print "x_1 = " , data[1], x_1
    return x_new


# Iterated soft thresholding algorithm
    
# Iterated soft thresholding algorithm
def denoise_image(noise_image):
    
    
    coeffs = pywt.wavedec2(noise_image, 'db2','symmetric', 3)
    temp_coeffs = ista(noise_image,coeffs,0.1)
    new_image = pywt.waverec2(temp_coeffs, 'db2')   
    return new_image   
            
def main():
    #y = np.matrix('2 1 4 0')
    #d = np.matrix('9 11 0 4')
    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread(file)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_image = add_gaussian_noise(gray_image)
    denoised_image = denoise_image(noise_image)
#    print "The noisy image is ", noise_image
#    print "The original image is ", gray_image
    #plt.imshow(gray_image)
    

    
    plt.figure(figsize=(10,20))
    plt.subplot(211),plt.imshow(noise_image, cmap = 'gray')
    plt.title('Noise Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(212),plt.imshow(denoised_image, cmap = 'gray')
    plt.title('DeNoise Image'), plt.xticks([]), plt.yticks([])
    
       
    
if __name__== "__main__":
    main()
