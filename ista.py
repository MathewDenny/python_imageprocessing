# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:52:05 2017

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

## Iterated soft thresholding algorithm
"""ISTA algorithm.
    
Arguments
---------
y : array-like
    input signal vector.
H : 2d array
    Dictionary Matrix.
lambda : scalar
    Proximal operator.
"""
def ista(y,H,lambda_val):
    ctr = 0
   
    #   x = np.zeros(0)
    rows,columns = H.shape
    #print "rows cols = ", rows, columns
    
    x_new = np.zeros((rows,1))
    x_old = np.zeros((rows,1))
    
    alpha_vals = LA.eigvals(np.dot(H.transpose() , H))
    
   
    alpha = np.amax(alpha_vals)
    
    #print "alpha = " , alpha;
    old_error = 1
    new_error = 1
    error_ratio = 2
    
    ctr = 0
   
    while abs(error_ratio) > 0.0001:
        
        old_error = new_error

        x_old = x_new

        to_norm = y - np.dot(H , x_old);       
        #print "to_norm shape =  " ,to_norm.shape[0],  to_norm.shape[1]
       # print "x_old shape =  " ,x_old.shape[0],  x_old.shape[1];
                                                
        data = (x_old + np.dot( H.transpose(),(to_norm))/alpha)
        #print "data =  " ,data,  ;
        scalar   = lambda_val / 2.0 * alpha ;

        x_new = pywt.threshold(data, scalar, 'soft')
        
        new_error = LA.norm((y.transpose() - np.dot(H , x_new)), 2)
        error_ratio = (old_error - new_error)/ old_error
        ctr = ctr + 1   
        #print "data = " , data
    return x_new


# Iterated soft thresholding algorithm
def DCT_basis_gen(N):
    """ISTA algorithm.
      Arguments
    ---------
    N : number
        number of vectors to be made.
   
    """
    dict = np.zeros((N ,N))
    h = np.zeros((N ,N))
    temp = np.zeros(N)
    for i in range(N):
        if (i == 0):
            alpha = math.sqrt((1.0/N))# ^ (1/2)   #check
        else:
            alpha = math.sqrt((2.0/N))## ^ (1/2)   
             
        for j in range(N):
            temp[j] = (math.cos( math.pi * i * (2*j+ 1) / (2 * N)))
        
        h[:,i] = alpha * temp.transpose()
        
    dict = h
    return dict
    
# Iterated soft thresholding algorithm
def denoise_image(noise_image):
    window_size_r = 8
    window_size_c = 8
    window = np.zeros((window_size_r ,window_size_c))
    x_basis = np.zeros((window_size_r ,window_size_r), dtype = float)
    y_basis = np.zeros((window_size_c ,window_size_c), dtype = float)
    temp = np.zeros((window_size_r ,window_size_c), dtype = float)
    new_image = np.zeros((noise_image.shape[0] ,noise_image.shape[1]))
    #temp_to_vector = np.zeros((1 ,(window_size_r * window_size_c)), dtype = float)
    image_basis = np.zeros(((window_size_r * window_size_c), (window_size_r * window_size_c)), dtype=float)
    
    x_basis = DCT_basis_gen(window_size_r)
    y_basis = DCT_basis_gen(window_size_c)
    for i in range(window_size_r):
                temp = np.outer(x_basis[:,i], y_basis[:,i]);
                image_basis[:,i] = temp.reshape(1, (window_size_r * window_size_c) )
            
    for r in range(0,noise_image.shape[0] - window_size_r, window_size_r):
        print "r value = ", r
        for c in range(0,noise_image.shape[1] - window_size_c, window_size_c):
            if (((r+window_size_r) < noise_image.shape[0]) and ((c+window_size_c) < noise_image.shape[1])):
                window = noise_image[r:r+window_size_r,c:c+window_size_c]
                window_to_image_vector = window.reshape( (window_size_r * window_size_c), 1 )
                
                image_v_denoised = ista( window_to_image_vector,image_basis, .1)
                
                recons_image_v = np.dot(image_basis , image_v_denoised)
                #print "test", image_v_denoised.shape[0], image_v_denoised.shape[1]
                recons_image = recons_image_v.reshape(window_size_r,  window_size_c)
                new_image[r:r+window_size_r,c:c+window_size_c] = recons_image
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
    
    dict = DCT_basis_gen(4)
   # print dict, dict.transpose()
    x_send = np.matrix('0 1 0 -2')
    y = dict * x_send.transpose()


    x_found = ista(y,dict, .01)
    print "x_send =" , x_send
    print "x_found =" , x_found
    
    
    
if __name__== "__main__":
    main()
