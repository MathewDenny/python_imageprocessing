# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 00:52:05 2017

@author: denny
"""

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
    print "rows cols = ", rows, columns
    
    x_new = np.zeros((rows,1))
    x_old = np.zeros((rows,1))
    
    alpha_vals = LA.eigvals((H.transpose() * H))
    
   
    alpha = np.amax(alpha_vals)
    
    print "alpha = " , alpha;
    old_error = 1
    new_error = 1
    error_ratio = 2
    
    ctr = 0
   
    while abs(error_ratio) > 0.0001:
        
        old_error = new_error

        x_old = x_new

        to_norm = y - np.dot(H , x_old);       
        #print "to_norm " , to_norm;
        data = (x_old + ( H.transpose()*(to_norm))/alpha)
        
        scalar   = lambda_val / 2.0 * alpha ;

        x_new = pywt.threshold(data, scalar, 'soft')
        
        new_error = LA.norm((y.transpose() - H * x_new.transpose()), 2)
        error_ratio = (old_error - new_error)/ old_error
        ctr = ctr + 1   
        print "data = " , data
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
    
    
def main():
    #y = np.matrix('2 1 4 0')
    #d = np.matrix('9 11 0 4')
    
    dict = DCT_basis_gen(4)
   # print dict, dict.transpose()
    x_send = np.matrix('0 1 0 -2')
    y = dict * x_send.transpose()


    x_found = ista(y,dict, .01)
    print "x_send =" , x_send
    print "x_found =" , x_found
    
    
    
if __name__== "__main__":
    main()
