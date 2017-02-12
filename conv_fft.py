# -*- coding: utf-8 -*-
"""
Created on Sat Feb 8 15:15:15 2017

@author: denny
"""

import cv2 # load opencv
import numpy as np
import sys
import matplotlib.pyplot as plt

## 1D Convolution Function definition is here
def do_subconv(input_image,x,y,filter_arr, filter_divider = 1):
    #print "point  x: ", x , "y  : ",y
    i_rows, i_columns = input_image.shape    
    rows, columns = filter_arr.shape    
    sum =0.0
    for j in range(rows):
        row_flipped = rows - j -1;
        for k in range(columns):
           column_flipped = columns - k -1;
           image_xindex = (x + j - int(rows/2))
           image_yindex = (y + k - int(columns/2))
           
           #print "xindex ", xindex, "yindex", yindex)
           if (image_xindex < 0             or 
               image_xindex >= i_rows       or 
               image_yindex < 0             or 
               image_yindex >= i_columns ):
                       sum = sum + 0
           else :
#           print "filter x  : ", row_flipped, "filter y  : ", column_flipped
           #print "padded_image ", padded_image[x + j - 1,y + k - 1], "* filter_arr  : ",filter_arr[row_flipped,column_flipped]
           
               sum = sum + ((input_image[image_xindex,image_yindex]) * (filter_arr[row_flipped,column_flipped]) / filter_divider)
#           print "sum  : ", sum
    return sum
         
def do_display_fft(input_2d_array):
    fft2 = np.fft.fft2(input_2d_array)
    freq = np.fft.fftshift(fft2)
    
    return freq
    
## 2D Convolution Function definition is here
def conv2d( image_arr, filter_arr, filter_divider = 1 ):
   
    # convolve both arrays and print the result."
    #padded_image = pad_zero(image_arr)
    rows, columns = image_arr.shape
   
    #conv_result = np.array([range(rows),range(columns)])
    conv_result = np.zeros((rows ,columns ),dtype=np.float32)
    norm_image = np.zeros((rows ,columns ),dtype=np.uint8)
    
    for i in range(rows):
        for j in range(columns):
            #if i >= 0 and j >= 0 and i < (rows -2) and j < (columns - 2):
            temp = do_subconv(image_arr, i , j , filter_arr, filter_divider);
            conv_result[i,j] = temp
                #print "conv_result  : ", conv_result[i,j]

    #conv_result = int(conv_result / maximum * 255.0)
    cv2.normalize(conv_result, norm_image, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U )
    print "Inside the function conv2d conv_result : ", conv_result
 
#    cv2.imshow('dst_rt', norm_image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return  norm_image

# To Call the conv2d function
def pad_zero( gray_image):
    height, width = gray_image.shape
    _result = np.zeros((height+2, width+2),'uint8')
    _result[1:(height+1), 1:(width+1)] = gray_image
    print _result
    return _result

      

def main():

    file = raw_input('Enter the input filename: ')
    #load image into environment
    try:
        img = cv2.imread(file)
    except:
        print "Unexpected error:", sys.exc_info()[0]
        sys.exit(1)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
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
            filter[i,j] = int(input('Enter filter[' + str(i) +', ' + str(j) + ']  = '))            

#    filter = np.append(filter, np.array([[d,m,n]]), axis=0)
    norm_image = conv2d(gray_image, filter, d)    
    
    
    image_freqresp = do_display_fft(gray_image)
    normimage_freqresp = do_display_fft(norm_image)
    filter_freqresp = do_display_fft(filter)
    
    image_magnitude_spectrum = np.log(np.abs(image_freqresp))
    normimage_magnitude_spectrum = np.log(np.abs(normimage_freqresp))
    filter_magnitude_spectrum = np.log(np.abs(filter_freqresp))

    plt.subplot(321),plt.imshow(gray_image, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(322),plt.imshow(image_magnitude_spectrum, cmap = 'gray')
    plt.title('Image Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(323),plt.imshow(norm_image, cmap = 'gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(324),plt.imshow(normimage_magnitude_spectrum, cmap = 'gray')
    plt.title('Output Image Spectrum'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(325),plt.imshow(filter_magnitude_spectrum, cmap = 'gray')
    plt.title('Filter Freq Response'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    
    #do_display_fft(norm_image)
    #do_display_fft(filter)
    cv2.destroyAllWindows()
    
    
if __name__== "__main__":
    main()



