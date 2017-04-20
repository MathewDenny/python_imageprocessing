# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import math
def leastsquares(imagepatch1, imagepatch2,w,h):
#    print "finding distances "
#    s = (imagepatch1 - imagepatch2)^2    
#    d= sqrt(sum(s)); #this is euclidean norm
    R1 = imagepatch1[:,:,0] 
    R2 = imagepatch2[:,:,0]
    G1 = imagepatch1[:,:,1]
    G2 = imagepatch2[:,:,1]
    B1 = imagepatch1[:,:,2]
    B2 = imagepatch2[:,:,2]
    s = (R1-R2)**2+(G1-G2)**2+(B1-B2)**2

    svector = s.reshape( 1,(w * h))
    d= math.sqrt(np.sum(svector)); # this is euclidean norm
       
    return d

#    dist = np.linalg.norm(imagepatch1 - imagepatch2)
#    return dist   
       
cap = cv2.VideoCapture('q1.avi')
min_area = 1000
Width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
Height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print Width, Height
#FrameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
#numFrames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
fgbg = cv2.createBackgroundSubtractorMOG2()
count = 1
while cap.isOpened():
    ret,frame = cap.read()
    
    fgmask = fgbg.apply(frame)
#    cv2.imshow('processedframe ' + str(count),fgmask)
    count = count + 1
#    cv2.waitKey()
    if count == 12:
        print "Exiting"
        break
cv2.imshow('processedframe ' + str(count),fgmask)
cv2.waitKey()



thresh = cv2.dilate(fgmask, None, iterations=2)
print thresh.size

_, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    
# loop over the contours
for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < min_area:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
#		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
       
    
cv2.imshow("Bounding", frame)
cv2.waitKey()
reference_frame = frame
ret,frame = cap.read()
Width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
Height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.imshow("Next", frame)
cv2.waitKey()
window_size_r =  h
window_size_c =  w
xpos = y
ypos = x
print x,x+window_size_r, y,y+window_size_c, Width, Height, window_size_r, window_size_c

mindist = 1e10

ref = reference_frame[xpos:xpos+window_size_r,ypos:ypos+window_size_c,:]



for r in range(0,Height , 1):      
        print "r value = ", r , "of ", Height
#        refvector = ref.reshape( 1,((window_size_r) * (window_size_c)) )
        for c in range(0,Width, 1):
            
            if (((r+window_size_r) < Height) and ((c+window_size_c) < Width)):
                patch = frame[r:r+window_size_r, c:c+window_size_c,:]
#                patch_to_patchvector = window.reshape(1, (window_size_r * window_size_c))
                dist = leastsquares(patch, ref, window_size_r, window_size_c)
                if dist < mindist:
                    mindist = dist
                    position = [r, c]

print position, x, y
cv2.rectangle(frame, (position[1], position[0]), (position[1] + w,  position[0] + h), (0, 255, 0), 2)

    
cv2.imshow("Result", frame)
cv2.waitKey()

cv2.destroyAllWindows()
cap.release()


