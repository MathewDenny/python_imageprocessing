# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:25:51 2017

@author: Denny
"""
import numpy as np
import cv2
import scipy as sp
def detect_local_maxima(image,r,c):
    thresh = 0.5
    if (image[r,c] > thresh * image.max()):         
        return 1
    else:
        return 0

def H_from_points(left,right):
  
    if left.shape != right.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points (important for numerical reasons)
    # --from points--
#    m = mean(fp[:2], axis=1)
#    maxstd = max(std(fp[:2], axis=1)) + 1e-9
#    C1 = diag([1/maxstd, 1/maxstd, 1]) 
#    C1[0][2] = -m[0]/maxstd
#    C1[1][2] = -m[1]/maxstd
#    fp = dot(C1,fp)
#    
#    # --to points--
#    m = mean(tp[:2], axis=1)
#    maxstd = max(std(tp[:2], axis=1)) + 1e-9
#    C2 = diag([1/maxstd, 1/maxstd, 1])
#    C2[0][2] = -m[0]/maxstd
#    C2[1][2] = -m[1]/maxstd
#    tp = dot(C2,tp)
    
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = left.shape[0]
    A = np.zeros((2*nbr_correspondences,9))
    print "nbr_correspondences - " ,nbr_correspondences
    for i in range(nbr_correspondences):        
        A[2*i] = [-left[i][0],-left[i][1],-1,0,0,0,
                    right[i][0]*left[i][0],right[i][0]*left[i][1],right[i][0]]
        A[2*i+1] = [0,0,0,-left[i][0],-left[i][1],-1,
                    right[i][1]*left[i][0],right[i][1]*left[i][1],right[i][1]]
#        A[2*i] = [-left[i][1],-left[i][0],-1,0,0,0,
#                    right[i][1]*left[i][1],right[i][1]*left[i][0],right[i][1]]
#        A[2*i+1] = [0,0,0,-left[i][1],-left[i][0],-1,
#                    right[i][0]*left[i][1],right[i][0]*left[i][0],right[i][0]]
    
    U,S,V =  np.linalg.svd(A)
    H = V[8].reshape((3,3))    
    
#    # decondition
#    H = dot(linalg.inv(C2),dot(H,C1))
    
    # normalize and return
    return H / H[2,2]

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
        pt_a = (int(match_list1[m][0]), int(match_list1[m][1]))
        pt_b = (int(match_list2[m][0] + w1), int(match_list2[m][1]))
        print "ploting = ", pt_a, pt_b
        
        cv2.line(view, pt_a , pt_b, (0,0,128),1)
        cv2.circle(view,pt_a, 5, (0,128,0), 1)
        cv2.circle(view,pt_b, 5, (255,128,0), 1)



    cv2.imshow("dst_rt", view)
    cv2.waitKey()
    
def find_good_points( harris_image):

    r = harris_image.shape[0]
    c = harris_image.shape[1]
    
    # Create an empty list to hold our points of interest.
    c_points = []
    exit = 0
    for i in range(r):        
        if (exit == 1):
            break
        for j in range(c):            
          
            is_good = detect_local_maxima(harris_image, i,j)
            if (is_good == 1):
                c_points.append([[j,i]])
#                maxima_found = maxima_found + 1 
#                print "Selected",  window, j, i
    c_points = np.float32(c_points)
    return c_points
                
                
def main():
    #y = np.matrix('2 1 4 0')
    #d = np.matrix('9 11 0 4')

 
    cap = cv2.VideoCapture('q2.avi')
   
    Width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    Height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print Width, Height
    ret,frame1 = cap.read()
    gray_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#    gray_image1 = np.float32(gray_image1)
    
    ret,frame2 = cap.read()
    ret,frame3 = cap.read()
    ret,frame4 = cap.read()
    gray_image4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
#    gray_image4 = np.float32(gray_image4)
    p0 = cv2.goodFeaturesToTrack(gray_image1, mask = None, maxCorners = 20,qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7)
#    print "shape = ",p0.shape, p0
    dest = cv2.cornerHarris(gray_image1,blockSize = 5, ksize = 3, k = 0.1 )
    print dest
    # Threshold for an optimal value, it may vary depending on the image.
#    frame1[dest>(0.1 * dest.max())]=[0,0,255]
    
 
    orig_points = find_good_points(dest)
    print orig_points.shape, orig_points
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(gray_image1, gray_image4, orig_points , None)
    
    # Select good points
    good_new = nextPts[status==1]
    print nextPts
    good_old = orig_points[status==1]
    # Create a mask image for drawing purposes
    mask = np.zeros_like(gray_image1)
    
    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    plot_matching_points(gray_image1, gray_image4, good_old, good_new)
    
#    # draw the tracks
#    for i,(new,old) in enumerate(zip(good_new,good_old)):
#        print "NEW =", new
#        print "old =", old
#        a,b = new.ravel()
#        c,d = old.ravel()
#        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
#        frame = cv2.circle(gray_image4,(a,b),5,color[i].tolist(),-1)
#    img = cv2.add(frame,mask)
#
#    cv2.imshow('frame',img)
    
     
    H = H_from_points(good_old,good_new)
    print H
    new_transformed = np.zeros(gray_image4.shape)
    r = gray_image4.shape[0]
    c = gray_image4.shape[1]
    
   
#    for i in range(r):        
#        for j in range(c): 
#            a = np.array([[i],[j],[1]])
#            x,y,z = np.dot(H ,a)
#            print  x/z , y/z , z
#            new_transformed[int(x/z),int(y/z)] = gray_image4[i,j]
#    new_transformed = np.dot(H,good_new)
    h, status = cv2.findHomography(good_old, good_new)
    print h
    im_dst = cv2.warpPerspective(gray_image4, H, (r,c))
    cv2.imshow('frame',im_dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
   
               
   
    
if __name__== "__main__":
    main()
