from distutils.log import error
import cv2
import numpy as np
import os
import random
from PIL import Image
import sys

import matplotlib.pyplot as plt

                
# orb            
def featureMatching_ransac(image1, image2):
    
    '''
    img_I_cv

    '''
    orb = cv2.ORB_create()
    (keypoints_I, descriptors_I) = orb.detectAndCompute(image1, None)
    (keypoints_J, descriptors_J) = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_I,descriptors_J,k=2)
    matches_threshold = []
    for m,n in matches:
        if m.distance/n.distance < 0.9:
            matches_threshold.append([m.queryIdx,m.trainIdx])
            
    
    keypoints =[]
    for i in matches_threshold:
        keypoints.append([(keypoints_I[i[0]].pt[0], keypoints_I[i[0]].pt[1]), (keypoints_J[i[1]].pt[0], keypoints_J[i[1]].pt[1])])
        
        
    return keypoints

    
                
    
# ransac
def ransac(number_of_matches):
    # set number of iteration
    number_of_iteration = 200
    # Variable to save max voted homography matrix
    max_count = -9999 
    # matrix to store final feature points after removing outliers
    final_src_features = []
    final_dest_features = []
    p = 0
    while p <= number_of_iteration:
        p+=1
        # get 4 random points index
        selected_points = random.sample(range(len(number_of_matches)-1), 4)
        selected_values = {}
        
        # get x,y values of those random points
        for i in range(len(selected_points)):
            selected_values["x"+str(i+1)] = number_of_matches[selected_points[i]][0][0]
            selected_values["y"+str(i+1)] = number_of_matches[selected_points[i]][0][1]
            
            selected_values["x_"+str(i+1)] = number_of_matches[selected_points[i]][1][0]
            selected_values["y_"+str(i+1)] = number_of_matches[selected_points[i]][1][1]
            
        
        # save points in single array. Example: src_points = [x1,y1, x2,y2, x3,y3, x4,y4]
        src_points = [selected_values["x1"],selected_values["y1"],selected_values["x2"],selected_values["y2"],selected_values["x3"],selected_values["y3"],selected_values["x4"],selected_values["y4"]]
        dest_points = [selected_values["x_1"],selected_values["y_1"],selected_values["x_2"],selected_values["y_2"],selected_values["x_3"],selected_values["y_3"],selected_values["x_4"],selected_values["y_4"]]
    
        try:
            # get transformation matrix using pair of 4 points
            H = get_transform_mat_ransac(4, dest_points, src_points)
        
            # inverse of transformation matrix
            H_inv = np.linalg.inv(H)
        except:
            if p!=0:
                p-=1
            continue
        
        

        count = 0   
                
        src_features = []
        dest_features = []         

        src_x = []
        src_y = []
        
        # loop on each feature point and check how good the transformation matrix is.
        for i in range(len(number_of_matches)):
            coords = np.array([number_of_matches[i][0][0],number_of_matches[i][0][1],1])
            new_i, new_j, scale = H_inv @ coords
            new_i, new_j = int(new_i/scale), int(new_j/scale)
            
            # threshold 
            if new_i > number_of_matches[i][1][0]-0.5 and new_i < number_of_matches[i][1][0]+0.5 and new_j > number_of_matches[i][1][1]-0.5 and new_j < number_of_matches[i][1][1]+0.5:
                src_features.append(number_of_matches[i][0][0])
                src_features.append(number_of_matches[i][0][1])
                
                src_x.append(number_of_matches[i][0][0])
                src_y.append(number_of_matches[i][0][1])
                
                dest_features.append(number_of_matches[i][1][0])
                dest_features.append(number_of_matches[i][1][1])
                count +=1
                
        # if more feature point agree with this transformation matrix, save it
        if max_count<count:
            max_count = count
            final_src_features = src_features
            final_dest_features = dest_features
            f_H=H
    
    # return source, destination feature points and transformation matrix with maximum vote.
    return final_src_features, final_dest_features,f_H
                
            
                
            
    
        

    
    
    
# get transformation matrix
def get_transform_mat_ransac(option, src, dest):
    if option==1:
        x1,y1 = src
        x1_, y1_ = dest
        H = np.array([[1,0,x1-x1_],[0,1,y1-y1_],[0,0,1]])
    elif option==2:
        x1,y1, x2, y2 = src
        x1_, y1_, x2_, y2_ = dest
        T = np.array([[1,0,x1-x1_],[0,1,x2-x2_],[0,0,1]])
        T = T.reshape(3,3)

        m1 = (y2_-y1_)/(x2_-x1_)
        m2 = (y2-y1)/(x2-x1)

        rad = np.arctan(np.abs(m2-m1)/(1+m2*m1))
        angle = np.rad2deg(rad)
        sh1 = np.array([[1, -np.tan(rad/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
        sh2 = np.array([[1, 0,0],[np.sin(rad),1,0],[0,0,1]]).reshape(3,3)
        sh3 = np.array([[1, -np.tan(rad/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
        H = T  @ sh1 @sh2 @sh3
    elif option==3:
        x1, y1, x2, y2, x3, y3 = src
        x1_, y1_, x2_, y2_, x3_, y3_ = dest

        A = np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
        b = np.array([x1_, y1_, x2_, y2_, x3_, y3_])
        H = np.linalg.solve(A,b)
        H = np.append(H,[0,0,1])
    else:
        x1,y1,x2,y2,x3,y3,x4,y4 = src
        x1_,y1_,x2_,y2_,x3_,y3_,x4_,y4_ = dest
        A = np.array([[x1,y1,1,0,0,0, -x1*x1_, -y1*x1_],
        [0,0,0,x1,y1,1, -x1*y1_, -y1*y1_],
        [x2,y2,1,0,0,0, -x2*x2_, -y2*x2_],
        [0,0,0,x2,y2,1, -x2*y2_, -y2*y2_],
        [x3,y3,1,0,0,0, -x3*x3_, -y3*x3_],
        [0,0,0,x3,y3,1, -x3*y3_, -y3*y3_],
        [x4, y4, 1, 0,0, 0, -x4*x4_, -y4*x4_],
        [0,0,0,x4, y4, 1, -x4*y4_, -y4*y4_]])
        b = np.array([x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_])
        
        H = np.linalg.solve(A,b)
        H = np.append(H, 1)

    H = H.reshape(3,3)
    
    return H

                    
                           
                           





def bilinear_interpolation_ransac(image, x,y):

    # reference: https://en.wikipedia.org/wiki/bilinear_interpolation#Example
    height, width = image.shape[0],image.shape[1]

    x1 = int(np.floor(x))
    x2 = x1 + 1
    y1 = int(np.floor(y))
    y2 = y1 + 1
    
    # get pixels at locations based on x1,y1,x2,y2
    img_a = image[y1,x1]
    img_b = image[y2,x1]
    img_c = image[y1,x2]
    img_d = image[y2,x2]

    # capture deltas for x and y
    dx = x - x1
    dy = y - y1

    point = img_a * (1-dx)*(1-dy) + img_b * (dy)* (1-dx) + img_c * (dx)*(1-dy) + img_d * dx*dy

    return np.round(point)




def inverse_warp_ransac(image1_arr,m,n,H, original_img):
    
    # get four border points of the image
    points = [0,0,n,0,0,m,n,m]
    x_transformed = []
    y_transformed = []
    transformed_points = []
    
    # transform 4 points of image using selected transformation matrix.
    for i in range(0,len(points),2):
        coords = np.array([points[i],points[i+1],1])
        new_i, new_j, scale = H @ coords
        new_i, new_j = int(new_i/scale), int(new_j/scale)
        transformed_points.append(new_i)
        transformed_points.append(new_j)
        x_transformed.append(new_i)
        y_transformed.append(new_j)
        
    
    ## to see where the four points of transformed second image ovarlaps on first image, uncomment below code
    
    # for i in range(0,len(transformed_points),2):
    #     plt.plot(transformed_points[i], transformed_points[i+1], marker='o', color="red", markersize=5)
    # plt.imshow(image1_arr)
    # plt.show()
    
    
    # min max values after transformation of image2
    min_x = np.min(x_transformed)
    max_x = np.max(x_transformed)
    min_y = np.min(y_transformed)
    max_y = np.max(y_transformed)
    
    
    # min max values of transformed image and image1 or original image
    min_x_val = min(min_x, 0)
    max_x_val = max(max_x, n)
    min_y_val = min(min_y, 0)
    max_y_val = max(max_y, m)
    
    
    
    
    # new empty image with dimension of original image1 and the transformad image2:
    
    y_len = len(image1_arr) + abs(min_y)
    
    new_image = np.zeros( [abs(y_len), abs(int(max_x_val)),3], dtype=np.uint8)
    new_image1 = np.zeros( [abs(y_len), abs(int(max_x_val)),3], dtype=np.uint8)

    # copy first image to the bottom left corner of our final, large, empty image
    new_image[-image1_arr.shape[0]:,:image1_arr.shape[1],:] = image1_arr[:,:,:]
    ## display image
    # plt.imshow(np.array(new_image))
    # plt.show()
    
    # inverse of our transformation matrix
    H_inv = np.linalg.inv(H)

    # empty image require to store our transformed image.
    transformed_image2 = np.zeros([abs(int(min_y)- int(max_y)),abs(int(min_x)- int(max_x)),3], dtype=np.uint8)
    
    
    R = 0
    C = 0
    
    # transform image
    for r in range(int(min_y),int(max_y),1):
        C = 0
        for c in range(int(min_x),int(max_x),1):
            
            # adjust for offset
            coords = np.array([c,r,1])
            new_i, new_j, scale = H_inv @ coords
            new_i, new_j = int(new_i/scale), int(new_j/scale)
            
            # perform bilinear interpolation
            
            if 0 <= new_j < (m - 1) and 0 <= new_i < (n - 1):
                transformed_image2[R, C, :] =  bilinear_interpolation_ransac(original_img, new_i, new_j)
                
            C+=1

        R+=1    
        
    
    ## uncomment to display transformed image2
    # plt.imshow(np.array(transformed_image2))
    # plt.title('Transformed_image2')
    # plt.show()
    
    ## uncomment to display image where image1 which is inserted to bottom left corner
    # plt.imshow(np.array(new_image))
    # plt.title('Before:')
    # plt.show()
    
    # copy transformed image to top right corner of other empty image , size of final image
    new_image1[:np.shape(transformed_image2)[0],-np.shape(transformed_image2)[1]:,:] = transformed_image2[:,:,:]
    
    ## Uncomment to display transform image copied to other empty image
    # plt.imshow(np.array(new_image1))
    # plt.title('new_image1')
    # plt.show()
    
    # combine both images
    combined_image = np.maximum(new_image,new_image1)
    
    ## uncomment to display final image
    # plt.imshow(combined_image)
    # plt.title('combined')
    # plt.show()
    
    # save final image
    cv2.imwrite(sys.argv[3], combined_image)
    
    # return final image
    return combined_image




# import images and convert it to array
image1 = sys.argv[1]
image2 = sys.argv[2]
img_I_cv= cv2.imread(image1)
img_J_cv = cv2.imread(image2)
image1_arr = np.array(img_I_cv)
image2_arr = np.array(img_J_cv)

# get features using orb
number_of_matches = featureMatching_ransac(img_I_cv,img_J_cv)

# apply ransac to remove outliers
src_feature_point, dest_feature_point, H = ransac(number_of_matches)


# use homography matrix and feature points to get final image: 
inverse_image = inverse_warp_ransac(image1_arr,image2_arr.shape[0], image2_arr.shape[1],H,image2_arr)
