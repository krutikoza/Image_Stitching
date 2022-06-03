import cv2
import numpy as np
import os
import random
from PIL import Image
import sys

import matplotlib.pyplot as plt


# get feature points
# def featureMatching(images_name_list):
#     nfeatures = 500
#     orb = cv2.ORB_create(nfeatures)
#     for image_I in range(len(images_name_list)-1):
#         print("Reference Image",images_name_list[image_I])
#         img_I_cv= cv2.imread(os.path.join(os.path.dirname(__file__),images_name_list[image_I]), cv2.IMREAD_GRAYSCALE)
#         (keypoints_I, descriptors_I) = orb.detectAndCompute(img_I_cv, None)
#         for image_J in range(1,len(images_name_list),1):
#             if images_name_list[image_I]!=images_name_list[image_J]:
#                 print("Compared Image",images_name_list[image_J])
#                 img_J_cv = cv2.imread(os.path.join(os.path.dirname(__file__),images_name_list[image_J]), cv2.IMREAD_GRAYSCALE)
#                 (keypoints_J, descriptors_J) = orb.detectAndCompute(img_J_cv, None)
#                 number_of_matches = []
#                 difference=[]
#                 for descriptor_in_I in range(len(descriptors_I)):
#                     distance_match = []
#                     for descriptor_in_J in range(len(descriptors_J)):
#                         distance_match.append(cv2.norm(descriptors_I[descriptor_in_I],descriptors_J[descriptor_in_J], cv2.NORM_HAMMING))
#                     top_2_index = np.array(distance_match).argsort()[:2]
#                     # print("Top 2 values",[distance_match[j] for j in top_2_index])
#                     threshold = 0.75
#                     match_or_not = distance_match[top_2_index[0]]/distance_match[top_2_index[1]]
#                     # print("match or not",match_or_not)
#                     if match_or_not<=threshold:
#                         number_of_matches.append([keypoints_I[descriptor_in_I].pt,keypoints_J[top_2_index[0]].pt])
                        
#                 #print("number_of_matches:",(number_of_matches))
                
#     return number_of_matches,difference
                
                
                
def featureMatching(image1, image2):
    
    
    '''
    img_I_cv

    '''
    orb = cv2.ORB_create()
    (keypoints_I, descriptors_I) = orb.detectAndCompute(img_I_cv, None)
    (keypoints_J, descriptors_J) = orb.detectAndCompute(img_J_cv, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_I,descriptors_J,k=2)
    matches_threshold = []
    for m,n in matches:
        if m.distance/n.distance < 0.9:
            matches_threshold.append([m.queryIdx,m.trainIdx])
            
    
    keypoints =[]
    for i in matches_threshold:
        keypoints.append([(keypoints_I[i[0]].pt[0], keypoints_I[i[0]].pt[1]), (keypoints_J[i[1]].pt[0], keypoints_J[i[1]].pt[1])])
        
    # print(keypoints)
    return keypoints



                
                
                
    
    
def ransac(number_of_matches):
    number_of_iteration = 100
    max_count = -9999  
    final_src_features = []
    final_dest_features = []
    
    for p in range(number_of_iteration):    
        
        
        selected_points = random.sample(range(len(number_of_matches)-1), 4)
        
        
        
        selected_values = {}
        
        for i in range(len(selected_points)):
            selected_values["x"+str(i+1)] = number_of_matches[selected_points[i]][0][0]
            selected_values["y"+str(i+1)] = number_of_matches[selected_points[i]][0][1]
            
            selected_values["x_"+str(i+1)] = number_of_matches[selected_points[i]][1][0]
            selected_values["y_"+str(i+1)] = number_of_matches[selected_points[i]][1][1]
            
        
        src_points = [selected_values["x1"],selected_values["y1"],selected_values["x2"],selected_values["y2"],selected_values["x3"],selected_values["y3"],selected_values["x4"],selected_values["y4"]]
        dest_points = [selected_values["x_1"],selected_values["y_1"],selected_values["x_2"],selected_values["y_2"],selected_values["x_3"],selected_values["y_3"],selected_values["x_4"],selected_values["y_4"]]
    
    
        print("src_points::", src_points)
        print("dest_points::", dest_points)
        H = get_transform_mat(4, dest_points, src_points)
        
        
        H_inv = np.linalg.inv(H)
        
        count = 0   
        
        
        src_features = []
        dest_features = []         

        src_x = []
        src_y = []
        

        for i in range(len(number_of_matches)):
            coords = np.array([number_of_matches[i][0][0],number_of_matches[i][0][1],1])
            #print("coords::",coords)
            new_i, new_j, scale = H_inv @ coords
            new_i, new_j = int(new_i/scale), int(new_j/scale)
            
            
            if new_i > number_of_matches[i][1][0]-0.5 and new_i < number_of_matches[i][1][0]+0.5 and new_j > number_of_matches[i][1][1]-0.5 and new_j < number_of_matches[i][1][1]+0.5:
                src_features.append(number_of_matches[i][0][0])
                src_features.append(number_of_matches[i][0][1])
                
                src_x.append(number_of_matches[i][0][0])
                src_y.append(number_of_matches[i][0][1])
                
                dest_features.append(number_of_matches[i][1][0])
                dest_features.append(number_of_matches[i][1][1])
                count +=1
                
                
        if max_count<count:
            max_count = count
            # print(count)
            final_src_features = src_features
            final_dest_features = dest_features
            f_H=H
    
    
    return final_src_features, final_dest_features,f_H
                
            
                
            
    
        

    
    
    
# use the ransac feature points where we removed outliers to get transformation matrix

def get_transform_mat(option, src, dest):
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
        sh1 = np.array([[1, -np.tan(angle/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
        sh2 = np.array([[1, 0,0],[np.sin(angle),1,0],[0,0,1]]).reshape(3,3)
        sh3 = np.array([[1, -np.tan(angle/2),0],[0,1,0],[0,0,1]]).reshape(3,3)
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
        #print(np.shape(A), np.shape(b))
        H = np.linalg.solve(A,b)
        H = np.append(H, 1)

    H = H.reshape(3,3)
    
    return H



def get_transform_mat_n(src, dest):
    
    
    val_dict_src = {}
    val_dict_dest = {}
    count = 1
    for i in range(0,len(src),2):
        #source
        var_x = "x_"+str(count)
        var_y = "y_"+str(count)
        
        
        val_dict_src[var_x] = src[i]
        val_dict_src[var_y] = src[i+1]
        
        # destination
        var_x = "x__"+str(count)
        var_y = "y__"+str(count)
        count +=1
        val_dict_dest[var_x] = dest[i]
        val_dict_dest[var_y] = dest[i+1]
        
    
    A = []
    b = []
    
    for i in range(int(len(val_dict_src)/2)):
        temp1 = [val_dict_src["x_"+str(i+1)],val_dict_src["y_"+str(i+1)],1,0,0,0, -val_dict_src["x_"+str(i+1)]*val_dict_dest["x__"+str(i+1)], -val_dict_src["y_"+str(i+1)]*val_dict_dest["x__"+str(i+1)]]
        temp2 = [0,0,0,val_dict_src["x_"+str(i+1)],val_dict_src["y_"+str(i+1)],1, -val_dict_src["x_"+str(i+1)]*val_dict_dest["y__"+str(i+1)], -val_dict_src["y_"+str(i+1)]*val_dict_dest["y__"+str(i+1)]]
        
        A.append(temp1)
        A.append(temp2)
        
        b.append(val_dict_dest["x__"+str(i+1)])
        b.append(val_dict_dest["y__"+str(i+1)])


    A = np.array(A)
    b = np.array(b)
    
    # b = np.array([x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_])
    
    H = np.linalg.solve(A,b)
    H = np.append(H, 1)
    
    H = H.reshape(3,3)   
    
    return H     
                    
                           
                           





def bilinear_interpolation(image, x,y):

    # reference: https://en.wikipedia.org/wiki/Bilinear_interpolation#Example
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



def inverse_warp(image1_arr,m,n,H, original_img):
    
    x1 = 0
    y1 = 0
    x2 = len(original_img)
    y2 = 0
    x3 = 0
    y3 = len(original_img[0])
    x4 = len(original_img[1])
    y4 = len(original_img[1])
    points = [x1, y1, x2, y2, x3, y3,x4, y4]
    points = [0,0,n,0,0,m,n,m]
    x_transformed = []
    y_transformed = []
    transformed_points = []
    
    for i in range(0,len(points),2):
        coords = np.array([points[i],points[i+1],1])
        new_i, new_j, scale = H @ coords
        new_i, new_j = int(new_i/scale), int(new_j/scale)
        transformed_points.append(new_i)
        transformed_points.append(new_j)
        x_transformed.append(new_i)
        y_transformed.append(new_j)
        
    for i in range(0,len(transformed_points),2):
        plt.plot(transformed_points[i], transformed_points[i+1], marker='o', color="red", markersize=5)
    plt.imshow(image1_arr)
    plt.show()
    
    print("Original points::",points)
    print("transformed_points::",transformed_points)
    
    
    # min max values of image after transformation of image2
    min_x = np.min(x_transformed)
    max_x = np.max(x_transformed)
    min_y = np.min(y_transformed)
    max_y = np.max(y_transformed)
    
    
    # min max values of transformed image and image1 or original image
    
    min_x_val = min(min_x, 0)
    max_x_val = max(max_x, n)
    min_y_val = min(min_y, 0)
    max_y_val = max(max_y, m)
    
    
    print("min_x_val:",min_x_val)
    print("max_x_val:",max_x_val)
    print("min_y_val:",min_y_val)
    print("max_y_val:",max_y_val)
    
    
    
    # new empty image with dimension of original image1 and the transformad image2:
    
    
    new_image = np.zeros( [abs(int(min_y_val)- int(max_y_val)), abs(int(min_x_val)- int(max_x_val)),3], dtype=np.uint8)
    new_image1= np.zeros( [abs(int(min_y_val)- int(max_y_val)), abs(int(min_x_val)- int(max_x_val)),3], dtype=np.uint8)
    
    
    new_image = np.zeros( [abs(len(original_img))-int(min_y_val), abs(int(max_x_val)),3], dtype=np.uint8)
    new_image1 = np.zeros( [abs(len(original_img))-int(min_y_val), abs(int(max_x_val)),3], dtype=np.uint8)
    #print("np.shape(new_image):::",np.shape(new_image))
    

    
    # print("np.shape(new_image[0+min_x:,0+min_y:]):",np.shape(new_image[int(min_x):,:-int(max_y)]))
    # print("image2_arr[:,:]::", np.shape(image1[:,:]))
    
    
    new_image[-image1_arr.shape[0]:,:image1_arr.shape[1],:] = image1_arr[:,:,:]
    
    
    plt.imshow(np.array(new_image))
    plt.show()
    
    
    
    
    
    
    
    H_inv = np.linalg.inv(H)
    # print("x_transformed:::", x_transformed)
    # print("min_x::",min_x)
    # print("max_x::",max_x)
    # print("y_transformed::",y_transformed)
    # print("min_y::",min_y)
    # print("max_y::",max_y)

    transformed_image2 = np.zeros([abs(int(min_y)- int(max_y)),abs(int(min_x)- int(max_x)),3], dtype=np.uint8)
    print("transformed_image2::", np.shape(transformed_image2))
    R = 0
    C = 0
    n_i = []
    n_j = []
    for r in range(int(min_y),int(max_y),1):#int(min_x),int(max_x),1):
        C = 0
        for c in range(int(min_x),int(max_x),1):#int(min_y),int(max_y),1):
            # adjust for offset
            coords = np.array([c,r,1])
            new_i, new_j, scale = H_inv @ coords
            n_i.append(new_i)
            n_j.append(new_j)
            
            new_i, new_j = int(new_i/scale), int(new_j/scale)
            
            # perform bilinear interpolation
            
            if 0 <= new_j < (m - 1) and 0 <= new_i < (n - 1):
                transformed_image2[R, C, :] =  bilinear_interpolation(original_img, new_i, new_j)
                
            C+=1

        R+=1    
        
    
    plt.imshow(np.array(transformed_image2))
    plt.title('Transformed_image2')
    plt.show()
    
    
    plt.imshow(np.array(new_image))
    plt.title('Before:')
    plt.show()
    
    
    new_image1[:np.shape(transformed_image2)[0],-np.shape(transformed_image2)[1]:,:] = transformed_image2[:,:,:]
    
    
    plt.imshow(np.array(new_image1))
    plt.title('new_image1')
    plt.show()
    
    
    plt.imshow(np.maximum(new_image,new_image1))
    plt.title('combined')
    plt.show()
    
    
    final_image= np.zeros( [abs(int(min_y_val)- int(max_y_val)), abs(int(min_x_val)- int(max_x_val)),3], dtype=np.uint8)
    
    # for i in range(len(new_image)):
    #     for j in range(len(new_image[0])):
    #     if new_image[i][j][:] != 0:
    #         if new_image1[i][j][:] !=0:
                
    
    
    
    
    # final_image= np.zeros( [abs(int(min_y_val)- int(max_y_val)), abs(int(min_x_val)- int(max_x_val)),3], dtype=np.uint8)
    
    # for i in range(len(new_image)):
    #     for j in range(len(new_image[0])):
            
    #         if i<len(transformed_image2) and j< len(transformed_image2[0]) and  i > max_x:
    #             final_image[i][j] =  transformed_image2[i+min_x][j+min_y]
            
            
    
    # plt.imshow(final_image)
    # plt.title('combined')
    # plt.show()
    
    
    return np.maximum(new_image,new_image1)






image1 = sys.argv[1]
image2 = sys.argv[2]
img_I_cv= cv2.imread(image1)
img_J_cv = cv2.imread(image2)
image1_arr = np.array(img_I_cv)
image2_arr = np.array(img_J_cv)


number_of_matches = featureMatching(img_I_cv,img_J_cv)



src_feature_point, dest_feature_point, H = ransac(number_of_matches)



# for i in range(0,len(src_feature_point),2):
#     plt.plot(src_feature_point[i], src_feature_point[i+1], marker='o', color="red", markersize=2)
# plt.imshow(image1_arr)
# plt.show()

# for i in range(0,len(dest_feature_point),2):
#     plt.plot(dest_feature_point[i], dest_feature_point[i+1], marker='o', color="red", markersize=2)
# plt.imshow(image2_arr)
# plt.show()


    

# H = get_transform_mat(4,src_feature_point[0:8],dest_feature_point[0:8])

# H = get_transform_mat_n(src_feature_point, dest_feature_point)








inverse_image = inverse_warp(image1_arr,image2_arr.shape[0], image2_arr.shape[1],H,image2_arr)


inverse_image_pil = Image.fromarray((inverse_image).astype(np.uint8))
# plt.imshow(np.array(inverse_image_pil))
# plt.show()