from matplotlib.pyplot import show
import numpy as np
import sys
import cv2
import glob#to read the files
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans # Our clustering algorithm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os

def featureMatching(images_name_list,cluster):
    '''
    Parameters
    -------------------------
    images_name_list: This is the list of all images that are passed through command line. The lenght would be equal to the number of images passed

    cluster: Number of clusters that was passed

    Returns
    -------------------------
    clustered_labels : Returns clustered labels of all given images
    '''
    similarity_matrix = []#This is the matrix that we would apply Clustering on
    for image_I in range(len(images_name_list)):#For all the images we get on eby one
        img_I_cv= cv2.imread(images_name_list[image_I], cv2.IMREAD_GRAYSCALE)#reading the image from given path
        matches_from_given_I_image = []#Creating the number of matches found from current image to the next image
        for image_J in range(len(images_name_list)):#for all other images
            if (images_name_list[image_I]!=images_name_list[image_J]):#If both images are not same, because we don't want to calculate all the matches of the same picture
                img_J_cv = cv2.imread(images_name_list[image_J], cv2.IMREAD_GRAYSCALE)#reading the second image
                matches_threshold = calculateMatches(img_I_cv,img_J_cv,False)#Function call to calculate the matches between two pictures
                matches_from_given_I_image.append(matches_threshold)
        similarity_matrix.append(matches_from_given_I_image)#append the whole matches list to generate a matrix
    similarity_matrix = np.array(similarity_matrix)#convert to array
    clustered_labels = clustering(cluster, similarity_matrix)#Apply clustering on the given similarity matrix
    return clustered_labels

def calculateMatches(img_I_cv,img_J_cv,show_SIFT_match=False):
    '''
    Parameters
    --------------------------------
    img_I_cv: First Image 
    img_J_cv: Second Image
    show_SIFT_match : Set to True if wanna visualize the matches between two images

    Returns
    --------------------------------
    Returns number of matches

    '''
    orb = cv2.ORB_create(nfeatures=500)#finding matches
    (keypoints_I, descriptors_I) = orb.detectAndCompute(img_I_cv, None)#take keypoints and descriptors of the matches from image 1
    (keypoints_J, descriptors_J) = orb.detectAndCompute(img_J_cv, None)#take keypoints and descriptors of the matches from image 2
    bf = cv2.BFMatcher()
    if descriptors_I is None:#If no features are found
        return 0
    if descriptors_J is None:#if no features are found, return zero 
        return 0
    matches = bf.knnMatch(descriptors_I,descriptors_J,k=2)#find top two matches
    matches_threshold = []
    for m,n in matches:
        if m.distance/n.distance < 0.75:
            matches_threshold.append([m])
    if show_SIFT_match:
        img3 = cv2.drawMatchesKnn(img_I_cv,keypoints_I,img_J_cv,keypoints_J,matches_threshold,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img3),plt.show()
    return len(matches_threshold)


def clustering(k,similarity_matrix):
    '''
    Parameters
    ---------------------------------
    k: Number of clusters
    similarity_matrix: Similarity Matrix that we get from function alculateMatches()

    Returns
    ---------------------------------
    Return labels of images
    '''

    std_wine = StandardScaler().fit_transform(similarity_matrix)#scaling the matrix 
    pca = PCA(n_components=int(len(similarity_matrix)/8))#Apply PCA which improved the results
    principalComponents = pca.fit_transform(std_wine)
    PCA_components = pd.DataFrame(principalComponents)
    clustering = KMeans(n_clusters=k).fit(PCA_components)#apply kMeans clustering
    return clustering.labels_

def calculating_accuracy(clustered_labels, images_name_list):
    '''
    Parameters
    --------------------------------
    clutered_labels : Labels of all clustered images
    image_name_list : List containing the name of all image passed

    Returns
    -------------------------------
    Returns the performance metric

    '''
    images_name = np.array([str(i).split('/')[-1].split('_')[0] for i in images_name_list])#retrieve names of files
    true_positive_labels = 0
    true_negative_labels = 0
    total_pairs = len(images_name_list)*(len(images_name_list)-1)#total pairs

    for i in range(len(clustered_labels)):
        for j in range(len(clustered_labels)):
            if i!=j:#to not compare the image with itself
                if (clustered_labels[i]==clustered_labels[j]) & (images_name[i]==images_name[j]):#If image belong to the same cluster and have same filename
                    true_positive_labels+=1
                elif (clustered_labels[i]!=clustered_labels[j]) & (images_name[i]!=images_name[j]):#if image belongs to different class and have separate file name
                    true_negative_labels+=1
                else:
                    pass
    return (true_positive_labels+true_negative_labels)/total_pairs#Metric

if __name__ == "__main__":
    # load an image
    cluster = int(sys.argv[1])
    arg = sys.argv[2]
    outpath = sys.argv[-1]
    images_name_list = []
    image_path = str(arg.split('/')[0]+str('/'))
    dirs = os.listdir(image_path)
    for files in dirs:
        images_name_list.append(str(image_path)+str(files))#Read all images given in the command
    clustered_labels = featureMatching(images_name_list,cluster)#Mind features and get cluster names
    clustered_labels_unique = np.unique(clustered_labels)#finding unique clutered names

    performance = calculating_accuracy(clustered_labels,images_name_list)#Calculating accuracy
    here = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(here,outpath)
    f = open(filepath,"w")
    for cluster in clustered_labels_unique:
        images_in_cluster = [images_name_list[i].split('/')[1] for i in range(len(images_name_list)) if clustered_labels[i]==cluster]
        for i in images_in_cluster:
            f.write(str(i)+" ")
        f.write("\n")
    f.close()
    
