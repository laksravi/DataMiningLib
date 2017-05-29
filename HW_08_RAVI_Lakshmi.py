import numpy as np
import csv
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_random_points(k, dataset):
    '''
    get K random points in space
    :param k:
    :param dataset: reference to get the range of points
    :return:
    '''
    dimensions = dataset.shape[1]
    min = np.min(dataset, axis=1)
    max = np.max(dataset, axis=1)
    centers= np.array([[0 for j in range(dimensions)] for i in range(k)])
    #for every dimension
    for i in range(dimensions):
        min_i , max_i = min[i], max[i]
        centers[:, i] = min_i + np.random.random_sample(k)*(max_i-min_i)
    return centers


def get_belonging(dataset, centers):
    '''
    for the given centers find to which center every dataset belongs to
    :param dataset:
    :param centers:
    :return: closet center point for every dataset
    '''
    p= np.argmin(cdist(dataset, centers), axis=1)
    return p

def redefine_Centers(dataset, belonging, centers):
    '''
    if every dataset belongs to a particular center, recompute the center
    :param dataset:
    :param belonging:
    :return:
    '''
    k = centers.shape[0]
    new_centers = []
    for i in range(k):
        i_cluster = (belonging==i)
        if i_cluster.sum() == 0:
            new_centers.append(centers[i,:])
        else:
            new_center=np.mean(dataset[i_cluster], axis=0)
            new_centers.append(new_center)
    new_centers = np.array(new_centers)
    #if all the 'k' centers before and after are the same
    centers_remain_same = (centers==new_centers).all()
    return new_centers, centers_remain_same


def get_SSE(dataset, centers, belonging):
    SSE=0
    for data, cluster_index in zip(dataset, belonging):
        SSE+=cdist(np.array([data]),np.array([centers[cluster_index]]))
    return SSE[0,0]


def run_k_means(dataset, k, centers=None):
    #initialize center points if no center points are known from last iteration
    if(centers is None):
        centers = initialize_random_points(k, dataset)

    belonging = get_belonging(dataset, centers)
    centers, isSameCenter = redefine_Centers(dataset, belonging, centers)
    #if center points are re-defined, then iterate again to know the data instances' clsuter belonging
    if not isSameCenter:
        return run_k_means(dataset, k, centers)
    return get_SSE(dataset, centers, belonging), belonging, centers




def getFileData(Input_file_Name):
    '''
    Reads the input csv file and
    :param Input_file_Name:
    :return:
    '''
    #get the
    training_file_handler = open(Input_file_Name, mode='r')
    training_file = csv.DictReader(training_file_handler, )
    #since we know that there are 3 attributes
    dataset=[]
    for line in training_file:
        list_attributes =[line['Attrib01'], line['Attrib02'], line['Attrib03']]
        dataset.append(list_attributes)

    #PREPROCESSING - convert values to theirr absolute values, since they are positions in space
    dataset=np.absolute(np.array(dataset).astype(dtype=float))

    #NOISE REMOVAL - points which are closer than 0.1 astronomical units are removed, since they represent stray data
    #identify points closer than 0.1 units
    close_by_points=(squareform(pdist(dataset)) < 0.1)
    #ignore the self-distances - same point distances are always zero
    np.fill_diagonal(close_by_points, False)
    #filter the closer point -indices
    indices_closer_points=(np.where(np.any(close_by_points == True, axis=1)))
    #remove the closer point indices
    np.delete(dataset, indices_closer_points)
    training_file_handler.close()
    return dataset


if __name__ == "__main__":
    dataset  = getFileData("C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A8\\HW08_KMEANS_DATA_v300.csv")
    SSE_List=[]
    best_belonging = None
    best_Centers = None
    best_SSE = 999999999
    best_k=-1
    range_of_clusters = 15
    #iterate over the clustering-sizes from 2 to 15 and find the clustering with least Sum of square errors
    for k in range(2,range_of_clusters):
        SSE, belonging, centers =run_k_means(dataset,k)
        SSE_List.append(SSE)
        if best_SSE > SSE and (best_SSE - SSE) > 30.0:
            best_SSE = SSE
            best_Centers = centers
            best_belonging = belonging
            best_k = k

    plt.plot(np.arange(2, range_of_clusters).tolist(),SSE_List, 'rx')
    plt.plot(np.arange(2, range_of_clusters).tolist(), SSE_List)
    plt.xlabel("K - Number of Clusters")
    plt.ylabel("SSE-  sum of squared errors")
    plt.show()

    #print the cluster belonging
    print("Best - k value", best_k)
    print("Instances in each cluster",np.sort(np.bincount(best_belonging)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors =['red', 'green','blue', 'yellow', 'magenta', 'black', 'gray', 'brown', 'orange', 'pink', 'cyan', 'mediumslateblue','indigo','olive','salmon']

    #plot the data in the cluster
    for data, belong in zip(dataset, best_belonging):
        xs, ys, zs = data
        ax.scatter(xs, ys, zs, c=colors[belong])
    #plot the cluster centers
    for i, data in enumerate(best_Centers):
        xc, yc, zc = data
        ax.scatter(xc, yc, zc, c=colors[i], s=200, zdir='z', depthshade=True)
    ax.set_xlabel('X Dimension')
    ax.set_ylabel('Y Dimension')
    ax.set_zlabel('Z Dimension')

    plt.show()



