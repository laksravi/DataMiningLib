import numpy as np
import csv, sys
from scipy.spatial.distance import pdist, cdist, squareform
import math
from matplotlib import  pyplot as plt

class Cluster:
    '''
    Cluster prototype that cluster in our system
    '''
    def __init__(self, features, ids, cluster_id):
        self.cluster_id =cluster_id
        #mean with respect to axis-0, one of the axis identifies the centroid of the data
        self.center = np.array(np.mean(features, axis=0))
        self.features = features
        self.ids = ids
        self.cluster_size = np.max(cdist(self.features, np.array([self.center])))

    def merge_cluster(self, cluster):
        '''
        merges the current cluster with another cluster
        :param cluster: to merge with
        '''
        #the features are appeneded with each other
        self.features = np.append(self.features, cluster.features, axis=0)
        self.ids = np.append(self.ids, cluster.ids, axis=0)
        #the new center is computed
        self.center = np.mean(self.features, axis=0)
        #retain the least cluster id
        self.cluster_id= min(self.cluster_id, cluster.cluster_id)
        #compute the new cluster size
        self.cluster_size = np.max(cdist(self.features, np.array([self.center])))


def formSingleCluster(clusters, stage=1):
    '''
    club the given clusters to a single cluster recursively
    :param clusters:
    :return:
    '''
    if len(clusters) == 1:
        return [clusters[0].cluster_size]

    #extract the center of mass from all the cluster
    center = []
    if len(clusters) ==3:
            print("3 Cluster - Individual Ids")
            for cluster in clusters:
                print(cluster.ids, cluster.cluster_id)

    #collect the center of the cluster
    for cluster in clusters:
        center.append(cluster.center)
    center = np.array(center)

    #compute the  distance between every cluster center to other
    center_distances = squareform(pdist(center, 'euclidean'))
    # To ignore the self-cluster distances - mark the diagonal as Not a value
    np.fill_diagonal(center_distances, math.inf)
    #find the indices of the two clusters that has the least distance between their centers
    from_cluster, to_cluster = np.unravel_index(np.argmin(center_distances), center_distances.shape)
    from_cluster_size, to_cluster_size = clusters[from_cluster].cluster_size, clusters[to_cluster].cluster_size
    #identify the least sized cluster
    least_cluster_size = [min(from_cluster_size,to_cluster_size)]
    print("Merging Clusters..", clusters[from_cluster].cluster_id," and " , clusters[to_cluster].cluster_id, " - distance is ", min(from_cluster_size, to_cluster_size))
    clusters[from_cluster].merge_cluster(clusters[to_cluster])
    clusters.pop(to_cluster)
    sub_cluster_sizes=formSingleCluster(clusters, stage+1)
    least_cluster_size+=sub_cluster_sizes
    return least_cluster_size


def executeClusteringonInputFile():
    clusters=[]
    # read the file from the argument given
    file_name = sys.args[0]
    dataset, datasetIds = getFileData("C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A7\\HW_07_SHOPPING_CART_v137.csv")
    for data, id in zip(dataset,datasetIds):
        single_id=[id]
        primitive_cluster = Cluster(data, np.array(single_id), int(id));
        clusters.append(primitive_cluster)
    least_Cluster_distance=formSingleCluster(clusters)

    plt.plot(np.arange(100).tolist(),least_Cluster_distance)
    plt.xlabel("Stages (Iterations) - Clustering")
    plt.ylabel("Smallest Cluster size")
    plt.show()
    print("Cluster sizes at every stage was ", least_Cluster_distance)


def getFileData(Input_file_Name):
    '''
    Reads the input csv file and
    :param Input_file_Name:
    :return features and their corresponding ids
    '''
    #get the
    training_file_handler = open(Input_file_Name, mode='r')
    training_file = csv.DictReader(training_file_handler, )
    #since we know that the name of the attributes, we hard code the values
    dataset=[]
    datasetIds=[]
    for line in training_file:
        list_attributes =[[line['Milk'], line['PetFood'], line['Veggies'], line['Cereal'], line['Nuts'], line['Rice'], line['Meat'], line['Eggs'], line['Yogurt'], line['Chips'], line['Beer'], line['Fruit']]]
        dataset.append(list_attributes)
        datasetIds.append(line['ID'])

    dataset=np.array(dataset).astype(dtype=int)
    datasetIds=np.array(datasetIds).astype(dtype=int)

    #printAttributes(dataset)
    training_file_handler.close()
    return dataset, datasetIds

if __name__ == "__main__":
    executeClusteringonInputFile()