from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np
import csv
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def plot_distances(distances, threhold, f_name):
    plot_path ="C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A10\\Plots\\"+f_name+".jpeg"
    len = distances.shape[0]+1
    plt.plot(np.arange(1, len).tolist(), distances)
    plt.plot([1,len], [threhold, threhold])
    #plt.plot(np.arange(1, len).tolist(), distances, 'rx')
    plt.savefig(plot_path)
    plt.clf()


def segregate_Points(points, k):
    '''
    returns a label for points that are
    :param points:
    :return:
    '''
    len_points = points.shape[0]
    labels=np.zeros(len_points)
    dist_between = cdist(points, points)
    dist_between=np.sort(dist_between, axis=1)
    #take the distance to the 4th nearst neighbor
    distance_to_k = dist_between[:, k]
    sorted_distance_to_k = np.sort(distance_to_k)
    #compute ths slope at every point distance
    slopes = np.gradient(sorted_distance_to_k)
    difference = np.diff(slopes)
    inflection_point = np.argmax(slopes)
    threshold = sorted_distance_to_k[inflection_point-3]
    plot_distances(sorted_distance_to_k, threshold, "distance")
    plot_distances(slopes, slopes[inflection_point-2], "slopes")
    #plot_distances(difference, difference[inflection_point-1], "difference")

    noise_points = distance_to_k > threshold
    labels[noise_points]=2

    core_points_indices = np.sum((dist_between < threshold), axis=1) >= 50
    labels[core_points_indices] = 1

    return labels, threshold


def plot_3DCluster(X, label, f_name):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plot_path ="C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A10\\Plots\\"+f_name+".jpeg"
    ax.plot3D(X[label == 0, 0], X[label == 0, 1], X[label == 0, 2],'.', color=plt.cm.jet(np.float(1) / np.max(label + 1)))
    ax.plot3D(X[label == 1, 0], X[label == 1, 1], X[label == 1, 2],'x', color=plt.cm.jet(np.float(2) / np.max(label + 1)))
    ax.plot3D(X[label == 2, 0], X[label == 2, 1], X[label == 2, 2],'o', color=plt.cm.jet(np.float(3) / np.max(label + 1)))
    plt.savefig(plot_path)
    plt.clf()


def plot_3DCluster_Lables(X, label, f_name, shape='o'):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    plot_path ="C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A10\\Plots\\"+f_name+".jpeg"
    for l in np.unique(label):
        if l != 0:
            ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],shape, color=plt.cm.jet(np.float(l) / np.max(label + 1)))
    plt.savefig(plot_path)
    plt.clf()



def label_clusters(dataset, labels, eps_distance,number_cluster):
    non_noise = (labels == 2)
    #all are lablled as noise
    cluster_labels = np.zeros(labels.shape[0])
    #segregate the non noise data points
    non_noise_instances = dataset[~non_noise]
    #segregate the core data points
    core_data_indices = (labels==1)
    core_data_points = dataset[core_data_indices]
    #compute the distance from every core point to every non_noise point (which also includes the core points)
    dist_core_other = cdist(core_data_points, non_noise_instances)
    merge_points = dist_core_other < eps_distance
    #every core point
    i=1
    for merge_coreset in merge_points:
        cluster_labels[merge_coreset] = i
        i+=1
    return cluster_labels



def cluster_data(dataset):
    '''
    cluster the dataset for different k values
    :param dataset:
    :return:
    '''
    for k in range(2, 10):
        labels, eps_distance = segregate_Points(dataset, k-1)
        cluster_labels = label_clusters(dataset, labels, eps_distance, k)
        plot_3DCluster(dataset, labels, "PlotNoise"+str(k))
        plot_3DCluster_Lables(dataset, cluster_labels, "ClusterLabels"+str(k))





def getFileData(Input_file_Name):
    '''
    Reads the input csv file and
    :param Input_file_Name:
    :return:
    '''
    #get the
    training_file_handler = open(Input_file_Name, mode='r')
    training_file = csv.reader(training_file_handler)
    #since we know that there are 3 attributes
    dataset=[]
    for line in training_file:
        dataset.append(line)
    dataset=np.array(dataset, dtype=float)
    training_file_handler.close()
    return dataset


dataset = getFileData("C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A10\\HW_08_DBScan_Data_NOISY_v300.csv")
cluster_data(dataset)
