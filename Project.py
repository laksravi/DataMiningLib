import csv
from matplotlib import pyplot as plt
import numpy as np
import sys
import random
import Library as myLib
import mpl_toolkits.mplot3d.axes3d as p3

def plot_attribute(att, att_name):
    '''
    Plots the histogram of the attribute and labels with attribute name
    :param att:
    :param att_name:
    :return:
    '''
    plt.hist(att, 7)
    plt.xlabel(att_name)
    plt.ylabel("Count of instances")
    plot_path ="C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Proj\\Final_Report\\Attributes\\"+att_name+".jpeg"
    plt.savefig(plot_path)
    plt.clf()

def plot_1d_cluster(plot_variable, label, f_name):
    shape = plot_variable.shape
    random_jitter = np.random.rand(shape[0])
    colors =['g.', 'b.', 'y.', 'k.']
    for l in np.unique(label):
        plt.plot(plot_variable[label==l], random_jitter[label==l], colors[l])
    plot_path = "C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Proj\\Final_Report\\Clusters\\" + f_name + ".jpeg"
    plt.savefig(plot_path)



def plot_3DCluster(X, label, f_name, shape='o'):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    #ax.view_init(7, -80)
    plot_path ="C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Proj\\Final_Report\\Classify_After_Cluster\\"+f_name+".jpeg"
    for l in np.unique(label):
        ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],shape, color=plt.cm.jet(np.float(l) / np.max(label + 1)))

    plt.savefig(plot_path)


def randomize_data(my_dataset_file, randomize_file_name):
    with open(my_dataset_file) as f:
        l = list(csv.reader(f))
    f.close()

    random.shuffle(l)
    with open(randomize_file_name, 'a+') as f:
        for row in l:
            for elem in row:
                f.write(str(elem)+",")
            f.write("\n")
    f.close()


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
    header=['Catch_Rate', 'Speed', 'Weight_kg',
            'Generation', 'Total', 'HP',
            'Attack', 'Sp_Def','Sp_Atk',
            'Defense', 'Height_m', 'Weight_kg',
            'hasMegaEvolution', 'isLegendary']
    for line in training_file:
        list_attributes =[line['Catch_Rate'],line['Speed'], line['Weight_kg'],
                          line['Generation'], line['Total'], line['HP'],
                          line['Attack'], line['Sp_Def'],line['Sp_Atk'],
                          line['Defense'], line['Height_m'], line['Weight_kg']]
        if line['hasMegaEvolution'] == 'False':
            list_attributes.append(0.0)
        else:
            list_attributes.append(1.0)

        if line['isLegendary'] == 'FALSE':
            list_attributes.append(0.0)
        elif line['isLegendary'] == 'TRUE':
            list_attributes.append(1.0)
        else:
            raise
        dataset.append(list_attributes)
    dataset = np.array(dataset, dtype=float)

    return dataset, header


def find_pearson_correlation(attribute, output):
    '''
    returns the pearsom co-efficient between two variables attribute and vector
    :param attribute:
    :param output:
    :return: Pearson_coeff value
    '''
    return np.corrcoef(attribute, output)[0][1]


def analyze_cross_correlation(dataset, header):
    '''
    analyzes the cross correlation between every attribute and output
    :param dataset:
    :param header:
    :return:
    '''
    cols = dataset.shape[1]-2
    for i in range(1,cols):
        print(header[i], ",", find_pearson_correlation(dataset[:, i], dataset[:,0]) )
    is_legend = cols-2

    print("Correlation with legandary")
    for i in range(1, is_legend-1):
        print(header[i], ",", find_pearson_correlation(dataset[:,i], dataset[:, is_legend]))


def plot_analysis(feature_set, labels):
    min_x, max_x = int(np.min(feature_set[:, 0])), int(np.max(feature_set[:,0]))
    min_y, max_y = int(np.min(feature_set[:, 1])), int(np.max(feature_set[:,1]))
    min_z, max_z = int(np.min(feature_set[:, 2])), int(np.max(feature_set[:,2]))
    dataset=[]
    print(min_x, max_x)
    print(min_y, max_y)
    print(min_z, max_z)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            for k in range(min_z, max_z):
                dataset.append([i,j,k])

    print("Created test dataset")
    test_dataset = np.array(dataset, dtype=float)
    test_labels = myLib.getLabelsDecisionTree(feature_set, labels, test_dataset)
    plot_3DCluster(feature_set, labels, "Training", 'o')
    plot_3DCluster(test_dataset, test_labels, "testing", '.')







dataset_file = sys.argv[1]
dataset, header =getFileData(dataset_file)
#analyze_cross_correlation(dataset, header)
cols = len(header)
labels = myLib.agglometriveClustering(dataset[:,6:9], clusters=i)
plot_3DCluster(dataset[:,6:9], labels, "Using_Attributes"+str(i))

labels = myLib.agglometriveClustering(dataset[:, 0:1], clusters=2)
plot_1d_cluster(dataset[:, 0:1], labels, "Catch_Rate_Itself" + str(2))
plot_analysis(dataset[:, 6:9], labels)
plot_3DCluster(dataset[:, 6:9], labels, "Using_CatchRate" + str(2))



'''for i in range(cols):
    plot_attribute(dataset[:,i], header[i])
'''

'''plot_attribute(dataset[:,1], "Speed of Pokemon")
plot_attribute(dataset[:,2], "Weight of pokemon")
plot_attribute(dataset[:,3], "Catch Rate of pokemon")
plot_attribute(dataset[:,4], "Legendary Pokemon (1-True, 0 -False)")'''

random_file = "C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Project\\pokemon\\random.csv"

#randomize_data(sys.argv[1], random_file)
dataset, header= getFileData(random_file)
training_dataset = dataset[:600,:]
testing_dataset = dataset[601:,:]
training_features = training_dataset[:,1:12]
training_output = training_dataset[:,13]
testing_features = testing_dataset[:,1:12]
testing_output = testing_dataset[:,13]

predicted =myLib.runNaiveBayes(training_features, training_output, testing_features)
print(np.sum(  np.logical_and(predicted != testing_output, predicted == 0.0)), "Misses for legendary")
print(np.sum(  np.logical_and(predicted != testing_output, predicted == 1.0)), "false alarms for legendary")
print(np.sum(predicted == testing_output) , "is the count correct", predicted.shape)



for i in range(1, 15):
    predicted =  myLib.getANNClassification(training_features, training_output, testing_features, i)

    predicted = np.array(predicted)
    print(np.sum(  np.logical_and(predicted != testing_output, predicted == 0.0)), "false alarms for legendary")
    print(np.sum(predicted == testing_output) , "is the count correct", predicted.shape, "for ", i , " hidden layes")
error_rate = []
for i in range(1, 50):
    error_rate.append(myLib.regressionRandomForest(training_features, training_output, testing_features, testing_output, i))


plt.plot(np.arange(1,50).tolist(),error_rate)
plt.xlabel("Number of Trees - Random Forest")
plt.ylabel("Sum of Squared Errors")
plt.show()

error_decision_tree =(myLib.regressionDecisionTree(training_features, training_output, testing_features, testing_output))
print("Errors in Decision Tree", error_decision_tree)