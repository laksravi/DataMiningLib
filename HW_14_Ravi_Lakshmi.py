
from sklearn.decomposition import PCA
import numpy as np
import csv
from matplotlib import pyplot as plt

def plot_variables(variables):
    '''
    plot the variance of the variables
    :param variables:
    :return:
    '''
    a = variables.shape[0]
    plt.plot(np.arange(0, a).tolist(), variables)
    plt.plot(np.arange(0, a).tolist(), variables, 'bx')
    plt.xlabel("Attribute -#")
    plt.ylabel("Cumulative variance till Attribute- # ")
    plt.show()

def plot_two_variables(var_1, var2):
    '''
    plot the values of two attributes one on the other - to visualize the clusters
    :param var_1:
    :param var2:
    :return:
    '''
    plt.plot(var_1, var2, 'b.')
    plt.xlabel("First Attribute")
    plt.ylabel("Second Attribute")

    plt.show()

def reduce_matrix(dataset):
    pca = PCA(n_components=dataset.shape[1])
    pca.fit(dataset)
    cumulative_variance = np.cumsum(pca.explained_variance_)
    plot_variables(cumulative_variance)
    print(pca.explained_variance_ratio_)
    print("the eigen vectors are", (pca.components_))
    return pca

def project_on_2_dimensions(dataset, pca):
    new_dataset =pca.transform(dataset)
    plot_two_variables(new_dataset[:, 0], new_dataset[:,1])



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
        list_attributes =[line['Milk'], line['PetFood'], line['Veggies'], line['Cereal'], line['Nuts'], line['Rice'], line['Meat'], line['Eggs'], line['Yogurt'], line['Chips'], line['Beer'], line['Fruit']]
        dataset.append(list_attributes)
        datasetIds.append(line['ID'])

    dataset=np.array(dataset).astype(dtype=int)
    training_file_handler.close()
    return dataset


dataset =getFileData("C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Homework\A11\\HW_21_SHOPPING_CART_v1400.csv")
pca=reduce_matrix(dataset)
project_on_2_dimensions(dataset, pca)