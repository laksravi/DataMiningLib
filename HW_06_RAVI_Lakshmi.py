import csv
import numpy as np
from math import log2
from matplotlib import pyplot as plt
import sys

class DecisionNode:
    '''
    One node in the decision tree.
    All nodes in decision tree (except leaf nodes) are instances of this class
    '''
    def __init__(self, index, threshold):
        '''
        specifications of the "IF_CONDITION"
        :param index: the index of the attribute vector to look at
        :param threshold: value to compare and make decision in IF_CONDITION
        left : if the current attribute value is less than
        '''
        self.attribute_index = index
        self.threshold = threshold
        self.left = None
        self.right = None

    def setLeft(self, decision_node):
        # the pointer to decision node which will be called when the value is less than the threshold set
        self.left = decision_node

    def setRight(self, decision_node):
        # the pointer to decision node which will be called when the value is greater or equal than the threshold set
        self.right = decision_node



def calculate_entropy(categoryA, categoryB, total):
    '''
    Calculates entropy of two classes
    :param categoryA: count of category A
    :param categoryB: count of category B
    :param total: sum of (A and B)
    :return: entropy value for the distribution (A, B)
    '''
    if total == 0:
        raise
    P_A = categoryA/total
    log_p_A =0
    if P_A != 0.0:
        log_p_A = log2(P_A)
    P_B = categoryB/total
    log_p_B =0
    if P_B != 0.0:
        log_p_B = log2(P_B)
    entropy = P_A *log_p_A + P_B * log_p_B
    return entropy


def getBestThreshold(attribute, output):
    '''
    Pass an attribute, for every possible threshold that could be set for that attribute, it calculates the best entropy
    :param attribute: the feature vector to evaluate
    :param output: the corresponding output vector
    :return:
    '''
    #Intialize the threshold for the attribute & gain_ratio
    threshold=-1
    best_gain_ratio=-1
    indices = attribute.argsort()
    sorted_attribues = attribute[indices]
    sorted_output = output[indices]
    total_rows = sorted_attribues.size

    #Identify the potential threshold by looking at every mid-point of the dataset
    for i in range(1,total_rows):
        potential_threshold = (sorted_attribues[i] + sorted_attribues[i-1])/2
        #at this threshold find the attributes less than this value
        #obtain the count of values lesser than and equal to the threshold
                    # &  greater than the threshold
        less_than_threshold = sorted_attribues <= potential_threshold
        greater_than_threshold = sorted_attribues > potential_threshold
        count_less = less_than_threshold.sum()
        count_greater= greater_than_threshold.sum()
        if count_less == 0 or count_greater==0:
            continue
        count_0 = (output ==0 ).sum()
        count_1 = (output ==1 ).sum()
        entropy_parent = calculate_entropy(count_0, count_1, count_0+count_1)
        '''
        different output values
        '''
        #count of values less than  (or greater) threshold and belonging to class 0 (or 1)
        less_output_0= (sorted_output[less_than_threshold] ==0).sum()
        less_output_1 = (sorted_output[less_than_threshold] == 1).sum()
        great_output_0 = (sorted_output[greater_than_threshold] == 0).sum()
        great_output_1 = (sorted_output[greater_than_threshold] == 1).sum()

        #compute both the entropies, the lesser values and greater values
        #given all the attribute values are less than the given threshold, what is the entropy
        entropy_less = calculate_entropy(less_output_0, less_output_1, count_less)

        #given all the attribute values are greater than the given threshold, what is the entropy
        entropy_greater = calculate_entropy(great_output_0, great_output_1, count_greater)

        #obtain the weighed entropy
        mixed_entropy = (count_less * entropy_less + count_greater * entropy_greater)/total_rows
        #information gained from the split
        Gain_split = entropy_parent - mixed_entropy
        gain_ratio = Gain_split /entropy_parent
        if best_gain_ratio <= gain_ratio:
            best_gain_ratio = gain_ratio
            threshold = potential_threshold
    return threshold, best_gain_ratio


def visualize(dataset, output):
    #pass the input dataset, generates
    #  the matplotlib plot to visualize the classes based on two of attributes
    for i,ds in enumerate(dataset):
        X=ds[1]
        Y=ds[2]
        if output[i] == 0:
            plt.plot(X, Y, 'ro')
        else:
            plt.plot(X,Y, 'go')
    plt.xlabel("Attr2")
    plt.ylabel("Attr3")
    plt.show()

    for i,ds in enumerate(dataset):
        X=ds[0]
        Y=ds[1]
        if output[i] == 0:
            plt.plot(X, Y, 'ro')
        else:
            plt.plot(X,Y, 'go')
    plt.xlabel("Attr1")
    plt.ylabel("Attr2")
    plt.show()
    for i,ds in enumerate(dataset):
        X=ds[2]
        Y=ds[3]
        if output[i] == 0:
            plt.plot(X, Y, 'ro')
        else:
            plt.plot(X,Y, 'go')
    plt.xlabel("Attr3")
    plt.ylabel("Attr4")
    plt.show()


def get_decision_tree_root(dataset, output):
    '''
    Get the root of the decision tree to classify the given dataset
    :param dataset: feature vector for N input instances
    :param output: labels for the N input instances
    :return:
    Attribute best-ness is  the measure of gain ratio
    '''
    best_gain_ratio = -1
    best_threshold = -1
    best_attribute = -1
    output_0 = (output == 0).sum()
    output_1 = (output == 1).sum()
    #return the class value
    if output_0 == 0:
        return 1
    if output_1 == 0:
        return 0
    '''
    Calculate the gain ratio for every attribute
    '''
    for i in range(4):
        current_Attribute = dataset[:,i]
        #get the best possible entropy for the current attribute
        threshold, gain_ratio = getBestThreshold(current_Attribute, output)
        if best_gain_ratio < gain_ratio:
            best_gain_ratio = gain_ratio
            best_attribute = i
            best_threshold = threshold

    root = DecisionNode(best_attribute, best_threshold)
    #all the rows less than the specified threshold
    left_indices = dataset[:,best_attribute] <= best_threshold
    right_indices = dataset[:,best_attribute] > best_threshold
    #segregate the dataset with the corresponding attribute lesser than threshold
    # & greater than threshold
    lesser_dataset = dataset[left_indices,:]
    lesser_output = output[left_indices]
    #classify the left and right chunks of data recursively
    left_node = get_decision_tree_root(lesser_dataset, lesser_output)
    root.setLeft(left_node)
    greater_dataset = dataset[right_indices,:]
    greater_output = output[right_indices]
    right_node = get_decision_tree_root(greater_dataset, greater_output)
    root.setRight(right_node)
    return root



def getFileData(Input_file_Name):
    '''
    Reads the input csv file and
    :param Input_file_Name:
    :return:
    '''
    #get the
    training_file_handler = open(Input_file_Name, mode='r')
    training_file = csv.DictReader(training_file_handler, )
    #since we know that there are 5 attributes
    dataset=[]
    output=[]
    for line in training_file:
        list_attributes =[line['Attr1'], line['Attr2'], line['Attr3'], line['Attr3']]
        dataset.append(list_attributes)
        output.append(line['Class'])

    dataset=np.array(dataset).astype(dtype=float)
    output = np.array(output).astype(dtype=int)

    #printAttributes(dataset)
    training_file_handler.close()
    return dataset, output



def printDecisionTree(root, prefix, file_handler):
    '''
    goes through the node in decision tree and prints the condition & recursively calls the child node
    lesser and greater than conditions
    :param root: the first condition node of decision tree
    :param prefix: the tab appendation
    :param file_handler: the file to write the decisions (conditions) to
    :prints the data(conditions) to the given python file handler:
    '''
    if not isinstance(root, DecisionNode):
        file_handler.write(prefix+"print(\""+str(root)+"\")"+"\n")
        return

    condition = "if features["+str(root.attribute_index)+"]" +" <" + str(root.threshold)+":\n"
    file_handler.write(prefix+condition)
    printDecisionTree(root.left, prefix+"\t", file_handler)
    file_handler.write(prefix+"else:\n")
    printDecisionTree(root.right, prefix+"\t", file_handler)

def writePrologueTestFunction(classifier_file_name, root):
    #read the given csv file and get the features
    classifier_handler = open(classifier_file_name, mode='a+')
    classifier_handler.write("import sys\nimport csv\nimport numpy as np\n\n")

    classifier_handler.write("# This file acts as a decision tree classifier\n\n\n")
    classifier_handler.write("testing_file_name = sys.argv[1]\n")
    classifier_handler.write("testing_file_handler = open(testing_file_name, mode='r')\n")
    classifier_handler.write("testing_file_csv = csv.DictReader(testing_file_handler, )\n")
    classifier_handler.write("for line in testing_file_csv:\n\tfeatures =[line['Attr1'], line['Attr2'], line['Attr3'], line['Attr3']]\n")
    classifier_handler.write("\tfeatures = np.array(features).astype(dtype=float)\n")
    printDecisionTree(root, "\t", classifier_handler)

training_file_path = sys.argv[1]
attributes, output =getFileData(training_file_path)
root = get_decision_tree_root(attributes, output)
writePrologueTestFunction("HW_06_RAVI_LAKSHMI_Classifier.py",root)
#visualize(attributes, output)