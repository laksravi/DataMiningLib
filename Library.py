from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import naive_bayes
def measure_error(expected, predicted):
    error = 0
    for e, p in zip(expected, predicted):
        err = abs((e-p))
        error +=err
    return error

def regressionRandomForest(training_set, training_output, testing_Set, testing_output, tree_count):
    regressor = RandomForestRegressor(n_estimators=tree_count);
    regressor.fit(training_set, training_output)
    predicted = regressor.predict(testing_Set)
    SSE=measure_error(testing_output, predicted)
    return SSE


def regressionDecisionTree(training_set, training_output, testing_Set, testing_output):
    regressor = DecisionTreeRegressor();
    regressor.fit(training_set, training_output)
    predicted = regressor.predict(testing_Set)
    SSE=measure_error(testing_output, predicted)
    return SSE


def agglometriveClustering(training_Set, clusters):
    ward = AgglomerativeClustering(n_clusters=clusters).fit(training_Set)
    label = ward.labels_
    return label


def runNaiveBayes(training_set, training_output, testing_set):
    Nb = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=False, class_prior=None)
    Nb.fit(training_set, training_output)
    return Nb.predict(testing_set)


def getLabelsDecisionTree(training_set, labels,testing_set):
    dtree = DecisionTreeClassifier();
    dtree.fit(training_set, labels)
    return dtree.predict(testing_set)

def getLabelsLinearClassifier(training_set, labels, testing_set):
    linRegr = LinearRegression()
    linRegr.fit(training_set, labels)
    return linRegr

def getANNClassification(trainingset, output, testing_set, i):
    mlp = MLPClassifier(hidden_layer_sizes=i)
    mlp.fit(trainingset, output)
    return mlp.predict(testing_set)