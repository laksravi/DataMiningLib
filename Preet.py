import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import product

def plot_matrix(a, file_name):
    plt.clf()
    plt.imshow(a, cmap='jet')
    plt.colorbar()
    plt.xlabel("Hours in Day 0 to 24")
    plt.ylabel("Hours in Day 0 to 24")
    plt.title("Correlation between every hour")
    plt.savefig(file_name)


def find_pearson_correlation(attribute, output):
    '''
    returns the pearsom co-efficient between two variables attribute and vector
    :param attribute:
    :param output:
    :return: Pearson_coeff value
    '''
    return np.corrcoef(attribute, output)[0][1]

def findCrossCorrelation(attributes):
    cols = attributes.shape[1]
    column_major=attributes.transpose()
    cross_correlation = np.zeros((cols, cols))
    attribute_indexing = list((i, j) for ((i, _), (j, _)) in product(enumerate(column_major), repeat=2))
    attribute_combinations = product(column_major, repeat=2)

    for index, comb in zip(attribute_indexing,attribute_combinations):
        hour_1_data, hour_2_data = comb
        i, j = index
        cross_correlation[i][j] = find_pearson_correlation(hour_1_data, hour_2_data)
    return cross_correlation




def represent_DataFormat(file_name, filterDay=None, filterWeekends = None):
    entire_dataset, dayMapper, dateMapper = getFileData(file_name)

    if filterDay is None and filterWeekends is None:
        dataset = entire_dataset

    if filterDay != None:
        dayKey = dayMapper[filterDay]
        dataset = entire_dataset[entire_dataset[:,4] == dayKey]


    if filterWeekends == True:
        dataset = entire_dataset[entire_dataset[:,5] == 1]
    elif filterWeekends == False:
        dataset = entire_dataset[entire_dataset[:,5] == 0]


    unique_dates=np.unique(dataset[:,3])
    wifis = []
    reshall=[]
    academic = []
    for date in unique_dates:
        one_date_instances = dataset[dataset[:,3]==date]
        total_new_single_date = np.zeros(24)
        wifi_new_single_date = np.zeros(24)
        academic_new_single_date = np.zeros(24)
        for date_instance in one_date_instances:
            hour_value = date_instance[6]
            total_new_single_date[hour_value-1] = date_instance[0]
            wifi_new_single_date[hour_value - 1] = date_instance[1]
            academic_new_single_date[hour_value - 1] = date_instance[2]

        wifis.append(wifi_new_single_date)
        reshall.append(total_new_single_date)
        academic.append(academic_new_single_date)
    return np.array(wifis), np.array(reshall), np.array(academic)



def getFileData(file_name):
    training_file_handler = open(file_name, mode='r')
    training_file = csv.DictReader(training_file_handler, )

    dataset = []
    dateMapper = {}
    dateCounter =0
    dayMapper = {}
    dayCounter =0
    for line in training_file:
        lineElements =[]
        lineElements.append(line['RESHALL'])
        lineElements.append(line['WIRELESS'])
        lineElements.append(line['ACADEMIC'])

        #append a numeric value for every date
        if line['DATE'] in dateMapper:
            lineElements.append(dateMapper[line['DATE']])
        else:
            dateMapper[line['DATE']] = dateCounter
            dateCounter+=1
            lineElements.append(dateMapper[line['DATE']])

        if line['Day']  in dayMapper:
            lineElements.append(dayMapper[line['Day']])
        else:
            dayMapper[line['Day']] = dayCounter
            dayCounter+=1
            lineElements.append(dayMapper[line['Day']])

        if line['Weekend'] == 'FALSE':
            lineElements.append(0)
        else:
            lineElements.append(1)

        lineElements.append(line['Hour'])

        dataset.append(lineElements)

    dataset = np.array(dataset, dtype=float)
    return dataset, dayMapper, dateMapper



def operate_on_dataset(Hour_Dataset_File, filterDay=None, filterWeekends=None, file_path=""):
    wifis, totals, academic = represent_DataFormat(Hour_Dataset_File, filterDay, filterWeekends)
    print(wifis.shape, totals.shape, academic.shape)
    cross_corelation = findCrossCorrelation(wifis)
    plot_matrix(cross_corelation, file_path+"_WIFI.png")

    cross_corelation = findCrossCorrelation(totals)
    plot_matrix(cross_corelation, file_path+"_RESHALL.png")

    cross_corelation = findCrossCorrelation(academic)
    plot_matrix(cross_corelation, file_path+"_Academic.png")

Hour_Dataset_File = "C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Random\\datatable_with_hour.csv"
plotPath = "C:\\Users\\laksh\\Documents\\Big_Data_Analytics\\Random\\"
operate_on_dataset(Hour_Dataset_File, filterWeekends=True, file_path=plotPath+"Weekend")
operate_on_dataset(Hour_Dataset_File, filterDay='Monday', file_path=plotPath+"Monday")
operate_on_dataset(Hour_Dataset_File, filterDay='Tuesday', file_path=plotPath+"Tuesday")
operate_on_dataset(Hour_Dataset_File, filterDay='Wednesday', file_path=plotPath+"Wednesday")
operate_on_dataset(Hour_Dataset_File, filterDay='Thursday', file_path=plotPath+"Thursday")
operate_on_dataset(Hour_Dataset_File, filterDay='Friday', file_path=plotPath+"Friday")
