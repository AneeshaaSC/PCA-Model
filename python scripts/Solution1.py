import csv
import numpy as np
from csv import reader, writer
from numpy import genfromtxt, savetxt
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy.core.defchararray as np_f
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.simplefilter(action='ignore')


with open('dataset_challenge_one.tsv') as file:
        list_rows = [x.replace('\n', '').split('\t') for x in file]
        data = np.array(list_rows)
class_0 = []
for i in data[1:]:
    if (i[-1] == '0'):
        class_0.append(i)
class_1 = []
for i in data[1:]:
    if (i[-1] == '1'):
        class_1.append(i)


def stats(data):
        dict_of_lists = defaultdict(list)
        data = np.array(data)
        data_tran = data.T
        for i in data_tran:
            dict_of_lists[i[0]].append(i[1:])
            dict_of_lists[i[0]] = np_f.replace(dict_of_lists[i[0]], 'NA', '0')
            for i in dict_of_lists.keys():
                dict_of_lists[i] = dict_of_lists[i].astype(float)
            dict_of_lists[i] = dict_of_lists[i][0]
            list_of_dicts = dict_of_lists
        max_list = []
        min_list = []
        mean_list = []
        median_list = []
        var = []
        
        for i in list_of_dicts.keys():
            max_list.append(max(list_of_dicts[i]))
            min_list.append(min(list_of_dicts[i]))
            mean_list.append(np.mean(list_of_dicts[i]))
            median_list.append(np.median(list_of_dicts[i]))
            var.append(i)
           
        return max_list, min_list, mean_list, median_list, var


max_list_1, min_list_1, mean_list_1, median_list_1, var_1 = stats(class_1)
plt.figure(figsize=(10, 5))
plt.title("Mean, Median, Maximum, Minimum values - Class 1")
plt.hist(max_list_1, label="Maximum Values")
plt.hist(min_list_1, label="Minimum Values", alpha=0.7)
plt.hist(median_list_1, label="Median Values", alpha=0.7)
plt.hist(mean_list_1, label="Mean Values", alpha=0.7)
plt.ylabel("Count of variables")
plt.xticks(rotation='vertical')
plt.legend()
plt.show()

max_list, min_list, mean_list, median_list, var = stats(class_0)
plt.figure(figsize=(10, 5))
plt.title("Mean, Median, Maximum, Minimum values - Class 1")
plt.hist(max_list, label="Maximum Values")
plt.hist(min_list, label="Minimum Values", alpha=0.7)
plt.hist(median_list, label="Median Values", alpha=0.7)
plt.hist(mean_list, label="Mean Values", alpha=0.7)
plt.ylabel("Count of variables")
plt.xticks(rotation='vertical')
plt.legend()
plt.show()


class read_data:

    def tsv_read(filename):
        print("Reading the data")
        with open(filename) as file:
            list_rows = [x.replace('\n', '').split('\t') for x in file]
            data = np.array(list_rows)
        return data

    def handle_missing_data(data):
        print("Replacing irrelevant values")
        data = np_f.replace(data, 'NA', '0')
        data = data.astype(float)
        return data

    def data_transformation(data):
        print ("Transforming data")
        predictors = data[0][:-1]
        target = data[0][-1]
        data = data[1:]
        target_data = data.T[-1]
        predictors_data = data.T[:-1]
        predictors_data = predictors_data.T
        return predictors_data, predictors, target_data, target
    
    def data_split(predictors, target, split_size):
        print ("Splitting data into training and test sets")
        X = predictors
        y = target
        y = y.astype(int)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                        test_size=split_size,
                                                        random_state=78)
        
        return Xtrain, Xtest, ytrain, ytest


data = read_data.tsv_read("dataset_challenge_one.tsv")
predictors_data, predictors, target_data, target = read_data.data_transformation(data)
transform_data = read_data.handle_missing_data(predictors_data)
count = predictors_data.shape[1]
split_size = 0.3
data_dict = {predictors[i]: transform_data.T[i] for i in range(2)}
data = list(data_dict.values())
data_array = np.array(data)
data_array = data_array.T
X_train, X_test, y_train, y_test = read_data.data_split(data_array, 
                                                        target_data,
                                                        split_size)

X = transform_data
clf = IsolationForest()
clf.fit(X)
pred = clf.predict(X)

p = list(pred)
normal = [i for i, x in enumerate(p) if x == 1]
outliers = [i for i, x in enumerate(p) if x == -1]
t = list(target_data)
class0 = [i for i, x in enumerate(t) if x == '0']
class1 = [i for i, x in enumerate(t) if x == '1']
class0_normal = len([i for i in class0 if i in normal])
class0_outlier = len([i for i in class0 if i in outliers])
class1_normal = len([i for i in class1 if i in normal])
class1_outlier = len([i for i in class1 if i in outliers])

plt.bar('Class 0', class0_normal, label="Class 0 - Normal")
plt.bar('Class 0', class0_outlier, label='Class 0 - Outlier')
plt.bar('Class 1', class1_normal, label="Class 1 - Normal")
plt.bar('Class 1', class1_outlier, label='Class 1 - Outlier')
plt.title("Outliers in Both classes")
plt.ylabel("Count")
plt.legend()

