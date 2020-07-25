import csv
import numpy as np
from csv import reader, writer
from numpy import genfromtxt, savetxt
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy.core.defchararray as np_f
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, RobustScaler, \
    StandardScaler

class read_data:

    def tsv_read(filename):
        print ('Reading the data')
        with open(filename) as file:
            list_rows = [x.replace('\n', '').split('\t') for x in file]
            data = np.array(list_rows)
        return data

    def handle_missing_data(data):
        print ('Replacing irrelevant values')
        data = np_f.replace(data, 'NA', '0')
        data = data.astype(float)
        return data

    def data_transformation(data):
        print ('Transforming data')
        predictors = (data[0])[:-1]
        target = data[0][-1]
        data = data[1:]
        target_data = data.T[-1]
        predictors_data = data.T[:-1]
        predictors_data = predictors_data.T

        return (predictors_data, predictors, target_data, target)

    def data_split(predictors, target, split_size):
        print ('Splitting data into training and test sets')
        X = predictors
        y = target
        y = y.astype(int)
        (Xtrain, Xtest, ytrain, ytest) = train_test_split(X, y,
                                                          test_size=split_size,
                                                          random_state=78)

        return (Xtrain, Xtest, ytrain, ytest)


class preprocessing:

    def __init__(self, scaler, data):
        self.scaler = scaler
        self.data = data

    def prep_scaler(self):
        print ('--------------------------------')
        if self.scaler == 'robust_scaler':
            print ('Scaling the data using Robust Scaler')
            std = RobustScaler()
            self.data = std.fit_transform(self.data)
            return self.data
        elif self.scaler == 'standard_scaler':
            print ('Scaling the data using Standard Scaler')
            std = StandardScaler()
            self.data = std.fit_transform(self.data)
            return self.data
        else:
            print ('Scaling the data using Min Max Scaler')
            std = MinMaxScaler()
            self.data = std.fit_transform(self.data)
            return self.data


