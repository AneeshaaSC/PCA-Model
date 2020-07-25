from ClassificationModels import optimization as opt
from ClassificationModels import classification_model as cm
from DataPreprocess import read_data as rd
from DataPreprocess import preprocessing
from FeatureSelection import dimensionality_reduction as dim_red
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV


class select_model:
    
    def best_model(transform_data, target_data, predictors,
                   model_list, count, split_size):
        
        print("-----------------------------------")
        print("Selecting the best model")
        
        feature_dict = []
        best_model = dict()
        
        for j in model_list:
            
            data_dict = {predictors[i]: transform_data.T[i]
                         for i in range(count)}
            data = list(data_dict.values())
            data_array = np.array(data)
            data_array = data_array.T
            X_train, X_test, y_train, y_test = rd.data_split(data_array,
                                                             target_data,
                                                             split_size)
            base_accuracy = cm(j, X_train, X_test, y_train,
                               y_test).base_model()
            best_model[j] = np.round(base_accuracy, 3)
        
        best_model_dict = {key: value for (key, value) in best_model.items()
                           if value == max(best_model.values())}
        bestmodel = list(best_model_dict.keys())[0]
        
        return bestmodel, best_model


