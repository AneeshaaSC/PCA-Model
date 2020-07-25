from ClassificationModels import optimization as opt
from ClassificationModels import classification_model as cm
from DataPreprocess import read_data as rd
from DataPreprocess import preprocessing
from FeatureSelection import dimensionality_reduction as dim_red
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from SelectModel import select_model
from sklearn.metrics import accuracy_score, confusion_matrix


class train_model:
    
    print("Training Model")
    
    def train(model, transform_data, target_data, predictors,
              scaler_list, count, split_size, max_features,
              cv_number, num_jobs):
        
        feature_dict = []
        final_model_list = []
        X_test_list = []
        y_test_list = []
        
        for i in scaler_list:
            
            prep_data = preprocessing(i, transform_data).prep_scaler()
            features, dic_score_all, variance, components_,pca_data = dim_red.pca(prep_data, predictors)
            data_dict = {predictors[i]: prep_data.T[i] for i in range(count)}
            features_dict = {key: value for (key, value) in data_dict.items() 
                             if key in features}
            print("Number of features selected using {}:".format(i),
                  len(features_dict.keys()))
            feature_dict.append(features_dict)
            data = list(features_dict.values())
            data_array = np.array(data)
            data_array = data_array.T
            X_train, X_test, y_train, y_test = rd.data_split(data_array,
                                                             target_data,
                                                             split_size)
            train_model = cm(model, X_train, X_test, y_train, 
                             y_test).best_model()
            train_params = opt.hyperparamters(model, max_features)
            final_model = opt.hypertune_model(train_model, train_params,
                                              cv_number, num_jobs,
                                              X_train, y_train)
            final_model_list.append(final_model)
            X_test_list.append(X_test)
            y_test_list.append(y_test)
        
        return final_model_list, X_test_list, y_test_list, dic_score_all