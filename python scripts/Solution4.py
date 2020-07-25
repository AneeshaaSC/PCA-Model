from ClassificationModels import optimization as opt
from ClassificationModels import classification_model as cm
from DataPreprocess import read_data as rd
from DataPreprocess import preprocessing
from FeatureSelection import dimensionality_reduction as dim_red
from SelectModel import select_model as sm
from Train import train_model as tr
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.simplefilter(action='ignore')


class run_model:

    def run(filename):

        scaler_list = ['robust_scaler', 'standard_scaler',
                       'min_max_scaler']
        model_list = ['LogisticRegression', 'XGBoost', 'RandomForest',
                      'DecisionTree']

        data = rd.tsv_read(filename)
        (predictors_data, predictors, target_data, target) = \
            rd.data_transformation(data)
        transform_data = rd.handle_missing_data(predictors_data)

        count = predictors_data.shape[1]
        split_size = 0.3
        cv_number = 5
        num_jobs = 5
        max_features = 120

        (model, model_dict) = sm.best_model(
            transform_data,
            target_data,
            predictors,
            model_list,
            count,
            split_size,
            )
        print ('-----------------------------------')
        for i in model_dict:

            #

            print ('Model: ' + i + ' --- Accuracy: ' + str(model_dict[i]))
        print ('The best model selected is {}'.format(model))
        (final_model_list, X_test_list, y_test_list, dic_score_all) = \
            tr.train(
            model,
            transform_data,
            target_data,
            predictors,
            scaler_list,
            count,
            split_size,
            max_features,
            cv_number,
            num_jobs,
            )

        for (i, j, x, y) in zip(final_model_list, scaler_list,
                                X_test_list, y_test_list):
            predictions = i.predict(x)
            print ('-----------------------------------')
            print ('Result using {} - Logistic Regression Model'.format(j))
            print ('Confusion Matrix: ', confusion_matrix(predictions,
                   y))
            print ('Accuracy: ', np.round(accuracy_score(predictions,
                   y), 4))


run_model.run('dataset_challenge_one.tsv')

