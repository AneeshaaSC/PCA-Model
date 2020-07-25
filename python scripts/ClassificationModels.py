import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
warnings.simplefilter(action='ignore')


class classification_model:

    def __init__(
        self,
        select_model,
        Xtrain,
        Xtest,
        ytrain,
        ytest,
                ):
        self.select_model = select_model
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest

    print ('Calcualting accuracy using base model')

    def base_model(self):

        if self.select_model == 'LogisticRegression':
            log_model = LogisticRegression(max_iter=50)
            log_model.fit(self.Xtrain, self.ytrain)
            log_predictions = log_model.predict(self.Xtest)
            log_accuracy = accuracy_score(log_predictions, self.ytest)

            return log_accuracy
        elif self.select_model == 'XGBoost':

            xgb_model = xgb.XGBClassifier()
            xgb_model.fit(self.Xtrain, self.ytrain)
            xgb_predictions = xgb_model.predict(self.Xtest)
            xgb_accuracy = accuracy_score(xgb_predictions, self.ytest)

            return xgb_accuracy
        elif self.select_model == 'RandomForest':

            rf_model = RandomForestClassifier()
            rf_model.fit(self.Xtrain, self.ytrain)
            rf_predictions = rf_model.predict(self.Xtest)
            rf_accuracy = accuracy_score(rf_predictions, self.ytest)

            return rf_accuracy
        else:

            dt_model = DecisionTreeClassifier()
            dt_model.fit(self.Xtrain, self.ytrain)
            dt_predictions = dt_model.predict(self.Xtest)
            dt_accuracy = accuracy_score(dt_predictions, self.ytest)

            return dt_accuracy

    def best_model(self):

        if self.select_model == 'LogisticRegression':
            log_model = LogisticRegression()
            print ('Returning the best model: Logistic Regression')
            return log_model
        elif self.select_model == 'XGBoost':

            print ('Returning the best model: XGBoost')
            xgb_model = xgb.XGBClassifier()
            return xgb_model
        elif self.select_model == 'RandomForest':

            print ('Returning the best model: Random Forest')
            rf_model = RandomForestClassifier()
            return rf_model
        else:

            dt_model = DecisionTreeClassifier()
            print ('Returning the best model: Decision Tree')
            return dt_model


class optimization:

    def hyperparamters(model, max_features):

        rf_params = {
            'max_depth': [2, 3, 4, 5],
            'n_estimators': [30, 40],
            'max_features': np.arange(1, max_features),
            'min_samples_leaf': [5, 10, 15],
            'min_samples_split': [5, 8],
            'max_leaf_nodes': [2, 3, 4],
            'random_state': [45],
            }
        dt_params = {
            'max_depth': [2, 3, 4, 5],
            'n_estimators': [30, 40],
            'max_features': np.arange(1, max_features),
            'min_samples_leaf': [5, 10, 15],
            'min_samples_split': [5, 8],
            'max_leaf_nodes': [2, 3, 4],
            'random_state': [45],
            }
        log_params = {
            'max_iter': [50, 100, 150],
            'multi_class': ['warn'],
            'penalty': ['l1'],
            'random_state': [45],
            'solver': ['liblinear'],
            'n_jobs': [20],
            }
        xgb_params = {
            'max_depth': [2, 3, 4, 5],
            'max_features': np.arange(1, max_features),
            'min_samples_leaf': [5, 10, 15],
            'min_samples_split': [5, 8],
            'max_leaf_nodes': [2, 3, 4],
            'random_state': [45],
            }

        if model == 'DecisionTree':
            print ('Getting the hyperparameters for Decision Tree')
            return dt_params
        elif model == 'RandomForest':
            print ('Getting the hyperparameters for Random Forest')
            return rf_params
        elif model == 'XGBoost':
            print ('Getting the hyperparameters for XGBoost')
            return xgb_params
        else:
            print ('Getting the hyperparameters for Logistic Regression')
            return log_params

    def hypertune_model(
        model,
        params,
        cv_number,
        num_jobs,
        X_train,
        y_train,
                       ):

        print ('Grid Search to get the best model using cross-validation')
        search = GridSearchCV(model, param_grid=params, cv=cv_number,
                              n_jobs=num_jobs, return_train_score=True)
        search.fit(X_train, y_train)
        return search.best_estimator_

