from FeatureSelection import dimensionality_reduction as dim_red
from DataPreprocess import read_data as rd
from DataPreprocess import preprocessing 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import KernelPCA
import collections


def pc_component(filename):
    d = rd.tsv_read("/Users/jananisundaresan/Desktop/ayasdi/dataset_challenge_one.tsv")
    predictors_data, predictors, target_data, target = rd.data_transformation(d)
    d = rd.handle_missing_data(predictors_data)
    new_dict, dic_score, variance, components, pca_data = dim_red.pca(d, predictors)
    dic = {'{}'.format(i+1): components[0][i] for i in range(272)}
    score = {'{}'.format(i+1): variance[i] for i in range(272)}
    dic_list = list(dic.values())
    score_list = list(score.values())
    count = -1
    for i in range(len(dic_list)):
        count = i+1
        dic_list[i] = (np.round(dic_list[i], 7))
        sortdf = {dic_list[i]: score_list[i] for i in range(272)}
    var = []
    for i in sortdf.values():
        var.append(i)
    f = np.array(var)
    
    return f, sortdf

f, sortdf = pc_component("dataset_challenge_one.tsv")
plt.figure(figsize=(10, 5))
plt.title("PCA Score vs Variance vs Cumulative Varaince", fontsize=20)
plt.plot(sortdf.values(), label='variance')
plt.plot(f.cumsum(), label="Cumulative variance")
plt.plot(sortdf.keys(), label='PC Score')
plt.legend(fontsize=10)
plt.ylabel("Value", fontsize=15)
plt.xlabel("PC #", fontsize=20)
plt.yticks(fontsize=15)
plt.axvline(39, c='red')
plt.xticks(np.arange(1, 272, 12), rotation='vertical', fontsize=15)
plt.grid()
plt.show()