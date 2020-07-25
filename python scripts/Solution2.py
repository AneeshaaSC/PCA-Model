from FeatureSelection import dimensionality_reduction as dim_red
from DataPreprocess import read_data as rd
from DataPreprocess import preprocessing
import numpy as np
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
warnings.simplefilter(action='ignore')


data = rd.tsv_read('dataset_challenge_one.tsv')
(predictors_data, predictors, target_data, target) = \
    rd.data_transformation(data)
transform_data = rd.handle_missing_data(predictors_data)

label = np.array(target_data).T

scaler_list = ['robust_scaler', 'standard_scaler', 'min_max_scaler']
kernel_list = ['linear', 'poly', 'rbf']
pca_data_list = []


for i in scaler_list:
    for k in kernel_list:
        prep_data = preprocessing(i, transform_data).prep_scaler()
        pca_data = dim_red.kernel_pca(prep_data, k, len(label))
        pca_data_list.append(pca_data)

count = -1
index_1 = []
index_0 = []
for i in label:
    count = count + 1
    if i == '1':
        index_1.append(count)
    else:
        index_0.append(count)

title = []
for s in scaler_list:
    for k in kernel_list:
        title.append(s + '-' + k)

(rows, cols) = (3, 3)
clusters = 2
cdict = {1: 'red', 0: 'blue'}
(fig, axs) = plt.subplots(3, 3, figsize=(15, 8), sharex=True,
                          sharey=True)

for (j, ax) in enumerate(axs.flat):

    ax.scatter(pca_data_list[j][0][index_0],
               pca_data_list[j][1][index_0], c=cdict[0], label='Class 0'
               )
    ax.scatter(pca_data_list[j][0][index_1],
               pca_data_list[j][1][index_1], c=cdict[1], label='Class 1'
               )
    ax.set_title(title[j], fontsize=20)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    if j % 3 == 0:
        ax.set_ylabel('Dimension 1', fontsize=20)
    if j in (6, 7, 8):
        ax.set_xlabel('Dimension 2', fontsize=20)

(handles, labels) = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', fontsize=15)

