from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import KernelPCA


class dimensionality_reduction:

    def pca(data, predictors):
        pca = PCA()
        pca_data = pca.fit_transform(data)

        def pca_components():
            print("pca")
            n_pcs = pca.components_.shape[0]
            variance = pca.explained_variance_ratio_.round(2)
            pc_number = [np.abs(pca.components_[i]).argmax()
                         for i in range(n_pcs)]
            initial_feature_names = predictors
            pc_names = [initial_feature_names[pc_number[i]]
                        for i in range(n_pcs)]
            pc_score = [np.round((pca.components_[i]).max(), 3)
                        for i in range(n_pcs)]
            dic_score = {pc_names[i]: pc_score[i] for i in range(n_pcs)}
            print("Number of unique features selected as ", len(dic_score))
            feature_dict = {key: value for (key, value)
                            in dic_score.items() if value >= 0.1}

            return feature_dict, dic_score, variance, pca.components_
        
        feature_dict, dic_score_all, variance, pca.components_ = pca_components()
        return feature_dict, dic_score_all, variance, pca.components_, pca_data
        
    def kernel_pca(data, kernel, n_comp):
        
        print("Performing PCA using {}".format(kernel))
        pca = KernelPCA(kernel=kernel, n_components=n_comp)
        pca_data = pca.fit_transform(data)
        
        return pca_data
        