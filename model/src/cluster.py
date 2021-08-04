"""
author: Wira D K Putra
25 February 2020

See original repo at
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score       


class OptimizedAgglomerativeClustering:
    def __init__(self, max_cluster=10):  #max cluster => 最大集群
        self.kmax = max_cluster
        
    def fit_predict(self, X):
        best_k = self._find_best_k(X)
        membership = self._fit(X, best_k)
        #print(membership, "membership==========================>") #建立說話者位置
        return membership

    def _fit(self, X, n_cluster):
        #print(n_cluster, "n_cluster=====================>") # 2 ~ 9
        #print(X, "x=======================>")
        return AgglomerativeClustering(n_cluster).fit_predict(X)
        
    def _find_best_k(self, X):
        cluster_range = range(2, min(len(X), self.kmax))
        #print(cluster_range, "cluster_range===============>")  #  range(2, 10)
        score = [silhouette_score(X, self._fit(X, k)) for k in cluster_range]
        #print(score, "score====================>") #[0.34995967, 0.3727883, 0.32661456, 0.24481739, 0.24548504, 0.22590487, 0.23958632, 0.23972072]
        best_k = cluster_range[np.argmax(score)] # np.argmax 返回最大值索引號
        #print(best_k, "best_k=================>") # 3  (分類出3個人)
        return best_k
