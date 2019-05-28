''' There was initially more to this, but this way of solving the problem was abandoned
    This file is just here as proof I tried clustering. It probably won't work anymore
'''

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, v_measure_score
from globals import *

def cluster(X, e=EPSILON):
    clust = AgglomerativeClustering(n_clusters=3, )
    cluster = clust.fit_predict(X)
    return cluster


def run(data=[], lables=[], e=EPSILON):
    if data == []:
        data = np.load(DIM_REDUX)
        lables = np.load(LABLES)

    clusters = cluster(data)
    score = v_measure_score(lables, clusters)
    print(str(score))


def test(e):
    data = np.load(DIM_REDUX)
    lables = np.load(LABLES)

    ld = get_lable_dict(lables)

    test_indexes = np.r_[ld[0], ld[1], ld[2]]
    test_data = data[test_indexes]
    test_lables = lables[test_indexes]

    run(data=test_data, lables=test_lables, e=e)


def tune():
    es = range(100, 10000, 100)
    for e in es:
        print("Epsilon: " + str(e) + ': ', end='')
        test(e)

tune()