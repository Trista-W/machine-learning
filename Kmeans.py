import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform, cdist
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# data = loadmat('ex7data2.mat')
# X = data['X']

X = np.loadtxt('kernel_kmeans_testdata.txt')

'''
Kmeans by principle and algorithm
'''


def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)
    for i in range(k):
        centroids[i, :] = X[idx[i], :]
    return centroids


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)
    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


def compute_centroids(X, idx, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids


def k_means(X, init_centroids, max_iters):
    m, n = X.shape
    k = init_centroids.shape[0]
    idx = np.zeros(m)
    centroids = init_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids


def gaussian_kernel(X, sigma, k, max_iter, n_init):
    '''
    :param X: input data
    :param sigma: sigma parameter in the gaussian kernel
    :param k: number of clusters
    :param init_idx: initial labels of data
    :param max_iter: max number of iterations
    :param n_init: number of different initializations to run kmeans
    :return: return the final labels, centroids and sse
    '''

    kernel_matrix = np.exp(-0.5/(sigma**2)*squareform(pdist(X, 'sqeuclidean')))
    n = X.shape[0]
    centroids_history = {}
    idx_history = {}
    sse_history = np.zeros((n_init, 1))
    for h in range(n_init):
        centroids = init_centroids(X, k)
        idx = np.random.choice(range(k), size=n)
        for j in range(max_iter):
            dist = np.zeros((n, k))
            dist1 = np.diag(kernel_matrix)
            for i in range(k):
                kth_cluster_ind = (idx == i) + 0.0
                kth_cluster_matrix = np.outer(kth_cluster_ind, kth_cluster_ind)
                dist2 = 2.0 * np.sum(np.tile(kth_cluster_ind, (n, 1))*kernel_matrix, axis=1)/np.sum(kth_cluster_ind)
                dist3 = np.sum(kth_cluster_matrix*kernel_matrix)/np.sum(kth_cluster_matrix)
                dist[:, i] = dist1 - dist2 + dist3
                indices = np.where(idx == i)
                centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
            idx = np.argmin(dist, axis=1)
            sse = np.sum((np.min(dist, axis=1))**2)
            centroids_history[h] = centroids
            idx_history[h] = idx
            sse_history[h] = sse
    best_iter = np.argmin(sse_history)
    best_sse = sse_history[best_iter]
    best_controids = centroids_history[best_iter]
    best_idx = idx_history[best_iter]

    return best_idx, best_controids, best_sse


''' K-Means '''
# init = init_centroids(X, 2)
# idx, centroids = k_means(X, init, 300)
''' Gaussian kernel K-Means'''
best_idx, best_controids, best_sse = gaussian_kernel(X, 5.0, 2, 100, 20)


'''
Kmeans by sklearn
'''
# scaler = StandardScaler()
# scaledX = scaler.fit_transform(X)
# # kmeans = KMeans(
# #     init='random',
# #     n_clusters=3,
# #     n_init=10,
# #     max_iter=300,
# #     random_state=42
# # )
# # kmeans.fit(scaledX)
# # '''the lowest SSE'''
# # SSE = kmeans.inertia_
# # idx = kmeans.labels_
#
# kmeans_kwargs = {
#     'init': 'random',
#     'n_init': 10,
#     'max_iter': 300,
#     'random_state': 42,
# }
# sse = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#     kmeans.fit(scaledX)
#     sse.append(kmeans.inertia_)
#
#
'''plot clusters'''
cluster1 = X[np.where(best_idx == 0)[0], :]
cluster2 = X[np.where(best_idx == 1)[0], :]
# cluster3 = X[np.where(best_idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
# ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()


'''
elbow point
'''
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()
