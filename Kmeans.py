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
        np.random.seed(h)
        idx = np.random.choice(range(k), size=n, replace=True)
        centroids = X[np.random.choice(np.arange(n), k, replace=False), :]
        print(centroids)
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


# centroids do not update
# init = init_centroids(X, 2)
# idx, centroids = k_means(X, init, 300)
best_idx, best_controids, best_sse = gaussian_kernel(X, 1.0, 2, 100, 20)


'''
Kmeans by Python libraries
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
#
# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()



# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import cm
# import time
#
#
# #params
# k = 2 #number of cluster
# var = 5 #var in RFB kernel
# iterationCounter = 0
# input = np.loadtxt('kernel_kmeans_testdata.txt')
# initMethod = "byOriginDistance" #options = random, byCenterDistance, byOriginDistance
#
# def initCluster(dataInput, nCluster, method):
#     listClusterMember = [[] for i in range(nCluster)]
#     if (method == "random"):
#         shuffledDataIn = dataInput
#         np.random.shuffle(shuffledDataIn)
#         for i in range(0, dataInput.shape[0]):
#             listClusterMember[i%nCluster].append(dataInput[i,:])
#     if (method == "byCenterDistance"):
#         center = np.matrix(np.mean(dataInput, axis=0))
#         repeatedCent = np.repeat(center, dataInput.shape[0], axis=0)
#         deltaMatrix = abs(np.subtract(dataInput, repeatedCent))
#         euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
#         dataNew = np.array(np.concatenate((euclideanMatrix, dataInput), axis=1))
#         dataNew = dataNew[np.argsort(dataNew[:, 0])]
#         dataNew = np.delete(dataNew, 0, 1)
#         divider = dataInput.shape[0]/nCluster
#         for i in range(0, dataInput.shape[0]):
#             listClusterMember[np.int(np.floor(i/divider))].append(dataNew[i,:])
#     if (method == "byOriginDistance"):
#         origin = np.matrix([[0,0]])
#         repeatedCent = np.repeat(origin, dataInput.shape[0], axis=0)
#         deltaMatrix = abs(np.subtract(dataInput, repeatedCent))
#         euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))
#         dataNew = np.array(np.concatenate((euclideanMatrix, dataInput), axis=1))
#         dataNew = dataNew[np.argsort(dataNew[:, 0])]
#         dataNew = np.delete(dataNew, 0, 1)
#         divider = dataInput.shape[0]/nCluster
#         for i in range(0, dataInput.shape[0]):
#             listClusterMember[np.int(np.floor(i/divider))].append(dataNew[i,:])
#     print(listClusterMember)
#     return listClusterMember
#
# def RbfKernel(data1, data2, sigma):
#     delta =abs(np.subtract(data1, data2))
#     squaredEuclidean = (np.square(delta).sum(axis=1))
#     result = np.exp(-(squaredEuclidean)/(2*sigma**2))
#     return result
#
# def thirdTerm(memberCluster):
#     result = 0
#     for i in range(0, memberCluster.shape[0]):
#         for j in range(0, memberCluster.shape[0]):
#             result = result + RbfKernel(memberCluster[i, :], memberCluster[j, :], var)
#     result = result / (memberCluster.shape[0] ** 2)
#     return result
#
# def secondTerm(dataI, memberCluster):
#     result = 0
#     for i in range(0, memberCluster.shape[0]):
#         result = result + RbfKernel(dataI, memberCluster[i,:], var)
#     result = 2 * result / memberCluster.shape[0]
#     return result
#
# def plotResult(listClusterMembers, centroid, iteration, converged):
#     n = listClusterMembers.__len__()
#     color = iter(cm.rainbow(np.linspace(0, 1, n)))
#     plt.figure("result")
#     plt.clf()
#     plt.title("iteration-" + iteration)
#     for i in range(n):
#         col = next(color)
#         memberCluster = np.asmatrix(listClusterMembers[i])
#         plt.scatter(np.ravel(memberCluster[:, 0]), np.ravel(memberCluster[:, 1]), marker=".", s=100, c=col)
#     color = iter(cm.rainbow(np.linspace(0, 1, n)))
#     for i in range(n):
#         col = next(color)
#         plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]), marker="*", s=400, c=col, edgecolors="black")
#     if (converged == 0):
#         plt.ion()
#         plt.show()
#         plt.pause(0.1)
#     if (converged == 1):
#         plt.show(block=True)
#
# def kMeansKernel(data, initMethod):
#     global iterationCounter
#     memberInit = initCluster(data, k, initMethod)
#     nCluster = memberInit.__len__()
#     #looping until converged
#     while(True):
#         # calculate centroid, only for visualization purpose
#         centroid = np.ndarray(shape=(0, data.shape[1]))
#         for i in range(0, nCluster):
#             memberCluster = np.asmatrix(memberInit[i])
#             centroidCluster = memberCluster.mean(axis=0)
#             centroid = np.concatenate((centroid, centroidCluster), axis=0)
#         #plot result in every iteration
#         plotResult(memberInit, centroid, str(iterationCounter), 0)
#         oldTime = np.around(time.time(), decimals=0)
#         kernelResultClusterAllCluster = np.ndarray(shape=(data.shape[0], 0))
#         #assign data to cluster whose centroid is the closest one
#         for i in range(0, nCluster):#repeat for all cluster
#             term3 = thirdTerm(np.asmatrix(memberInit[i]))
#             matrixTerm3 = np.repeat(term3, data.shape[0], axis=0); matrixTerm3 = np.asmatrix(matrixTerm3)
#             matrixTerm2 = np.ndarray(shape=(0,1))
#             for j in range(0, data.shape[0]): #repeat for all data
#                 term2 = secondTerm(data[j,:], np.asmatrix(memberInit[i]))
#                 matrixTerm2 = np.concatenate((matrixTerm2, term2), axis=0)
#             matrixTerm2 = np.asmatrix(matrixTerm2)
#             kernelResultClusterI = np.add(-1*matrixTerm2, matrixTerm3)
#             kernelResultClusterAllCluster =\
#                 np.concatenate((kernelResultClusterAllCluster, kernelResultClusterI), axis=1)
#         clusterMatrix = np.ravel(np.argmin(np.matrix(kernelResultClusterAllCluster), axis=1))
#         listClusterMember = [[] for l in range(k)]
#         for i in range(0, data.shape[0]):#assign data to cluster regarding cluster matrix
#             listClusterMember[np.asscalar(clusterMatrix[i])].append(data[i,:])
#         for i in range(0, nCluster):
#             print("Cluster member numbers-", i, ": ", listClusterMember[0].__len__())
#         #break when converged
#         boolAcc = True
#         for m in range(0, nCluster):
#             prev = np.asmatrix(memberInit[m])
#             current = np.asmatrix(listClusterMember[m])
#             if (prev.shape[0] != current.shape[0]):
#                 boolAcc = False
#                 break
#             if (prev.shape[0] == current.shape[0]):
#                 boolPerCluster = (prev == current).all()
#             boolAcc = boolAcc and boolPerCluster
#             if(boolAcc==False):
#                 break
#         if(boolAcc==True):
#             break
#         iterationCounter += 1
#         #update new cluster member
#         memberInit = listClusterMember
#         newTime = np.around(time.time(), decimals=0)
#         print("iteration-", iterationCounter, ": ", newTime - oldTime, " seconds")
#     return listClusterMember, centroid
#
# clusterResult, centroid = kMeansKernel(input, initMethod)
# plotResult(clusterResult, centroid, str(iterationCounter) + ' (converged)', 1)
# print("converged!")
