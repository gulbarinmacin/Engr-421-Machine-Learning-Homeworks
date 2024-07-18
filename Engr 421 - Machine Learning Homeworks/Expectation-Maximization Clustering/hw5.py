import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, -0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, +0.0]])

group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=',')
    distances = dt.cdist(X, means)
    assignments = np.argmin(distances, axis=1)

    covariances = []
    for k in range(K):
        Xk = X[assignments == k] 
        mean_k = means[k]
        var_k = np.mean(np.sum((Xk - mean_k)**2, axis=1)) 
        covariances.append(np.eye(X.shape[1]) * var_k) 

    priors = np.ones(K) / K            
    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    N = X.shape[0]
    step = 0
    while step < 100:
        responsibilities = np.zeros((K, N))
        for k in range(K):
            responsibilities[k] = priors[k] * stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
        responsibilities /= np.sum(responsibilities, axis=0)

        for k in range(K):
            means[k] = np.sum(responsibilities[k].reshape(-1, 1) * X, axis=0) / np.sum(responsibilities[k])
            covariances[k] = np.dot((responsibilities[k].reshape(-1, 1) * (X - means[k])).T, (X - means[k])) / np.sum(responsibilities[k])
            priors[k] = np.sum(responsibilities[k]) / N

        step += 1

    assignments = np.argmax(responsibilities, axis=0)    
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    colors = ['b', 'g', 'r', 'c']
    x1, x2 = int(X[:, 0].min()-1), int(X[:, 0].max() + 1 )
    y1, y2 = int(X[:, 1].min()), int(X[:, 1].max() )

    x1_x2, y1_y2 = np.meshgrid(np.arange(x1,x2, 0.1),
                         np.arange(y1, y2, 0.1))

    for i in range(K):
        Z = multivariate_normal.pdf(np.c_[x1_x2.ravel(), y1_y2.ravel()], mean=group_means[i], cov=group_covariances[i])
        Z = Z.reshape(x1_x2.shape)
        plt.contour(x1_x2, y1_y2, Z, levels=[0.01], linestyles='dashed')

    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k])

    for k in range(K):
        Z = multivariate_normal.pdf(np.c_[x1_x2.ravel(), y1_y2.ravel()],mean=means[k], cov=covariances[k])
        Z = Z.reshape(x1_x2.shape)
        plt.contour(x1_x2, y1_y2, Z, levels=[0.01], colors=colors[k])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)