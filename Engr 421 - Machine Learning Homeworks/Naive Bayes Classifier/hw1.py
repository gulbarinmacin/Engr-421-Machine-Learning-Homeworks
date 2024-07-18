import numpy as np
import pandas as pd


X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]
    X_test = X[50000:]

    y_train = y[:50000]
    y_test = y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    K = np.max(y)
    class_priors = [np.mean(y == (c + 1)) for c in range(K)]
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    K=np.max(y)
    lenght=X.shape[1] # numbers of charcters in each sequence    

    #for pAcd
    pAcd = np.zeros((K, lenght))
    for k in range(K):
        for l in range(lenght):
            subset = X[y==(k+1)]
            result = np.sum(subset[:, l] == 'A')
            pAcd[k][l] = (result / subset.shape[0])
    
    #for pCcd
    pCcd = np.zeros((K, lenght))
    for k in range(K):
        for l in range(lenght):
            subset = X[y==(k+1)]
            result = np.sum(subset[:, l] == 'C')
            pCcd[k][l] = (result / subset.shape[0])
    
    #for pGcd
    pGcd = np.zeros((K, lenght))
    for k in range(K):
        for l in range(lenght):
            subset = X[y==(k+1)]
            result = np.sum(subset[:, l] == 'G')
            pGcd[k][l] = (result / subset.shape[0])
            
    #for pTcd
    pTcd = np.zeros((K, lenght))
    for k in range(K):
        for l in range(lenght):
            subset = X[y==(k+1)]
            result = np.sum(subset[:, l] == 'T')
            pTcd[k][l] = (result / subset.shape[0])
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    row = len(X)
    column = len(class_priors)
    score_values = np.zeros((row, column))
    for c in range(column):
        for i in range(len(X[0])):
            pcds = [pAcd[c][i], pCcd[c][i], pGcd[c][i], pTcd[c][i]]
            for j, k in enumerate(X):
                score_values[j, c] += np.log(pcds['ACGT'.index(k[i])])
        score_values[:, c] += np.log(class_priors[c])
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    prediction = np.argmax(scores, axis=1) + 1
    confusion_matrix =  pd.crosstab(prediction, y_truth, rownames=['y_pred'], colnames=['y_truth'])
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
