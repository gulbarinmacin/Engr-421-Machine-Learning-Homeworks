import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

# get X and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 0.00
maximum_value = 2.00
step_size = 0.002
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlim([-0.05, 2.05])
    plt.xlabel("Time (sec)")
    plt.ylabel("Signal (millivolt)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)

#helper node for building tree
def buildTree(node_indices, is_terminal, need_split, node_features, node_splits, node_means, a=1):
    if not need_split.get(a, 0):
        return

    need_split[a] = 0
    node_means[a] = np.mean(y_train[node_indices[a]])

    if len(node_indices[a]) <= P:
        is_terminal[a] = 1
        return

    is_terminal[a] = 0

    def splitting(split):
        result = (1 / len(node_indices[a])) * (
            np.sum((y_train[node_indices[a][X_train[node_indices[a], 0] > split]] - np.mean(y_train[node_indices[a][X_train[node_indices[a], 0] > split]])) ** 2) +
            np.sum((y_train[node_indices[a][X_train[node_indices[a], 0] <= split]] - np.mean(y_train[node_indices[a][X_train[node_indices[a], 0] <= split]])) ** 2)
        )
        return result

    node_features[a] = 0
    node_splits[a] = min(pd.Series(np.unique(X_train[node_indices[a]])).rolling(2).mean().dropna().values, key=splitting)

    node_indices.update({2 * a: node_indices[a][X_train[node_indices[a], 0] > node_splits[a]], 2 * a + 1:  node_indices[a][X_train[node_indices[a], 0] <= node_splits[a]]})
    is_terminal.update({child_node: 0 for child_node in [2 * a, 2 * a + 1]})
    need_split.update({child_node: 1 for child_node in [2 * a, 2 * a + 1]})

    buildTree(node_indices, is_terminal, need_split, node_features, node_splits, node_means, 2 * a) 
    buildTree(node_indices, is_terminal, need_split, node_features, node_splits, node_means, 2 * a + 1)


# STEP 2
# should return necessary data structures for trained tree
def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}
    node_means = {}

    # your implementation starts below
    node_indices, is_terminal, need_split = {1: np.arange(len(X_train))}, {1: 0}, {1: 1}
    buildTree(node_indices, is_terminal, need_split, node_features, node_splits, node_means)
    # your implementation ends above
    return(is_terminal, node_features, node_splits, node_means)

# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    def predict_single(x):
        node = 1
        while not is_terminal[node]:
            node = 2 * node + (x[node_features[node]] <= node_splits[node])
        return node_means[node]

    y_hat = np.array([predict_single(x) for x in X_query])

    # your implementation ends above
    return(y_hat)

# STEP 4
# assuming that there are T terminal node
# should print T rule sets as described
def extract_rule_sets(is_terminal, node_features, node_splits, node_means):
    def get_rules(node, rules=None):
        if rules is None:  
            rules = []
        if node == 1:  
            return rules[::-1]  
        parent = node // 2
        comparison = ">" if node % 2 == 0 else "<="
        rules.append(f"'x{node_features[parent] + 1} {comparison} {node_splits[parent] : .02f}'")
        return get_rules(parent, rules.copy())  # Pass a copy of the list to avoid modification

    for node in [key for key, value in is_terminal.items() if value]:
        rule_set = get_rules(node)
        print(f"Node {node:01d}: [{' and '.join(rule_set)}] => {node_means[node]}")
    # your implementation ends above

P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)
fig.savefig("decision_tree_regression_{}.pdf".format(P), bbox_inches = "tight")

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets(is_terminal, node_features, node_splits, node_means)
