import heapq
import os
import re
import numpy as np
import pandas as pd


def rank_for_class(adj, seed_objects, column_names, location):
    beta = 0.7

    # for creating v (seed matrix)
    v = np.zeros(len(column_names))
    for seed in seed_objects:
        index = column_names.index(seed)
        v[index] = 1
    v = v.reshape(-1, 1)

    # normalizing adjaceny matrix across columns
    adj = adj / adj.sum(axis=0, keepdims=1)
    v = v / v.sum(axis=0, keepdims=1)

    # initializing u matrix
    u = v.copy()

    for _ in range(100):
        tem_mat1 = np.matmul(adj, u)
        tem_mat1 = tem_mat1 * (1 - beta)
        tem_mat2 = v * beta
        u_dash = np.add(tem_mat1, tem_mat2)
        if np.array_equal(u, u_dash):
            break
        else:
            u = u_dash
    return u[location][0]


def main():
    # file = input("Enter the similarity matrix you want to use:")
    data = pd.read_csv("similarity_matrix_pca_tf.csv")
    datadir = input("File path for directory: ")
    all_files_objects = os.listdir(os.path.join(datadir, "W"))
    all_files_objects.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    all_label_names = pd.read_csv("labels.csv", header=None)
    all_label_names.columns = ["index", "name", "labels"]
    mapping = dict()
    mapping[1] = "vattene"
    mapping[2] = "combinato"
    mapping[3] = "daccordo"

    column_names = list(data.columns)
    data_copy = data.drop([column_names[0]], axis=1)
    data_copy = data_copy.to_numpy()
    training_names_1 = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv"]
    training_names_2 = ["249.csv", "250.csv", "251.csv", "252.csv", "253.csv", "254.csv", "255.csv", "256.csv",
                        "257.csv", "258.csv"]
    training_names_3 = ["580.csv", "581.csv", "582.csv", "583.csv", "584.csv", "585.csv", "586.csv", "587.csv",
                        "588.csv", "589.csv"]

    query_object = input("Enter the query object number: ")
    query_object = query_object + ".csv"
    location = all_files_objects.index(query_object)

    adjacency_matrix = np.zeros(data_copy.shape)
    heap = []
    column_names.pop(0)
    k = 10
    for i in range(len(data_copy)):
        for idx, value in enumerate(data_copy[i]):
            heapq.heappush(heap, (-value, column_names[idx]))
        for _ in range(k):
            value, obj = heapq.heappop(heap)
            y = column_names.index(obj)
            adjacency_matrix[i][y] = data_copy[i][y]
    output = list()
    current_rank1 = rank_for_class(adjacency_matrix.copy(), training_names_1, column_names, location)
    current_rank2 = rank_for_class(adjacency_matrix.copy(), training_names_2, column_names, location)
    current_rank3 = rank_for_class(adjacency_matrix.copy(), training_names_3, column_names, location)
    output.append((current_rank1, 1))
    output.append((current_rank2, 2))
    output.append((current_rank3, 3))
    output.sort(key=lambda x: x[0])
    print("-----------------------")
    print(mapping[output[-1][1]])


if __name__ == "__main__":
    main()
