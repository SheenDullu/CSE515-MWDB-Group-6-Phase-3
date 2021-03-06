import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param

def get_all_sub_folders(folder_directory):
    return [d for d in os.listdir(folder_directory) if os.path.isdir(os.path.join(folder_directory, d))]


def main():
    # file = input("Enter the similarity matrix you want to use:")
    data = pd.read_csv("similarity_matrix_pca_tf.csv")
    heap = list()
    column_names = list(data.columns)
    # column_names.pop(0)
    datadir = read_directory()
    k = int(input("Enter the k value: "))
    n = int(input("Enter the value of n: "))

    seed_objects = list()
    for i in range(n):
        num = input("Enter the object number {} : ".format(str(i + 1)))
        seed_objects.append(num + ".csv")
    m = int(input("Enter the number of dominant gestures: "))

    # dropping the first column
    data_copy = data.drop([column_names[0]], axis=1)
    data_copy = data_copy.to_numpy()

    # for creating adjacency matrix
    adjacency_matrix = np.zeros(data_copy.shape)
    column_names.pop(0)
    for i in range(len(data_copy)):
        for idx, value in enumerate(data_copy[i]):
            heapq.heappush(heap, (-value, column_names[idx]))
        for _ in range(k):
            value, obj = heapq.heappop(heap)
            y = column_names.index(obj)
            adjacency_matrix[i][y] = data_copy[i][y]

    beta = 0.7
    # print(adjacency_matrix)

    # for creating v (seed matrix)
    v = np.zeros(len(column_names))
    for seed in seed_objects:
        index = column_names.index(seed)
        v[index] = 1
    v = v.reshape(-1, 1)

    adjacency_matrix = adjacency_matrix / adjacency_matrix.sum(axis=0, keepdims=1)
    v = v / v.sum(axis=0, keepdims=1)
    # print(adjacency_matrix)
    u = v.copy()
    # print(u.shape)
    for _ in range(100):
        tem_mat1 = np.matmul(adjacency_matrix, u)
        tem_mat1 = tem_mat1 * (1 - beta)
        tem_mat2 = v * beta
        u_dash = np.add(tem_mat1, tem_mat2)
        if np.array_equal(u, u_dash):
            break
        else:
            u = u_dash
    # print(u)

    # finding the m dominent gestures.
    heap2 = list()
    for i in range(len(u)):
        heapq.heappush(heap2, (-u[i][0], i))
    output = list()
    for _ in range(m):
        res, index = heapq.heappop(heap2)
        output.append(column_names[index])

    print("-------------------")
    print(output)

    dirs = get_all_sub_folders(datadir)
    for i in output:
        for folder in dirs:
            file_ = datadir + '\\' + folder + "\\" + i
            file_data = pd.read_csv(file_, header=None).T
            # file_data = file_data.values.tolist()
            file_data.plot.line()
            plt.title("" + folder + " component of " + i)
            plt.show()
        #     break
        # break


if __name__ == "__main__":
    main()
