import heapq

import numpy as np
import pandas as pd


def main(results,t):
    # file = input("Enter the similarity matrix you want to use:")
    data = pd.read_csv("similarity_matrix_pca_tf.csv")
    heap = list()
    column_names = list(data.columns)

    data_copy = data.drop([column_names[0]],axis=1)
    data_copy = data_copy.to_numpy()

    k = 10
    beta = 0.7
    adjacency_matrix = np.zeros(data_copy.shape)
    column_names.pop(0)
    for i in range(len(data_copy)):
        for idx,value in enumerate(data_copy[i]):
            heapq.heappush(heap,(-value,column_names[idx]))
        for _ in range(k):
            value, obj = heapq.heappop(heap)
            y = column_names.index(obj)
            adjacency_matrix[i][y] = data_copy[i][y]
    
    # for creating v (seed matrix)
    v = np.zeros(len(column_names))
    for seed in results:
        if seed[1] == 1:
            index = column_names.index(seed[0])
            v[index] = 1
        v = v.reshape(-1,1)
    print(v)

    adjacency_matrix = adjacency_matrix/adjacency_matrix.sum(axis=0,keepdims=1)

    v = v / v.sum(axis=0, keepdims=1)

    u = v.copy()

    for _ in range(100):
        tem_mat1 = np.matmul(adjacency_matrix,u)
        tem_mat1 = tem_mat1 * (1-beta)
        tem_mat2 = v * beta
        u_dash = np.add(tem_mat1, tem_mat2)
        if np.array_equal(u,u_dash):
            break
        else:
            u = u_dash
    # print(u)

    # finding the m dominent gestures.
    heap2 = list()
    for i in range(len(u)):
        heapq.heappush(heap2,(-u[i][0],i))
    output = list()
    for _ in range(t):
        res, index = heapq.heappop(heap2)
        output.append(column_names[index])

    print("-------------------")
    print(output)


if __name__ == "__main__":
    main()