import heapq
import os
import re
import numpy as np
import pandas as pd

def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param
    
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
    datadir = read_directory()
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
    training_names_1 = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv", "11.csv", "12.csv", "13.csv", "14.csv", "15.csv", "16.csv", "17.csv", "18.csv", "19.csv", "20.csv", "21.csv", "22.csv", "23.csv", "24.csv", "25.csv", "26.csv", "27.csv", "28.csv", "29.csv", "30.csv","31.csv"]
    training_names_2 = ["249.csv", "250.csv", "251.csv", "252.csv", "253.csv", "254.csv", "255.csv", "256.csv", "257.csv", "258.csv", "259.csv", "260.csv", "261.csv", "262.csv", "263.csv", "264.csv", "265.csv", "266.csv", "267.csv", "268.csv", "269.csv", "270.csv", "271.csv", "272.csv", "273.csv", "274.csv", "275.csv", "276.csv", "277.csv", "278.csv","279.csv"]
    training_names_3 = ["559.csv","560.csv","561.csv","562.csv","563.csv","564.csv","565.csv","566.csv","567.csv","568.csv","569.csv","570.csv","571.csv","572.csv","573.csv","574.csv","575.csv","576.csv","577.csv","578.csv","579.csv","580.csv", "581.csv", "582.csv", "583.csv", "584.csv", "585.csv", "586.csv", "587.csv",
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



    ################################################# Testing ################################################
    # not_testing_names = pd.read_csv("all_labels.csv",header=None)
    # not_testing_names = not_testing_names[0].values.tolist()
    # not_testing_names = [str(i)+".csv" for i in not_testing_names]

    # testing_names = pd.read_csv("labels.csv",header=None)
    # labelling_val = testing_names[2].values.tolist()
    # testing_names = testing_names[0].values.tolist()
    # final_testing_name = list()
    # final_labelling = list()
    # for i in testing_names:
    #     if i in not_testing_names:
    #         continue
    #     final_testing_name.append(i)
    #     final_labelling.append(labelling_val[testing_names.index(i)])

    # new_labels = list()
    # for i in final_testing_name:
    #     location = all_files_objects.index(i)
    #     output = list()
    #     current_rank1 = rank_for_class(adjacency_matrix.copy(), training_names_1, column_names, location)
    #     current_rank2 = rank_for_class(adjacency_matrix.copy(), training_names_2, column_names, location)
    #     current_rank3 = rank_for_class(adjacency_matrix.copy(), training_names_3, column_names, location)
    #     output.append((current_rank1, 1))
    #     output.append((current_rank2, 2))
    #     output.append((current_rank3, 3))
    #     output.sort(key=lambda x: x[0])
    #     new_labels.append(output[-1][1])
    # acc = sum(1 for x,y in zip(new_labels,final_labelling) if x == y) / float(len(new_labels))
    # print(acc)



if __name__ == "__main__":
    main()
