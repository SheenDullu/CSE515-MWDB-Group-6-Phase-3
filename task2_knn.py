import os
import re
import numpy as np
import pandas as pd


def getAVector(file):
    file = open(file, 'r')
    f = file.read()
    vector_string = f.split(',')
    vector = [float(i) for i in vector_string]
    return vector


# def distance(pA, pB):
#     return np.sum((pA - pB)**2)**0.5

def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param

def distance(a, b, p=2):
    dim = len(a)
    distance = 0

    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p

    distance = distance ** (1 / p)

    return distance


def knn(X, y, x_query, k=6):
    m = X.shape[0]
    distances = []

    for i in range(m):
        dis = distance(x_query, X[i])
        distances.append((dis, y[i]))

    distances = sorted(distances)
    distances = distances[:k]

    distances = np.array(distances)
    labels = distances[:, 1]

    uniq_label, counts = np.unique(labels, return_counts=True)
    pred = uniq_label[counts.argmax()]
    return int(pred)


def main():
    datadir = read_directory()
    mapping = dict()
    mapping[1] = "vattene"
    mapping[2] = "combinato"
    mapping[3] = "daccordo"

    # file = input("Enter the latent feature file: ")
    old_data = pd.read_csv("latent_features_pca_task1.txt", header=None)
    training_names = list()
    data_ = pd.read_csv("all_labels.csv",header=None)
    for i in data_[0].values.tolist():
        training_names.append(str(i) + ".csv")
    # training_names = ["1.csv", "2.csv", "3.csv", "4.csv", "5.csv", "6.csv", "7.csv", "8.csv", "9.csv", "10.csv",
    #                   "249.csv", "250.csv", "251.csv", "252.csv", "253.csv", "254.csv", "255.csv", "256.csv", "257.csv",
    #                   "258.csv", "580.csv", "581.csv", "582.csv", "583.csv", "584.csv", "585.csv", "586.csv", "587.csv",
    #                   "588.csv", "589.csv"]
    # x = old_data.to_numpy()
    all_files_objects = os.listdir(os.path.join(datadir, "W"))
    # all_files_objects.sort(key=lambda x: int(x.split(".")[0]))
    all_files_objects.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    all_label_names = pd.read_csv("labels.csv", header=None)
    all_label_names.columns = ["index", "name", "labels"]
    x_train = list()
    y = list()
    for i, j in enumerate(training_names):
        idx = all_files_objects.index(j)
        x_train.append(old_data.iloc[idx].values.tolist())
        y.append(all_label_names["labels"].iloc[idx])
        # break
    x_train = np.asarray(x_train)
    y_train = np.asarray(y)

    query_object = input("Enter the query object number: ")
    query_object = query_object + ".csv"
    x_test = old_data.iloc[all_files_objects.index(query_object)].to_numpy()
    predictions = knn(x_train, y_train, x_test)
    print("" + query_object + " belongs to '" + mapping[predictions] + "' class")

    ################################################## Testing ##############################################

    # not_testing_names = some_data = pd.read_csv("all_labels.csv",header=None)
    # not_testing_names = not_testing_names[0].values.tolist()
    # not_testing_names = [str(i)+".csv" for i in not_testing_names]

    # testing_names = some_data = pd.read_csv("labels.csv",header=None)
    # testing_names = testing_names[0].values.tolist()

    # final_testing_name = list()
    # for i in testing_names:
    #     if i in not_testing_names:
    #         continue
    #     final_testing_name.append(i)

    # x_testing = list()
    # y_testing = list()
    # for i,j in enumerate(final_testing_name):
    #     idx = all_files_objects.index(j)
    #     x_testing.append(old_data.iloc[idx].values.tolist())
    #     y_testing.append(all_label_names["labels"].iloc[idx])
    # x_testing = np.asarray(x_testing)

    # accuracies = list()
    # for g in range(3,50):
    #     y_hat = list()
    #     for h in range(len(x_testing)):
    #         temp_pred = knn(x_train,y_train,x_testing[h],g)
    #         y_hat.append(temp_pred)
    #     acc = sum(1 for x,y in zip(y_testing,y_hat) if x == y) / float(len(y_testing))
    #     accuracies.append(acc)
    # print(accuracies)


if __name__ == "__main__":
    main()
