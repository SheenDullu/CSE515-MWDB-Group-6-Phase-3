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
    datadir = input("Enter the directory containing all the files: ")
    mapping = dict()
    mapping[1] = "vattene"
    mapping[2] = "combinato"
    mapping[3] = "daccordo"

    # file = input("Enter the latent feature file: ")
    old_data = pd.read_csv("latent_features_pca_tf.txt", header=None)
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
    print(mapping[predictions])

    # testing_names = ["11.csv","12.csv","13.csv","14.csv","15.csv","16.csv","17.csv","18.csv","19.csv","20.csv","21.csv","22.csv","23.csv","24.csv","25.csv","26.csv","27.csv","28.csv","29.csv","30.csv","31.csv","259.csv","260.csv","261.csv","262.csv","263.csv","264.csv","265.csv","266.csv","267.csv","268.csv","269.csv","270.csv","271.csv","272.csv","273.csv","274.csv","275.csv","276.csv","277.csv","278.csv","279.csv","559.csv","560.csv","561.csv","562.csv","563.csv","564.csv","565.csv","566.csv","567.csv","568.csv","569.csv","570.csv","571.csv","572.csv","573.csv","574.csv","575.csv","576.csv","577.csv","578.csv","579.csv"]
    # x_testing = list()
    # y_testing = list()
    # for i,j in enumerate(testing_names):
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
