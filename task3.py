import math
import random

import numpy as np
import pandas as pd


def generateRandomVector(length, min, max):
    arr = []
    for i in range(length):
        num = round(random.uniform(min, max), 2)
        arr.append(num)
    return arr


def LSH(layers, hashes):
    data = pd.read_csv("similarity_matrix_pca.csv")
    column_names = list(data.columns)
    data_copy = data.drop([column_names[0]], axis=1)
    data_copy = data_copy.to_numpy()
    middleIndex = math.ceil(len(data_copy) / 2)
    LayerStr = []
    for l in range(layers):
        strs = ["" for x in range(len(data_copy))]
        for i in range(hashes):
            randomVector = generateRandomVector(len(data_copy), -2.0, 2.0)
            dotp = []
            for j in range(len(data_copy)):
                dotp.append(np.dot(randomVector, data_copy[j]))

            dotpCopy = dotp.copy()
            dotpCopy.sort()

            for index in range(len(dotp)):
                if dotp[index] <= dotpCopy[middleIndex]:
                    strs[index] += "0"
                else:
                    strs[index] += "1"
        LayerStr.append(strs)

    return LayerStr


def findGestures(bins, gesture, t):
    data = pd.read_csv("similarity_matrix_pca.csv")
    column_names = list(data.columns)
    data_copy = data.drop([column_names[0]], axis=1)
    data_copy = data_copy.to_numpy()
    index = column_names.index(gesture) - 1
    binCount = 0
    binIndices = []
    for bin in bins:
        indices = [i for i, x in enumerate(bin) if x == bin[index]]
        binCount += 1
        [binIndices.append(x) for x in indices if x not in binIndices]

    indexChange = 1
    while len(binIndices) < t:
        if indexChange <= len(bin[index]):
            binVal = bin[index]
            if binVal[len(binVal) - indexChange] == "0":
                binVal = binVal.replace(binVal[len(binVal) - indexChange], '1')
            else:
                binVal = binVal.replace(binVal[len(binVal) - indexChange], '0')

            for bin in bins:
                indices = [i for i, x in enumerate(bin) if x == binVal]
                binCount += 1
                [binIndices.append(x) for x in indices if x not in binIndices]
            indexChange += 1
        else:
            binVal = bin[index]
            indexChange2 = (indexChange - len(bin[index])) * -1

            for bin in bins:
                indices = [i for i, x in enumerate(bin) if x[:indexChange2] == binVal[:indexChange2]]
                binCount += len(binVal) * (indexChange - len(bin[index]))
                [binIndices.append(x) for x in indices if x not in binIndices]

            indexChange += 1

    binIndices.sort()
    distList = []
    for val in binIndices:
        dist = np.linalg.norm(
            data_copy[index] - data_copy[val])  # This can be changed (Calculates similarity with L2 distance)
        distList.append(dist)

    indexList = np.argsort(distList)[:t]
    resultList = [binIndices[i] for i in indexList]
    fileNameList = [column_names[i + 1] for i in resultList]
    return fileNameList, binCount


def main():
    layers = int(input("Enter the amount of layers you want: "))
    hashes = int(input("Enter the amount of hashes per layer you want: "))

    bins = LSH(layers, hashes)

    gesture = input("Enter a gesture file name: ")
    t = int(input("Enter how many similar gestures you want: "))

    output, count = findGestures(bins, gesture + ".csv", t)
    print("Number of Bins Searched: " + str(count))
    return output,t


if __name__ == "__main__":
    print(main())
