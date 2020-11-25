import pandas as pd
import os
import glob
import task3
import heapq
import re
import math

def read_directory():
    with open("directory.txt", 'r') as f:
        param = f.read()
        f.close()
    return param


def getAllVectors(directory, model):
    vectors = dict()
    all_files = glob.glob(directory + "/" + model + "-vectors-*.txt")
    all_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split('(\d+)', var)])
    for filename in all_files:
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split(',')
                vector = [float(i) for i in row]
                vectors[filename.split('\\')[-1].split("-")[-1].split('.')[0]] = vector
    return vectors

#("1.csv",1)
#["1":[],"1_0":[]]
def main(outputs,t):
    # outputs,t = task3.main()
    directory = read_directory()



    all_vectors = getAllVectors(directory,"tf")
    cap_N = 0
    cap_R = 0
    t = 6
    # outputs = [("5.csv",1),("10.csv",1),("249.csv",0),("579.csv",0),("3.csv",1),("251.csv",0)]
    for item in outputs:
        if item[1] == 1:
            cap_R += 1
        cap_N += 1
    sim_value_all_objects = list()
    for i in all_vectors.keys():
        final_sim_value = 0
        for j in range(len(all_vectors["1"])):
            small_r = 0
            small_non_rel = 0
            if all_vectors[i][j]>0:
                d = 1
            else:
                # print("asdfasd")
                continue
            for item in outputs:
                if all_vectors[str(item[0].split(".")[0])][j]>0:
                    if item[1] == 1:
                        small_r += 1
                    else:
                        small_non_rel += 1
            
            p_i = (small_r + 0.5)/(cap_R + 1)
            u_i = (small_non_rel + 0.5)/(cap_N - cap_R + 1)
            sim_value = d * math.log((p_i*(1-u_i))/(u_i*(1-p_i)))
            final_sim_value += sim_value
        heapq.heappush(sim_value_all_objects,(-final_sim_value,str(i)+".csv"))
    
    output = list()
    for i in range(t):
        a,b = heapq.heappop(sim_value_all_objects)
        output.append(b)
    print(output)
    
if __name__ == "__main__":
    main()