import glob
import os

import numpy as np
import pandas as pd
import scipy.stats
import math
import Phase3


def find_target(comp,sensor,df):
    rel_target=df[(df["Comp"] ==comp) & (df["Sensor_id"]==sensor)]

    mean=rel_target['mean'].values[0]
    std=rel_target['std'].values[0]

    return scipy.stats.norm(mean, std)

def calc(comp,sensor,value):

    #Find pdf value wrt to relevant irrelevant and entire dataset
    rel=find_target(comp,sensor,rel_df).pdf(value)
    irrel=find_target(comp,sensor,irrel_df).pdf(value)
    p_x=find_target(comp,sensor,quant_df).pdf(value)

    #Naiive Bayes
    p=rel*p_x
    q=irrel*p_x

    multiplier=1

    important_sensors = set([5, 6, 7, 8, 9, 10, 11, 12])
    important_comp=set(["X","Y"])
    if sensor in important_sensors:
        multiplier*=4
    if comp in important_comp:
        multiplier*=4
    try:
        result= np.log(((p*(1-q))+0.5)/((q*(1-p))+1))
    except RuntimeWarning:
        result= 0

    return result*multiplier


def main(results,t):

    # file = input("Enter the similarity matrix you want to use:")
    datadir = Phase3.read_directory()
    global rel_df
    global irrel_df
    global quant_df
    rel_flag=1
    irrel_flag=1

    #Calculate mean and std wrt sensor and component for relevant and irrelevant files
    for file in results:
        wrd=pd.read_csv(os.path.join(datadir,(file[0].split(".")[0]+".wrd")),header=None,names=["Comp","Sensor_id","Symbolic_Rep","Time","Average_Amp","Std","Average_Quantization"])
        quant_wrd=(wrd[["Comp","Sensor_id","Average_Quantization"]].groupby(["Comp","Sensor_id"], as_index=False).mean())
        if(file[1]==1):
            if(rel_flag):
                rel_df=quant_wrd
                rel_flag=0
            else:
                rel_df= rel_df.merge(quant_wrd,how='inner',left_on=["Comp","Sensor_id"],right_on=["Comp","Sensor_id"])
        else:
            if(irrel_flag):
                irrel_df=quant_wrd
                irrel_flag=0
            else:
                irrel_df=irrel_df.merge(quant_wrd,how='inner',left_on=["Comp","Sensor_id"],right_on=["Comp","Sensor_id"])

    print("Relevant and Irrelevant document distributions noted")
    rel_df['mean']=rel_df.apply(lambda x: np.mean(x[2:]),axis=1)
    irrel_df['mean']=irrel_df.apply(lambda x: np.mean(x[2:]),axis=1)
    rel_df['std']=rel_df.apply(lambda x: np.std(x[2:]),axis=1)
    irrel_df['std']=irrel_df.apply(lambda x: np.std(x[2:]),axis=1)
    rel_df=(rel_df[["Comp","Sensor_id","mean","std"]])
    irrel_df=irrel_df[["Comp","Sensor_id","mean","std"]]


    #Calculate mean and std wrt component and std
    file_list=(glob.glob(datadir+"\\*.wrd"))
    flag=1
    for file in file_list:
        wrd=pd.read_csv(file,header=None,names=["Comp","Sensor_id","Symbolic_Rep","Time","Average_Amp","Std","Average_Quantization"])
        quant_wrd = (wrd[["Comp", "Sensor_id", "Average_Quantization"]].groupby(["Comp", "Sensor_id"], as_index=False).mean())
        if(flag):
            quant_df=quant_wrd
            flag=0
        else:
            quant_df=quant_df.merge(quant_wrd,how='inner',left_on=["Comp","Sensor_id"],right_on=["Comp","Sensor_id"])

    print("dataset distributions noted")

    quant_df['mean'] = quant_df.apply(lambda x: np.mean(x[2:]), axis=1)
    quant_df['std'] = quant_df.apply(lambda x: np.std(x[2:]), axis=1)
    quant_df = (quant_df[["Comp", "Sensor_id", "mean", "std"]])

    quant_df=quant_df.fillna(0.1)
    rel_df=rel_df.fillna(0.1)
    irrel_df=irrel_df.fillna(0.1)
    prob_list=[]

    #Caclulate probability value
    for file in file_list:
        wrd=pd.read_csv(file,header=None,names=["Comp","Sensor_id","Symbolic_Rep","Time","Average_Amp","Std","Average_Quantization"])
        quant_wrd = (wrd[["Comp", "Sensor_id", "Average_Quantization"]].groupby(["Comp", "Sensor_id"], as_index=False).mean())
        quant_wrd["prob"]=quant_wrd.apply(lambda x: calc(x[0],x[1],x[2]) ,axis=1)
        sigma_prob=quant_wrd["prob"].sum()
        prob_list.append([file.split("\\")[-1],(sigma_prob)])

    print("probabilities calculated")
    for doc in sorted(prob_list,key= lambda x: x[1], reverse=True)[:t]:
        print(doc[0].split(".")[0]+".csv")

# results = [('6.csv', 1), ('561_7.csv', 0), ('23_0.csv', 1), ('265_6.csv', 0), ('14_5.csv', 1), ('274_1.csv', 0), ('31_6.csv', 1), ('570_2.csv', 0), ('257_0.csv', 0), ('578_8.csv', 0)]
# t = 10
# main(results,t)

