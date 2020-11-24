import glob
import os

import numpy as np
import pandas as pd
import scipy.stats


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


    return np.log(((p*(1-q))+0.5)/((q*(1-p))+1))

def main(results,t):
    # file = input("Enter the similarity matrix you want to use:")
    datadir = pd.read_csv("similarity_matrix_pca_tf.csv")
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

    quant_df['mean'] = quant_df.apply(lambda x: np.mean(x[2:]), axis=1)
    quant_df['std'] = quant_df.apply(lambda x: np.std(x[2:]), axis=1)
    quant_df = (quant_df[["Comp", "Sensor_id", "mean", "std"]])
    #
    # quant_df=quant_df.fillna(0.1)
    # rel_df=rel_df.fillna(0.1)
    # irrel_df=irrel_df.fillna(0.1)
    prob_list=[]

    #Caclulate probability value
    for file in file_list:
        wrd=pd.read_csv(file,header=None,names=["Comp","Sensor_id","Symbolic_Rep","Time","Average_Amp","Std","Average_Quantization"])
        quant_wrd = (wrd[["Comp", "Sensor_id", "Average_Quantization"]].groupby(["Comp", "Sensor_id"], as_index=False).mean())
        quant_wrd["prob"]=quant_wrd.apply(lambda x: calc(x[0],x[1],x[2]) ,axis=1)
        sigma_prob=quant_wrd["prob"].sum()
        prob_list.append([file.split("\\")[-1],(sigma_prob)])

    print(sorted(prob_list,key= lambda x: x[1], reverse=True)[:10])

# results = [('3.csv', 1), ('9.csv', 1), ('269.csv', 0), ('1.csv', 1), ('568.csv', 0), ('2.csv', 1), ('249.csv', 0),
#            ('13.csv', 1), ('560.csv', 0), ('278.csv', 0)]
# t = 10
# main(results,t)

