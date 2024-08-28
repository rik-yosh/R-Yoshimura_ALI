print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import math
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score,StratifiedKFold, train_test_split
from sklearn.metrics import auc, RocCurveDisplay, r2_score, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
import math


os.chdir('./1_ALF_new')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

print("___ data loading ___")
rf_seed =111
alpha = 0.75

X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_median.csv").drop([223],axis=0).reset_index(drop=True)
Labels=pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv").drop([223],axis=0).reset_index(drop=True)

# X, label, Lifeを並び替え
df_forSort = pd.concat([X,Labels],axis=1)
df_forSort["InternalMedicineSurvival"] = pd.Categorical(df_forSort["InternalMedicineSurvival"], categories = [0,1])
df_forSort = df_forSort.sort_values("InternalMedicineSurvival")
index_sort = df_forSort.index
df_forSort = df_forSort.reset_index(drop=True)

X = df_forSort.drop(["InternalMedicineSurvival","dtw_6"],axis=1)
clusteringDF = df_forSort.loc[:,['PT_percentage_d0','PT_percentage_d7']]
Life = df_forSort["InternalMedicineSurvival"].astype(int)
Label = df_forSort["dtw_6"]

f = open(path+'/data/ChanpionData/logiR_thresh.txt',"rb")
PT_thresh = pickle.load(f)


fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5))
for l_x, Gname in enumerate(sorted(set(Label))):
    boolist = Label==Gname
    index = [ i for i in range(len(Label)) if boolist[i]]
    dat = clusteringDF.iloc[index,:]
    life = Life.iloc[index]
    for id in dat.index:
        axes[l_x].plot([1,2],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life[id]),zorder=1)
        axes[l_x].scatter([1,2],dat.loc[id,:],alpha=alpha,s=100,edgecolor='k',marker="o",color="w",zorder=2)
    axes[l_x].axhline(PT_thresh, linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
    # )
    axes[l_x].set(
        ylim=(0,160),
        xlim=(0,3)
    )
    axes[l_x].set_title(Gname,fontsize=14)
    axes[l_x].set_xticks([0,1,2,3])
    axes[l_x].set_xticklabels(['','day0', 'day7',''],fontsize=13)
    axes[l_x].set_yticks([0,20,40,60,80,100,120,140,160])
    axes[l_x].set_yticklabels([0,20,40,60,80,100,120,140,160],fontsize=13)
# plt.legend(fontsize=13)
fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
fig.supylabel("PT% (%)",fontsize=14)
fig.savefig(path+"/results/ChanpionData/ptVisualization_comp07.pdf",bbox_inches="tight")


G456_pat_num_underThresh = 0
G456_pat_num_all = 0
# Day7のPTでthreshを超えている数の割合を求める
for l_x, Gname in enumerate(sorted(set(Label))):
    boolist = Label==Gname
    index = [ i for i in range(len(Label)) if boolist[i]]
    dat = clusteringDF.iloc[index,:]
    life = Life.iloc[index]
    pat_num_all = len(dat.index)
    print("Group:",Gname,pat_num_all)
    pat_num_underThresh = sum(dat['PT_percentage_d7'].to_list()<=PT_thresh[0])
    
    if (Gname == "G4")|(Gname == "G5")|(Gname == "G6"):
        G456_pat_num_all += pat_num_all
        G456_pat_num_underThresh += pat_num_underThresh
    print("under thresh ratio:",round(pat_num_underThresh/pat_num_all*100,1))
    print("Death ratio:",round(life.sum()/len(life.index)*100,1))

print(G456_pat_num_underThresh,G456_pat_num_all)
print("G4,5,6: under thresh ratio:",round(G456_pat_num_underThresh/G456_pat_num_all*100,1))


clusteringDF = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
expOver_list = []
expUnder_list = []
for index_x in range(len(clusteringDF)):
    dat = clusteringDF.iloc[index_x,:]
    expOver = sum(dat[['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']].to_list()>=PT_thresh[0]) >= 1
    expUnder = sum(dat[['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']].to_list()>=PT_thresh[0]) == 0
    expOver_list.append(expOver)
    expUnder_list.append(expUnder)

EO_inedx = clusteringDF[expOver_list].index
EU_index = clusteringDF[expUnder_list].index

print(Life[EO_inedx].sum(),"/",len(Life[EO_inedx].index))
print("Severe Ratio in patients who have had PT over 50:",round(Life[EO_inedx].sum()/len(Life[EO_inedx].index)*100,1))

print(Life[EU_index].sum(),"/",len(Life[EU_index].index))
print("Severe Ratio in patients who haven't had PT over 50:",round(Life[EU_index].sum()/len(Life[EU_index].index)*100,1))