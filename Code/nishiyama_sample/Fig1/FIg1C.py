print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tslearn.clustering import TimeSeriesKMeans


os.chdir('./1_ALF_new')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

print("___ data loading ___")
rf_seed =111
alpha = 0.9
n_x = 6

y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters.txt',sep=",").drop([223],axis=0).reset_index(drop=True)
y = y.loc[:,['g_mean','d_mean','P0_mean']]

X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_median.csv").drop([223],axis=0).reset_index(drop=True)

# PTの実際のデータ作成
clusteringDF = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
print("clusteringDF.head: \n",clusteringDF.head())

# km = TimeSeriesKMeans(n_clusters=n_x,
#             init='k-means++',     # k-means++法によりクラスタ中心を選択
#             n_init=10,
#             metric="dtw",
#             max_iter=100,
#             random_state=111)                      # クラスタリングの計算を実行
# Label = km.fit_predict(clusteringDF)

# Label = pd.DataFrame({"dtw_"+str(n_x):Label})
# Label = Label.replace({
#     0:"G5",
#     1:"G2",
#     2:"G4",
#     3:"G6",
#     4:"G1",
#     5:"G3"
# })
# Label.to_csv(path+"/data/ChanpionData/DTWclustLabel.csv",index=False)

Label = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv").drop([223],axis=0).reset_index(drop=True)

Life = X["InternalMedicineSurvival"]

# X, label, Lifeを並び替え
df_forSort = pd.concat([X,Label],axis=1)
df_forSort["InternalMedicineSurvival"] = pd.Categorical(df_forSort["InternalMedicineSurvival"], categories = [0,1])
df_forSort = df_forSort.sort_values("InternalMedicineSurvival")
index_sort = df_forSort.index
df_forSort = df_forSort.reset_index(drop=True)

X = df_forSort.drop(["InternalMedicineSurvival","dtw_6"],axis=1)
clusteringDF = df_forSort.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
Life = df_forSort["InternalMedicineSurvival"]
Label = df_forSort["dtw_6"]

# G4のIDを取得
raw = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_rawData.csv").drop([223],axis=0).reset_index(drop=True)
raw_d0 = raw[raw['採血日程']=="day0"].reset_index()
#並び替え
raw_d0 = raw_d0.iloc[index_sort,:].reset_index(drop=True)
#並び替えたlabelをつける
raw_d0['Group'] = Label.to_list()

DF_g4 = raw_d0.iloc[raw_d0[raw_d0['Group']=="G4"].index,[0,1,-1,-2]].reset_index(drop=True)
DF_g4['index'] = DF_g4['index']+2
DF_g4 = DF_g4.rename(columns={'index':"元データindex"})
DF_g4.to_excel(path+"/data/ChanpionData/dataloading/Group4_ID.xlsx",index=False)


fig, axes = plt.subplots(1,2,figsize=(8,5))
for life in sorted(set(Life)):
    index = Life[Life==life].index
    dat = clusteringDF.iloc[index,:]
    for id in dat.index:
        axes[life].plot([0,1,2,3,7],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life))
        axes[life].scatter([0,1,2,3,7],dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life))
        axes[life].set_ylim(0, 160)
        if life==0:
            axes[life].set_title("TFS Patients",fontsize=14)
        if life==1:
           axes[life].set_title("non-TFS Patients",fontsize=14)
        axes[life].set_yticks([0,20,40,60,80,100,120,140,160])
        axes[life].set_yticklabels([0,20,40,60,80,100,120,140,160],fontsize=13)
        axes[life].set_xticks([0,1,2,3,7])
        axes[life].set_xticklabels([0,1,2,3,7],fontsize=13)
        axes[life].set_ylim(0,160)
        axes[life].set_xlim(-1, 8)
# fig.suptitle("Time-Series PT change")
fig.supxlabel("Days post-admission",fontsize=14)
fig.supylabel("PT% (%)",fontsize=14)
fig.savefig(path+"/results/ChanpionData/ptVisualization/ptVisualization_allpatients_PTp.pdf",bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5))
for l_x, Gname in enumerate(sorted(set(Label))):
    boolist = Label==Gname
    index = [ i for i in range(len(Label)) if boolist[i]]
    dat = clusteringDF.iloc[index,:]
    life = Life.iloc[index]
    for id in dat.index:
        axes[l_x].plot([0,1,2,3,7],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life[id]))
        axes[l_x].scatter([0,1,2,3,7],dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life[id]))
        # axes[l_x].axhline(PT_thresh, linewidth = 2,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
        # axes[l_x].set_ylim(0, 160)
        # axes[l_x].set_title(Gname)
        axes[l_x].set_yticks([0,20,40,60,80,100,120,140,160])
        axes[l_x].set_yticklabels([0,20,40,60,80,100,120,140,160],fontsize=13)
        axes[l_x].set_xticks([0,1,2,3,7])
        axes[l_x].set_xticklabels([0,1,2,3,7],fontsize=13)
        axes[l_x].set_ylim(0,160)
        axes[l_x].set_xlim(-1, 8)
        axes[l_x].set_title(Gname,fontsize=14)
fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
fig.supylabel("PT% (%)",fontsize=14)
fig.savefig(path+"/results/ChanpionData/ptVisualization/ptVisualization_dtwCluster_PTp.pdf",bbox_inches="tight")



# #------------------------------------------------------------------------------------------------------------------------------
# # PTINRでの描画
# X = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_TS_Wide.csv")
# clusteringDF = X.loc[:,['PTINR_d0','PTINR_d1','PTINR_d2','PTINR_d3','PTINR_d7']]
# print("clusteringDF.head: \n",clusteringDF.head())

# inrThresh = 1.371011

# fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5))
# for l_x, Gname in enumerate(sorted(set(Label))):
#     boolist = Label==Gname
#     index = [ i for i in range(len(Label)) if boolist[i]]
#     dat = clusteringDF.iloc[index,:]
#     life = Life.iloc[index]
#     for id in dat.index:
#         axes[l_x].plot([0,1,2,3,7],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life[id]))
#         axes[l_x].scatter([0,1,2,3,7],dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life[id]))
#         axes[l_x].axhline(inrThresh, linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (inrThresh))
#         # axes[l_x].axhline(PT_thresh, linewidth = 2,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
#         # axes[l_x].set_ylim(0, 160)
#         # axes[l_x].set_title(Gname)
#         axes[l_x].set_yticks([0,2.5,5,7.5,10,12.5,5,15,17.5,20])
#         axes[l_x].set_yticklabels([0,2.5,5,7.5,10,12.5,5,15,17.5,20],fontsize=13)
#         axes[l_x].set_xticks([0,1,2,3,7])
#         axes[l_x].set_xticklabels([0,1,2,3,7],fontsize=13)
#         axes[l_x].set_ylim(0,22)
#         axes[l_x].set_xlim(-1, 8)
#         axes[l_x].set_title(Gname,fontsize=14)
# fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
# fig.supylabel("PT-INR",fontsize=14)
# fig.savefig(path+"/results/ChanpionData/ptVisualization/ptVisualization_dtwCluster_INR.pdf",bbox_inches="tight")



# #------------------------------------------------------------------------------------------------------------------------------
# # PTsでの描画
# X = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_TS_Wide.csv")
# clusteringDF = X.loc[:,['PTs_d0','PTs_d1','PTs_d2','PTs_d3','PTs_d7']]
# print("clusteringDF.head: \n",clusteringDF.head())

# ptsThresh = 17.36044

# fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5))
# for l_x, Gname in enumerate(sorted(set(Label))):
#     boolist = Label==Gname
#     index = [ i for i in range(len(Label)) if boolist[i]]
#     dat = clusteringDF.iloc[index,:]
#     life = Life.iloc[index]
#     for id in dat.index:
#         axes[l_x].plot([0,1,2,3,7],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life[id]))
#         axes[l_x].scatter([0,1,2,3,7],dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life[id]))
#         axes[l_x].axhline(ptsThresh, linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (ptsThresh))
#         # axes[l_x].axhline(PT_thresh, linewidth = 2,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
#         # axes[l_x].set_ylim(0, 160)
#         # axes[l_x].set_title(Gname)
#         axes[l_x].set_yticks([0,10,20,30,40,50,60,70,80])
#         axes[l_x].set_yticklabels([0,10,20,30,40,50,60,70,80],fontsize=13)
#         axes[l_x].set_xticks([0,1,2,3,7])
#         axes[l_x].set_xticklabels([0,1,2,3,7],fontsize=13)
#         axes[l_x].set_ylim(0,83)
#         axes[l_x].set_xlim(-1, 8)
#         axes[l_x].set_title(Gname,fontsize=14)
# fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
# fig.supylabel("Prothrombin time (s)",fontsize=14)
# fig.savefig(path+"/results/ChanpionData/ptVisualization/ptVisualization_dtwCluster_PTs.pdf",bbox_inches="tight")