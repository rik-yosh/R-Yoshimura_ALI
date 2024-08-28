print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tslearn.clustering import TimeSeriesKMeans


os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

print("___ data loading ___")
X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")

# PTの実際のデータ作成
clusteringDF = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
Life = X["InternalMedicineSurvival"]

seed =42
alpha = 0.9
n_x = 6

# km = TimeSeriesKMeans(n_clusters=n_x,
#             init='k-means++',     # k-means++法によりクラスタ中心を選択
#             n_init=10,
#             metric="dtw",
#             max_iter=100,
#             random_state=seed)                      # クラスタリングの計算を実行
# Label = km.fit_predict(clusteringDF)

# Label = pd.DataFrame({"dtw_"+str(n_x):Label})
# Label = Label.replace({
#     0:"G1",
#     1:"G5",
#     2:"G2",
#     3:"G6",
#     4:"G3",
#     5:"G4"
# })
# Label.to_csv(path+"/Data/DTWclustLabel.csv",index=False)
Label = pd.read_csv(path+"/Data/DTWclustLabel.csv")["dtw_6"]

fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5))
for l_x, Gname in enumerate(sorted(set(Label))):
    print(Gname)
    boolist = Label==Gname
    index = [ i for i in range(len(Label)) if boolist[i]]
    dat = clusteringDF.iloc[index,:]
    life = Life.iloc[index]
    for id in dat.index:
        axes[l_x].plot([0,1,2,3,7],dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life[id]))
        axes[l_x].scatter([0,1,2,3,7],dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life[id]))
        axes[l_x].set_yticks([0,20,40,60,80,100,120,140,160])
        axes[l_x].set_yticklabels([0,20,40,60,80,100,120,140,160],fontsize=13)
        axes[l_x].set_xticks([0,1,2,3,7])
        axes[l_x].set_xticklabels([0,1,2,3,7],fontsize=13)
        axes[l_x].set_ylim(0,160)
        axes[l_x].set_xlim(-1, 8)
        axes[l_x].set_title(Gname,fontsize=14)
fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
fig.supylabel("PT% (%)",fontsize=14)
fig.savefig(path+"/Output/Fig2/Fig2A.png",bbox_inches="tight",dpi=200)



# Fig 2B
clusteringDF = X.loc[:,['PT_percentage_d0','PT_percentage_d7']]
### pickleで保存したファイルを読み込み
with open(path+'/Data/logiR_thresh.txt', mode='br') as fi:
  PT_thresh = pickle.load(fi)[0]

print("PT thresh: ",PT_thresh)

alpha = 0.7
fig, axes = plt.subplots(1,6,tight_layout=True,figsize=(15,5),)
for l_x, Gname in enumerate(sorted(set(Label))):
    print(Gname)
    boolist = Label==Gname
    index = [ i for i in range(len(Label)) if boolist[i]]
    dat = clusteringDF.iloc[index,:]
    life = Life.iloc[index]
    for id in dat.index:
        axes[l_x].plot([0,1],dat.loc[id,:],alpha=alpha,zorder=1,lw=2,color=cm.Pastel2(life[id]))
        axes[l_x].scatter([0,1],dat.loc[id,:],alpha=alpha,zorder=2,s=50,marker="o",edgecolors="k",color="w",)
        axes[l_x].set_yticks([0,20,40,60,80,100,120,140,160])
        axes[l_x].set_yticklabels([0,20,40,60,80,100,120,140,160],fontsize=13)
        axes[l_x].set_xticks([0,1])
        axes[l_x].set_xticklabels(["day0","day7"],fontsize=13)
        axes[l_x].set_ylim(0,160)
        axes[l_x].set_xlim(-1,2)
        axes[l_x].set_title(Gname,fontsize=14)

    axes[l_x].axhline(y=PT_thresh, zorder=3, lw=2, color="magenta",linestyle="--")

fig.supxlabel("Days post-admission",y=0.06,fontsize=14)
fig.supylabel("PT% (%)",fontsize=14)
fig.savefig(path+"/Output/Fig2/Fig2B.png",bbox_inches="tight",dpi=200)