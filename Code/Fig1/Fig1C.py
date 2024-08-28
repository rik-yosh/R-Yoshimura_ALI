print("___ Import Module ___")
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm


os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

print("___ data loading ___")
# Data loading
data = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")

# PTの実際のデータ作成
ptData = data.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
Life = data["InternalMedicineSurvival"]
alpha = 0.9

fig, axes = plt.subplots(1,2,figsize=(8,5))
for life in sorted(set(Life)):
    index = Life[Life==life].index
    dat = ptData.iloc[index,:]
    for id in dat.index:
        axes[life].plot([0,1,2,3,7], dat.loc[id,:],alpha=alpha,color=cm.Pastel2(life))
        axes[life].scatter([0,1,2,3,7], dat.loc[id,:],alpha=alpha,marker="o",color=cm.Pastel2(life))
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
fig.savefig(path+"/Output/Fig1/Fig1C.png",bbox_inches="tight")
plt.close()