print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import math
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score,StratifiedKFold, train_test_split
from sklearn.metrics import auc, RocCurveDisplay, r2_score, PrecisionRecallDisplay
from sklearn.inspection import permutation_importance
import math


if os.path.exists('./R-Yoshimura_ALI'):
    os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "sans-serif"

#---------------------------#
# new dataset
#---------------------------#
print("___ split columns each date and treatment ___")
# Clustering ClustDataの追加
X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")
RMSE_wider = pd.read_csv(path+"/Data/Fig3/RMSEchangge_wider.csv")
print(RMSE_wider)
dName = ["d0","d0m","d1","d1m","d2","d2m","d3","d3m","d7","d7m"]
RMSEdata = RMSE_wider[dName]
Labels=pd.read_csv(path+"/Data/DTWclustLabel.csv")
Life = X["InternalMedicineSurvival"].reset_index(drop=True)

########################################################
# Plot all patient
########################################################
rmse_array = RMSEdata.to_numpy()
i_list = RMSEdata.index.to_list()

# 最初に0が出現するインデックスを取得
first_zero_index = Life.tolist().index(0)

# 最初に1が出現するインデックスを取得
first_one_index = Life.tolist().index(1)

for i in i_list:
    rmse_values = rmse_array[i,:]

rmse_mean = rmse_array.mean(0)

dif_d7 = rmse_mean[-1] - rmse_mean[0]
dif_d2 = rmse_mean[3]- rmse_mean[0]
dif_rat = round((1-dif_d2/dif_d7)*100,1)
print("Dif ratio until day2:",dif_rat)


fig, axes = plt.subplots(1,2,figsize=(15,6))
axes[0].plot(range(len(dName)),rmse_mean,alpha=1,lw=3,color="k",label="Mean")
# axes[0].set_title("Mean RMSE change",fontsize=14)
axes[0].set_xticks(range(len(dName)))
axes[0].set_ylim(0,10)
axes[0].tick_params(axis="y",labelsize=13)
axes[0].set_xticklabels(["$D_{0}$","$DTI_{0}$","$D_{1}$","$DTI_{1}$","$D_{2}$","$DTI_{2}$","$D_{3}$","$DTI_{3}$","$D_{7}$","$DTI_{7}$"], rotation=0, fontsize=13, ha='center')

for i in i_list:
    rmse_values = rmse_array[i,:]
    if i == first_zero_index:
        axes[1].plot(range(len(dName)),rmse_values,alpha=0.7,color=cm.Pastel2(Life[i]),label="Restored")
    elif i == first_one_index:
        axes[1].plot(range(len(dName)),rmse_values,alpha=0.7,color=cm.Pastel2(Life[i]),label="Severe")
    else:
        axes[1].plot(range(len(dName)),rmse_values,alpha=0.7,color=cm.Pastel2(Life[i]))
axes[1].set_xticks(range(len(dName)))
axes[1].set_xticklabels(["$D_{0}$","$DTI_{0}$","$D_{1}$","$DTI_{1}$","$D_{2}$","$DTI_{2}$","$D_{3}$","$DTI_{3}$","$D_{7}$","$DTI_{7}$"], rotation=0,fontsize=14, ha='center')
axes[1].tick_params(axis="y",labelsize=14)
fig.supylabel("Rooted Mean Squired Error (RMSE)",fontsize=14,x=0.08)
fig.savefig(path+"/Output/Fig3/Fig3C.png",bbox_inches='tight',dpi=200)
plt.close()