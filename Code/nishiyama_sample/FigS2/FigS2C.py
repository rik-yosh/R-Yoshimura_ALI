print("___ Import Module ___")
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score

os.chdir('./1_ALF_new')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15


print("___ data loading ___")
y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters.txt',sep=",")
y = y.loc[:,['g_mean','d_mean','P0_mean']]

NgdP0 = ['g_mean', 'd_mean', 'P0_mean']
Nclust = ['Clustering']

Nlife = ['InternalMedicineSurvival']
Nd0_data = ['Age','Encephalopathy_grade_d0','MELDscore','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex_M','diagnosis_ALF_subacute','diagnosis_ALF_without_coma','diagnosis_ALI','diagnosis_LOHF','LiverAtrophy_.','ALTdevideLDH_new_under1.5_N','Plt_d0','PT_percentage_d0','PTINR_d0','PTs_d0','Alb_d0','BUN_d0','Cre_d0','Tbil_d0','Dbil_d0','AST_d0','ALT_d0','LDH_d0','NH3_d0','TDratio_d0']
Nd0_med = ['PE_NotDone','TASIT_NotDone','peripheral_pulse_NotDone','Anticoagulation_NotDone','CHDF_NotDone','VasopressorUse_1.0','Yesterday_FFPprescribing_U_d0','Yesterday_rhTM_NotDone_d0']
Nd1_data = ['Plt_d1','PT_percentage_d1','Alb_d1','BUN_d1','Cre_d1','AST_d1','ALT_d1','LDH_d1','NH3_d1','TDratio_d1']
Nd1_med = ['Yesterday_FFPprescribing_U_d1','Yesterday_rhTM_NotDone_d1']
Nd2_data = ['Plt_d2','PT_percentage_d2','Alb_d2','BUN_d2','Cre_d2','AST_d2','ALT_d2','LDH_d2','NH3_d2','TDratio_d2']
Nd2_med = ['Yesterday_FFPprescribing_U_d2','Yesterday_rhTM_NotDone_d2']
Nd3_data = ['Plt_d3','PT_percentage_d3','Alb_d3','BUN_d3','Cre_d3','AST_d3','ALT_d3','LDH_d3','TDratio_d3']
Nd3_med = ['Yesterday_FFPprescribing_U_d3','Yesterday_rhTM_NotDone_d3']
Nd7_data = ['Plt_d7','PT_percentage_d7','Alb_d7','BUN_d7','Cre_d7','AST_d7','ALT_d7','LDH_d7','TDratio_d7']
Nd7_med = ['Yesterday_FFPprescribing_U_d7','Yesterday_rhTM_NotDone_d7']
Ndlast = ['Encephalopathy_grade_worst','Etiology_DILI','Etiology_HAV','Etiology_HBV','Etiology_Others','Etiology_unknown']

n_tree=500
rf_seed=111
n_cv=5

rfr = RandomForestRegressor(n_estimators=n_tree, random_state=rf_seed)

r2_DF = pd.DataFrame(columns=['g_mean','d_mean','P0_mean'],index=[])
sse_DF = pd.DataFrame(columns=['g_mean','d_mean','P0_mean'],index=[])

if os.path.isfile(path+"/data/ChanpionData/r2_DF.csv"):
    r2_DF = pd.read_csv(path+"/data/ChanpionData/r2_DF.csv")
else:
    for i in range(1,51):
        print("data:",i)
        data = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_"+str(i)+".csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={"TDratio_bil":"TDratio_d0"})
        X = data.loc[:,Nd0_data]

        loo = LeaveOneOut()
        rfr = RandomForestRegressor(n_estimators=n_tree, random_state=rf_seed)

        y_pred = pd.DataFrame(index=[],columns=['g_mean','d_mean','P0_mean'])
        perm_imp_dfs = pd.DataFrame(index=[],columns=[])

        for train_index, test_index in loo.split(X):
            X_train, X_test,y_train, y_test = X.iloc[train_index,:],X.iloc[test_index,:],y.iloc[train_index,[0,1,2]],y.iloc[test_index,[0,1,2]]
            
            rfr.fit(X_train,y_train)
            
            # 予測データを出力
            one_pred = pd.DataFrame(rfr.predict(X_test),columns=['g_mean','d_mean','P0_mean'])
            y_pred = pd.concat([y_pred,one_pred],axis=0)

        r2_one = pd.DataFrame({
            "g_mean": r2_score(y["g_mean"],y_pred["g_mean"]),
            "d_mean": r2_score(y["d_mean"],y_pred["d_mean"]),
            "P0_mean": r2_score(y["P0_mean"],y_pred["P0_mean"])
        },index=[i-1])
        r2_DF = pd.concat([r2_DF,r2_one],axis=0)
    
    r2_DF.to_csv(path+"/data/ChanpionData/r2_DF.csv",index=False)
# r2_DF_norm = pd.DataFrame({
#     "g_mean":(r2_DF["g_mean"]-min(r2_DF["g_mean"]))/(max(r2_DF["g_mean"])-min(r2_DF["g_mean"])),
#     "d_mean":(r2_DF["d_mean"]-min(r2_DF["d_mean"]))/(max(r2_DF["d_mean"])-min(r2_DF["d_mean"])),
#     "P0_mean":(r2_DF["P0_mean"]-min(r2_DF["P0_mean"]))/(max(r2_DF["P0_mean"])-min(r2_DF["P0_mean"]))
# })
# sse_DF_norm = pd.DataFrame({
#     "g_mean":(sse_DF["g_mean"]-min(sse_DF["g_mean"]))/(max(sse_DF["g_mean"])-min(sse_DF["g_mean"])),
#     "d_mean":(sse_DF["d_mean"]-min(sse_DF["d_mean"]))/(max(sse_DF["d_mean"])-min(sse_DF["d_mean"])),
#     "P0_mean":(sse_DF["P0_mean"]-min(sse_DF["P0_mean"]))/(max(sse_DF["P0_mean"])-min(sse_DF["P0_mean"]))
# })

# R2（決定係数）
fig,ax = plt.subplots(nrows=1, ncols=3, tight_layout=True,figsize=(15,5))
# グラフ
ax[0].hist(r2_DF["g_mean"],color="black",alpha=0.4)
ax[0].axvline(x=r2_DF["g_mean"][0],color='black',linestyle='--')
ax[0].set_xlim(r2_DF["g_mean"].mean()-0.1, r2_DF["g_mean"].mean()+0.1)
ax[0].set_title("g",fontsize=18)


ax[1].hist(r2_DF["d_mean"],color="black",alpha=0.4)
ax[1].axvline(x=r2_DF["d_mean"][0],color='black',linestyle='--')
ax[1].set_xlim(r2_DF["d_mean"].mean()-0.1, r2_DF["d_mean"].mean()+0.1)
ax[1].set_title("d",fontsize=18)

ax[2].hist(r2_DF["P0_mean"],color="black",alpha=0.4)
ax[2].axvline(x=r2_DF["P0_mean"][0],color='black',linestyle='--')
ax[2].set_xlim(r2_DF["P0_mean"].mean()-0.1, r2_DF["P0_mean"].mean()+0.1)
ax[2].set_title("$P_{0}$",fontsize=18)

fig.supxlabel("coefficient of determination ($R^{2}$)",fontsize=15,y=0.08)
fig.supylabel("Frequency")
fig.savefig(path+"/results/ChanpionData/evaluation_MICE.pdf",)