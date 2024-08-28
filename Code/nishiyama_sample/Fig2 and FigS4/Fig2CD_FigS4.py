print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from onAdmission_d7_Blood import DrawROC
from sklearn.inspection import permutation_importance
import shap

path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16


def shapBeeWarmplot(model,X,y,figsize=(10,5),max_display=10,colorBar=True,fileName="default.pdf"):
    # prex = [a.rsplit("_",1)[0] for a in X.columns.tolist()]
    # X.columns = prex
    # X = X.rename(columns={"TDratio":"D-bil/T-bil"})
    oriName = ['Age','Encephalopathy_grade_d0','MELDscore','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex_M','diagnosis_ALF_subacute','diagnosis_ALF_without_coma','diagnosis_ALI','diagnosis_LOHF','LiverAtrophy_.','ALTdevideLDH_new_under1.5_N','Plt_d0','PT_percentage_d0','Alb_d0','BUN_d0','Cre_d0','AST_d0','ALT_d0','LDH_d0','NH3_d0','TDratio_d0']
    repName = ['Age','HE','MELD','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex M','ALF SA','ALF NC','ALI','LOHF','No LA','ALT/LDH<1.5','Plt','PT%','Alb','BUN','Cre','AST','ALT','LDH','NH3','D/T-bil']
    X = X.rename(columns=dict(zip(oriName,repName)))

    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if colorBar:
        shap.summary_plot(shap_values[1],plot_size=figsize,features=X, show=False, color_bar=True, max_display=max_display)
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(figsize)

        ax.set_xlim(-0.15,0.25)
        ax.tick_params(labelsize=15)
        ax.set_xlabel("SHAP value", fontsize=15)

        # Get colorbar
        cb_ax = fig.axes[1] 

        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_ylabel("Feature value", fontsize=15)
    else:
        shap.summary_plot(shap_values[1],plot_size=figsize,features=X, show=False, color_bar=False, max_display=max_display)
        fig, ax = plt.gcf(), plt.gca()
        fig.set_size_inches(figsize)

        ax.set_xlim(-0.15,0.25)
        ax.tick_params(labelsize=15)
        ax.set_xlabel("SHAP value", fontsize=15)

    plt.savefig(path+fileName, bbox_inches="tight")
    plt.close()
    shapvalDF = pd.DataFrame({
        "Feature":X.columns,
        "Value":np.abs(shap_values[1]).mean(axis=0)
    })
    shapvalDF = shapvalDF.sort_values("Value",ascending=False)
    return shapvalDF
#---------------------------#
# new dataset
#---------------------------#
print("___ split columns each date and treatment ___")
# データを日数の段階ごとに分けていく
NgdP0 = ['g_mean', 'd_mean', 'P0_mean']
Nclust = ['Clustering']

Nlife = ['InternalMedicineSurvival']
Nd0_data = ['Age','Encephalopathy_grade_d0','MELDscore','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex_M','diagnosis_ALF_subacute','diagnosis_ALF_without_coma','diagnosis_ALI','diagnosis_LOHF','LiverAtrophy_.','ALTdevideLDH_new_under1.5_N','Plt_d0','PT_percentage_d0','Alb_d0','BUN_d0','Cre_d0','AST_d0','ALT_d0','LDH_d0','NH3_d0','TDratio_d0']
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

print("___ data loading ___")
ClustData = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv").drop([223],axis=0).reset_index(drop=True)["dtw_6"]
# Clustering labelの追加
data = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_median.csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={'TDratio_bil':'TDratio_d0'}).drop([223],axis=0).reset_index(drop=True)
Labels=pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv").drop([223],axis=0).reset_index(drop=True)
Life = data["InternalMedicineSurvival"].astype(bool)

rmIndex = [i for i,x in enumerate(Labels[["dtw_6"]].values) if x[0] not in ["G1","G2"]]
data = data.iloc[rmIndex,:].reset_index(drop=True)
Labels = Labels.iloc[rmIndex,:].reset_index(drop=True)

X_all = data.loc[:,Nd0_data+Nd0_med+Nd1_data+Nd1_med+Nd2_data+Nd2_med+Nd3_data+Nd3_med+Nd7_data+Nd7_med+Ndlast].reset_index(drop=True)
y_all = pd.get_dummies(Labels)
y_all.columns = ["G3","G4","G5","G6"]
y_all["G3+G4"] = (y_all["G3"]==1)|(y_all["G4"]==1)


shapCol = []
rfc = RandomForestClassifier(random_state=123,n_estimators=500)
# for Gx in y_all.columns:
#     print("loop: "+Gx+"")
#     y = y_all[[Gx]].to_numpy().squeeze()
#     X = X_all.drop(["PT_percentage_d0","PT_percentage_d1","PT_percentage_d2","PT_percentage_d3","PT_percentage_d7"],axis=1)
#     DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/results/ChanpionData/GroupPrediction/"+Gx+"_alldataPrediction_ROC.pdf",chance=False)
#     shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(5,5),max_display=10,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_alldataPrediction_shap.pdf')
#     shapCol.extend(shapvalDF.iloc[0:10,0].to_list())


perm_imp_DFs = pd.DataFrame(index=[],columns=[])
X = data.loc[:,Nd0_data].drop(["MELDscore"],axis=1).reset_index(drop=True)

rfc = RandomForestClassifier(random_state=123,n_estimators=500)
shapCol_d0 = []
for Gx in y_all.columns:
    print("loop: "+Gx+"")
    y = pd.Series(y_all[[Gx]].to_numpy().squeeze())

    rfc.fit(X, y)
    # result = permutation_importance(rfc, X, y, n_repeats=20, random_state=1)
    # perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)
    # perm_imp_df=perm_imp_df[perm_imp_df['importances_mean']!=0].sort_values(by="importances_mean",ascending=False)
    # print(perm_imp_df.index)
    # # factors = ['Anticoagulation_Did', 'Plt_d0', 'gGTP','ALT_d0','PT_percentage_d0', 'Che','ATIII', 'NH3_d0']
    # # perm_imp_df = perm_imp_df.loc[factors,:]
    # perm_imp_df["Clustering"] = Gx
    # perm_imp_DFs = pd.concat([perm_imp_DFs,perm_imp_df],axis=0)

    # fig, axes = plt.subplots(figsize=(8,6))
    # perm_imp_df.importances_mean.plot.barh(ax=axes,color="black")
    # axes.set(
    #     title=""+Gx+": Permutation Importance",
    #     xlabel="Accuracy difference from original model",
    #     xlim=(0,0.035)
    # )
    # plt.savefig(path+'/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_permImp.pdf', bbox_inches="tight")
    # plt.close()
    if Gx == "G6":
        plt.rcParams["font.size"] = 16
        DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/results/ChanpionData/GroupPrediction/"+Gx+"_d0Prediction_ROC.pdf",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap.pdf')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap_alldisplay.pdf')
    elif Gx == "G3+G4":
        plt.rcParams["font.size"] = 20
        DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/results/ChanpionData/GroupPrediction/"+Gx+"_d0Prediction_ROC.pdf",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap.pdf')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap_alldisplay.pdf')
    else:
        plt.rcParams["font.size"] = 16
        DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/results/ChanpionData/GroupPrediction/"+Gx+"_d0Prediction_ROC.pdf",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,colorBar=False,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap.pdf')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,colorBar=False,fileName='/results/ChanpionData/GroupPrediction/'+Gx+'_d0Prediction_shap_alldisplay.pdf')
print(list(set(shapCol_d0)))

# # "Clustering"の種類ごとにデータをフィルタリングしてプロット
# clustering_types = perm_imp_DFs['Clustering'].unique()

# fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # 1行4列のサブプロット
# for i, clustering_type in enumerate(clustering_types):
#     ax = axes[i]
#     subset = perm_imp_DFs[perm_imp_DFs['Clustering'] == clustering_type]
    
#     prex = [a.split("_")[0] for a in subset.index.tolist()]
    
#     subset.index = prex
#     subset.rename(index={"Anticoagulation":"ACG"})
    
#     ax.barh(subset.index, subset['importances_mean'], color='black')
#     ax.set_xlim(0, 0.015)
#     ax.set_title(clustering_type,fontsize=15)

#     # x軸目盛りのサイズを統一
#     ax.tick_params(axis='x', labelsize=13)
#     ax.tick_params(axis='y', labelsize=14)


#     # y軸のメモリは一番左端のみ残して、他は削除
#     if i > 0:
#         ax.set_yticklabels(["","","","","","","",""])
    
#     fig.supxlabel("Accuracy difference from original model",size=14)

# plt.tight_layout()
# plt.savefig(path+'/results/ChanpionData/GroupPrediction/permImp_d0Prediction.pdf',bbox_inches="tight")