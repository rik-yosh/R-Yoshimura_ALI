print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
import shap

from Fig1.Fig1AB import DrawROC


if os.path.exists('./R-Yoshimura_ALI'):
    os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16


def shapBeeWarmplot(model,X,y,figsize=(10,5),max_display=10,colorBar=True,fileName="default.png"):
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

    plt.savefig(path+fileName, bbox_inches="tight",dpi=200)
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

print("___ data loading ___")
Labels = pd.read_csv(path+"/Data/DTWclustLabel.csv")["dtw_6"]
data = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={'TDratio_bil':'TDratio_d0'}).reset_index(drop=True)
Life = data["InternalMedicineSurvival"].astype(bool)

seed = 42


y_all = pd.get_dummies(Labels)
y_all.columns = ["G1","G2","G3","G4","G5","G6"]

perm_imp_DFs = pd.DataFrame(index=[],columns=[])
X = data.loc[:,Nd0_data].drop(["MELDscore"],axis=1)

rfc = RandomForestClassifier(random_state=seed,n_estimators=500)
shapCol_d0 = []
for Gx in y_all.columns:
    print("loop: "+Gx+"")
    y = pd.Series(y_all[[Gx]].to_numpy().squeeze())

    rfc.fit(X, y)
    if Gx == "G6":
        plt.rcParams["font.size"] = 16
        DrawROC(X,y=y,model=rfc,n_cv=5,sm_k=5,random_state=seed,prex_axTitle=Gx,fileName="/Output/Fig2/Fig2C_"+Gx+"_ROC.png",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap.png')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap_alldisplay.png')
    elif Gx == "G3+G4":
        plt.rcParams["font.size"] = 20
        DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/Output/Fig2/Fig2C_"+Gx+"_ROC.png",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap.png')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap_alldisplay.png')
    else:
        plt.rcParams["font.size"] = 16
        DrawROC(X,y=y,model=rfc,n_cv=4,prex_axTitle=Gx,fileName="/Output/Fig2/Fig2C_"+Gx+"_ROC.png",chance=False)
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,4),max_display=5,colorBar=False,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap.png')
        # print(shapvalDF.iloc[0:5,0])
        shapCol_d0.extend(shapvalDF.iloc[0:5,0].to_list())
        shapvalDF = shapBeeWarmplot(rfc,X,y,figsize=(4,10.5),max_display=50,colorBar=False,fileName='/Output/Fig2/Fig2D_'+Gx+'_shap_alldisplay.png')
print(list(set(shapCol_d0)))