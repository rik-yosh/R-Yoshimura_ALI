print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.inspection import permutation_importance
import shap

os.chdir('./1_ALF_new')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 13


def DrawROC(X,y,model,n_cv,sm_k=4,figsize=(5,5),prex_axTitle="AUC",fileName="default.pdf",chance=True):
    # 予測精度を視覚的に評価するために、ROCカーブをプロットする
    cv = StratifiedKFold(n_splits=n_cv,shuffle=True,random_state=np.random.randint(0,1000))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    sm = SMOTE(k_neighbors=sm_k,random_state=np.random.randint(0,1000))

    fig, ax = plt.subplots(figsize=figsize)
    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train,:].reset_index(drop=True)
        y_train = y[train].reset_index(drop=True)
        X_test = X.iloc[test,:].reset_index(drop=True)
        y_test = y[test].reset_index(drop=True)

        # inner loopに対してSMOTEを適用（つまりtrainだけ）
        kX_train, ky_train = sm.fit_resample(X_train, y_train)
        # print(len(ky_train[ky_train==1]),len(ky_train[ky_train==0]),kX_train.shape,ky_train.shape)
        model.fit(kX_train, ky_train)
        viz = RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    print(aucs)
    plt.close()
    fig, ax = plt.subplots(figsize=figsize)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="black",
        label="Mean ROC",
        lw=2,
        alpha=0.8,
    ) 

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    if chance:
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )
    ax.set_title(""+prex_axTitle+": %0.2f $\pm$ %0.2f" % (mean_auc, std_auc),fontsize=19)
    ax.legend(fontsize=16, bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1, frameon=False)
    # ax.legend(loc="lower right")
    plt.axis('square')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig(path+fileName, bbox_inches="tight")
    plt.close()

    return mean_auc,std_auc

def permImpPlot(model,X,y,n_rep,seed,figsize=(10,5),fileName="default.pdf"):
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=n_rep, random_state=seed)
    perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)
    perm_imp_df = perm_imp_df[perm_imp_df["importances_mean"]!=0]
    perm_imp_df=perm_imp_df.sort_values("importances_mean", ascending=True)

    prex = [a.rsplit("_",1)[0] for a in perm_imp_df.index.tolist()]
    perm_imp_df.index = prex
    perm_imp_df = perm_imp_df.rename(index={"TDratio":"D-bil/T-bil"})

    fig, axes = plt.subplots(figsize=figsize)
    axes.barh(perm_imp_df.index,perm_imp_df.importances_mean, color="black")
    axes.set(
        title="Permutation Importance",
        xlabel="Accuracy difference from original model"
    )

    fig.savefig(path+fileName, bbox_inches="tight")

    plt.close()
    return perm_imp_df

def shapBeeWarmplot(model,X,y,figsize=(10,5),colorBar=True,max_display=10,fileName="default.pdf"):
    # prex = [a.rsplit("_",1)[0] for a in X.columns.tolist()]
    # X.columns = prex
    # X = X.rename(columns={"TDratio":"D-bil/T-bil"})
    oriName = ['Age','Encephalopathy_grade_d0','MELDscore','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex_M','diagnosis_ALF_subacute','diagnosis_ALF_without_coma','diagnosis_ALI','diagnosis_LOHF','LiverAtrophy_.','ALTdevideLDH_new_under1.5_N','Plt_d0','PT_percentage_d0','Alb_d0','BUN_d0','Cre_d0','AST_d0','ALT_d0','LDH_d0','NH3_d0','TDratio_d0','PE_NotDone','TASIT_NotDone','peripheral_pulse_NotDone','Anticoagulation_NotDone','CHDF_NotDone','VasopressorUse_1.0','Plt_d1','PT_percentage_d1','Alb_d1','BUN_d1','Cre_d1','AST_d1','ALT_d1','LDH_d1','NH3_d1','TDratio_d1','Yesterday_FFPprescribing_U_d1','Yesterday_rhTM_NotDone_d1','Plt_d2','PT_percentage_d2','Alb_d2','BUN_d2','Cre_d2','AST_d2','ALT_d2','LDH_d2','NH3_d2','TDratio_d2']
    repName = ['Age','HE','MELD','WBC','Fib','ATIII','APTT','ALP','gGTP','Che','CRP','Ferritin','IgG','IgA','IgM','AFP','PIVKA','sIL2R','Sex M','ALF SA','ALF NC','ALI','LOHF','No LA','ALT/LDH<1.5','Plt','PT%','Alb','BUN','Cre','AST','ALT','LDH','NH3','D/T-bil','No PE','No TASIT','No PAP','No AC','No CHDF','Vasopressor','Plt d1','PT% d1','Alb d1','BUN d1','Cre d1','AST d1','ALT d1','LDH d1','NH3 d1','D/T-bil d1',"FFP d0",'No rhTM d0','Plt d2','PT% d2','Alb d2','BUN d2','Cre d2','AST d2','ALT d2','LDH d2','NH3 d2','D/T-bil d2']

    X = X.rename(columns=dict(zip(oriName,repName)))

    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=figsize)
    if len(shap_values[1]) == 319:
        shap_values = shap_values[1]
    print(shap_values.shape)
    if colorBar:
        shap.summary_plot(shap_values,plot_size=figsize,features=X, show=False, color_bar=True, max_display=max_display)
        fig, ax = plt.gcf(), plt.gca()
        # Get colorbar
        cb_ax = fig.axes[1] 

        # Modifying color bar parameters
        cb_ax.tick_params(labelsize=15)
        cb_ax.set_ylabel("Feature value", fontsize=15)

    else:
        shap.summary_plot(shap_values,plot_size=figsize,features=X, show=False, color_bar=False, max_display=max_display)
        fig, ax = plt.gcf(), plt.gca()
    
    fig.set_size_inches(figsize)    
    ax.tick_params(labelsize=15)
    ax.set_xlabel("SHAP value", fontsize=15)
    

    plt.savefig(path+fileName, bbox_inches="tight")
    plt.close()
    shapvalDF = pd.DataFrame({
        "Feature":X.columns,
        "Value":np.abs(shap_values).mean(axis=0)
    })
    shapvalDF = shapvalDF.sort_values("Value",ascending=False)
    return shapvalDF


if __name__ == "__main__":
    #---------------------------#
    # new dataset
    #---------------------------#
    print("___ split columns each date and treatment ___")
    # データを日数の段階ごとに分けていく
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

    print("___ data loading ___")
    data = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_median.csv").drop([223],axis=0).reset_index(drop=True)

    Nd0_Bdata = ['Plt_d0','PT_percentage_d0','Alb_d0','BUN_d0','Cre_d0','AST_d0','ALT_d0','LDH_d0','NH3_d0','TDratio_d0']

    X = data.loc[:,Nd7_data]
    LifeData = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_d0all.csv").drop([223],axis=0).reset_index(drop=True)
    y = LifeData["InternalMedicineSurvival"].reset_index(drop=True)
    rfc = RandomForestClassifier(random_state=0,n_estimators=500)
    mean_auc,std_auc = DrawROC(X,y=y,model=rfc,n_cv=5,figsize=(5,5),fileName="/results/ChanpionData/onAdmission_d7Blood_ROCAUC.pdf")

    X_shap = X.rename(columns=dict(zip(Nd7_data,["Plt","PT%","Alb","BUN","Cre","AST","ALT","LDH","D/T-bil"])))

    # perm_imp_df = permImpPlot(model=rfc,X=X,y=y,n_rep=10,seed=0,figsize=(10,5),fileName='/results/ChanpionData/onAdmission_d7Blood_permImp.pdf')
    shapvalDF = shapBeeWarmplot(model=rfc,X=X_shap,y=y,figsize=(10,5),fileName='/results/ChanpionData/onAdmission_d7Blood_shap.pdf')