print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.inspection import permutation_importance
# import shap

os.chdir('./1_ALF_new')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 16


def DrawROC(X,y,model,n_cv,prex_axTitle="AUC",fileName="default.pdf"):
    # 予測精度を視覚的に評価するために、ROCカーブをプロットする
    cv = StratifiedKFold(n_splits=n_cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6,6))
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X.iloc[train,:], y[train])
        viz = RocCurveDisplay.from_estimator(
            model,
            X.iloc[test,:],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="black",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
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

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=""+prex_axTitle+":%0.2f" % (mean_auc),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )
    ax.legend(fontsize=13)
    # ax.legend(loc="lower right")
    plt.axis('square')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig(path+fileName, bbox_inches="tight")
    plt.close()

    return mean_auc,std_auc

def permImpPlot(model,X,y,n_rep,seed,figsize=(12,6),fileName="default.pdf"):
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=n_rep, random_state=seed)
    perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)
    fig, axes = plt.subplots(figsize=figsize)
    perm_imp_df=perm_imp_df.sort_values("importances_mean", ascending=True)
    perm_imp_df.importances_mean.plot.barh(ax=axes,color="black")
    axes.set(
        title="Permutation Importance",
        xlabel="Accuracy difference from original model"
    )
    plt.savefig(path+fileName, bbox_inches="tight")
    plt.close()
    return perm_imp_df

# def shapBeeWarmplot(model,X,y,figsize=(12,6),max_display=10,fileName="default.pdf"):
#     model.fit(X, y)
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X)
#     plt.figure(figsize=figsize)
#     shap.summary_plot(shap_values[1],features=X, show=False,max_display=max_display)
#     plt.savefig(path+fileName, bbox_inches="tight")
#     plt.close()
#     shapvalDF = pd.DataFrame({
#         "Feature":X.columns,
#         "Value":np.abs(shap_values[1]).mean(axis=0)
#     })
#     shapvalDF = shapvalDF.sort_values("Value",ascending=False)
#     return shapvalDF



if __name__ == "__main__":
    #---------------------------#
    # new dataset
    #---------------------------#
    print("___ split columns each date and treatment ___")
    # データを日数の段階ごとに分けていく
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

    print("___ data loading ___")

    ClustData = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv")["dtw_6"]
    # Clustering labelの追加
    
    LifeData = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_d0all.csv")
    y = LifeData["InternalMedicineSurvival"].reset_index(drop=True)

    if os.path.isfile(path+"/data/ChanpionData/auc_list.txt"):
        f = open(path+"/data/ChanpionData/auc_list.txt","rb")
        auc_list = pickle.load(f)
    else:
        auc_list = []
        for i in range(1,51):
            print("data:",i)
            data = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_"+str(i)+".csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={"TDratio_bil":"TDratio_d0"})
        
            X = data.loc[:,Nd7_data]
            rfc = RandomForestClassifier(random_state=0,n_estimators=500)
            mean_auc,std_auc = DrawROC(X,y=y,model=rfc,n_cv=5,fileName="/results/ChanpionData/miceData/onAdmission_data_"+str(i)+".pdf")
            auc_list.append(mean_auc)
        
        f = open(path+"/data/ChanpionData/auc_list.txt", 'wb')
        pickle.dump(auc_list, f)

    print(auc_list[0])
    mean_mean = np.mean(auc_list)
    std_mean = np.std(auc_list)

    print("mean AUC: %0.2f $\pm$ %0.2f" % (round(mean_mean,2),round(std_mean,2)))

    plt.hist(auc_list,color="k",alpha=0.4#,rwidth=0.025
             )
    plt.axvline(x=auc_list[0],color='black',linestyle='--')
    plt.xlim(mean_mean-0.1, mean_mean+0.1,)
    # plt.title("mean AUC: %0.2f $\pm$ %0.2f" % (round(mean_mean,2),round(std_mean,2)))
    plt.xlabel("ROC AUC")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path+"/results/ChanpionData/Hist_onAdmission_Pred.pdf")
    # perm_imp_df = permImpPlot(model=rfc,X=X,y=y,n_rep=10,seed=0,figsize=(12,6),fileName='/results/ChanpionData/onAdmission_d7Blood_permImp.pdf')
    # shapvalDF = shapBeeWarmplot(model=rfc,X=X,y=y,figsize=(12,6),fileName='/results/ChanpionData/onAdmission_d7Blood_shap.pdf')