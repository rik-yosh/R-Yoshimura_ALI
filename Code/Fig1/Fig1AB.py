print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import shap

os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 13

def DrawROC(X,y,model,n_cv=5,sm_k=5,figsize=(5,5),random_state=42,prex_axTitle="AUC",fileName="default.pdf",chance=True):
    # 予測精度を視覚的に評価するために、ROCカーブをプロットする
    cv = StratifiedKFold(n_splits=n_cv)
    figsize=(6,6)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    sm = SMOTE(k_neighbors=sm_k)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (train, test) in enumerate(cv.split(X, y)):
        X_train = X.iloc[train,:].reset_index(drop=True)
        y_train = y[train].reset_index(drop=True)
        X_test = X.iloc[test,:].reset_index(drop=True)
        y_test = y[test].reset_index(drop=True)

        # SMOTEを適用（trainだけ）
        kX_train, ky_train = sm.fit_resample(X_train, y_train)

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
    plt.close()
    fig, ax = plt.subplots(figsize=figsize)
    if chance:
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

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

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )
    ax.set_title(""+prex_axTitle+": %0.2f $\pm$ %0.2f" % (mean_auc, std_auc),fontsize=22)
    ax.legend(fontsize=16)
    # ax.legend(loc="lower right")
    plt.axis('square')
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig(path+fileName, bbox_inches="tight",dpi=200)
    plt.close()

    return mean_auc,std_auc

def shapBeeWarmplot(model,X,y,figsize=(10,5),max_display=10,fileName="default.pdf"):
    
    oriName = ["Age","Encephalopathy_grade_d0","MELDscore","WBC","Fib","ATIII","APTT","ALP","gGTP","Che","CRP","Ferritin","IgG","IgA","IgM","IgE","AFP","PIVKA","sIL2R","ALT_under200_1","ALTdevideLDH_new","TDratio_bil","Sex_M","diagnosis_AcuteHepatitis","diagnosis_AcuteHepatitis_non_coma","diagnosis_AcuteLiverFailure_coma","diagnosis_AcuteLiverFailure_non_coma","diagnosis_LOHF","diagnosis_SubacuteLiverFailure_coma","LiverAtrophy_d0_Assessor_A_..","LiverAtrophy_d0_Assessor_A_.","LiverAtrophy_d0_Assessor_B_.","LiverAtrophy_d0_Assessor_B_v","ALTdevideLDH_new_under1.5_Y","FDP_more_5_low","Plt_d0","PT_percentage_d0","Alb_d0","BUN_d0","Cre_d0","AST_d0","ALT_d0","LDH_d0","NH3_d0"]
    repName = ["Age","EAHE","MELD","WBC","Fib","ATIII","APTT","ALP","gGTP","Che","CRP","Ferritin","IgG","IgA","IgM","IgE","AFP","PIVKA","sIL2R","ALT<200","ALT/LDH","D/T-bil","Male","AH","AH (non coma)","ALF","ALF (non coma)","LOHF","SHF (coma)","A(A) ++","NA(A)","NA(B)","A(B) v","ALT/LDH<1.5","FDP<5","Plt","PT","Alb","BUN","Cre","AST","ALT","LDH","NH3"]
    X = X.rename(columns=dict(zip(oriName,repName)))
    
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print(np.array(shap_values).shape)
    print(np.array(shap_values[1]).shape)

    plt.figure(figsize=figsize)
    shap.summary_plot(shap_values[1],plot_size=figsize,features=X, show=False, color_bar=True, max_display=max_display)
    fig, ax = plt.gcf(), plt.gca()
    fig.set_size_inches(figsize)

    ax.tick_params(labelsize=15)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=15)
    # Get colorbar
    cb_ax = fig.axes[1] 
    # Modifying color bar parameters
    cb_ax.tick_params(labelsize=15)
    cb_ax.set_ylabel("Feature value", fontsize=15)
    

    plt.savefig(path+fileName, bbox_inches="tight",dpi=200)
    plt.close()
    shapvalDF = pd.DataFrame({
        "Feature":X.columns,
        "Value":np.abs(shap_values[1]).mean(axis=0)
    })

    shapvalDF = shapvalDF.sort_values("Value",ascending=False)
    return shapvalDF


if __name__ == "__main__":
    #---------------------------#
    # new dataset
    #---------------------------#
    print("___ split columns each date and treatment ___")
    # データを日数の段階ごとに分けていく
    Nd7_data=["Plt_d7","PT_percentage_d7","Alb_d7","BUN_d7","Cre_d7","AST_d7","ALT_d7","LDH_d7","TDratio_d7"]
    print("___ data loading ___")

    # Data loading
    data = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")

    X = data.loc[:,Nd7_data]
    y = data["InternalMedicineSurvival"].reset_index(drop=True)
    seed = 1234

    rfc = RandomForestClassifier(random_state=seed,n_estimators=500)
    mean_auc,std_auc = DrawROC(X,y=y,model=rfc,n_cv=5,sm_k=5,random_state=seed,figsize=(5,5),fileName="/Output/Fig1/Fig1A.png")
    X_shap = X.rename(columns=dict(zip(Nd7_data,["Plt","PT","Alb","BUN","Cre","AST","ALT","LDH","D/T-bil"])))
    
    shapvalDF = shapBeeWarmplot(model=rfc,X=X_shap,y=y,figsize=(10,5),fileName='/Output/Fig1/Fig1B.png')
