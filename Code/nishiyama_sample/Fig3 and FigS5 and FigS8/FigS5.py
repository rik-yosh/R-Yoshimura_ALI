print("___ Import Module ___")
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import math
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score

if os.path.exists('./1_ALF_new'):
    os.chdir('./1_ALF_new')
    path = os.getcwd()
else:
    path = os.getcwd()
path = os.getcwd()
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "sans-serif"

f = open(path+'/data/ChanpionData/logiR_thresh.txt',"rb")
PT_thresh = pickle.load(f)

def PTcalculator(g,d,PT0,timeList):
    PT_list = []
    for t in timeList:
        PT_value = g/d - (g/d - PT0)*math.exp(-d*t)
        PT_list.append(PT_value)
    return PT_list

# PTの実際の値と、推定値（monolix）と、予測値（RF）のを使って決定係数を計算する
def get_allPat(y,y_pred,X_PTdat_L):
    allPat = pd.DataFrame()
    for i in range(0,len(y)):
        PTest =PTcalculator(y.iloc[i,0],y.iloc[i,1],y.iloc[i,2],[0,1,2,3,7])
        PTpred =PTcalculator(y_pred.iloc[i,0],y_pred.iloc[i,1],y_pred.iloc[i,2],[0,1,2,3,7])
        PTtrue = X_PTdat_L[X_PTdat_L['id_again']==i+1]['PT_percentage']
        onePat = pd.DataFrame({'id_again':np.repeat(i+1,5),'Date':[0,1,2,3,7],'PT_true':PTtrue,'PT_est':PTest,'PT_pred':PTpred})
        allPat = pd.concat([allPat,onePat],axis=0)

    r2_TE = r2_score(allPat['PT_true'],allPat['PT_est'])
    r2_TP = r2_score(allPat['PT_true'],allPat['PT_pred'])
    r2_EP = r2_score(allPat['PT_est'],allPat['PT_pred'])

    print("R2(true-est): "+str(round(r2_TE,4)))
    print("R2(true-pred): "+str(round(r2_TP,4)))
    print("R2(est-pred): "+str(round(r2_EP,4)))
    return allPat

print("___ Define Function ___")
def pred_plot(plotDataFrame=None, colorNumdata=None, filename=None,allPat=None, plot_real=True, plot_est=True, plot_pred=True, plot_interval=True,thresh=False,actcolor="gray"):

    fig, axes = plt.subplots(
        20,16,
        sharex=True,
        sharey=True,
        figsize=(16,20),
        dpi=200
    )
    # color
    plot_num = [plot_real, plot_est, plot_pred, plot_interval]

    for a, id in enumerate(set(plotDataFrame['id_again'])):
        dataline =  plotDataFrame[plotDataFrame['id_again'] == id]
        dataPT = allPat[allPat['id_again']== id]
        # 12で割った商と余りを出す
        row,col = a // 16, a % 16
        ax = axes[row,col]

        # sample毎に描画
        if plot_pred:
            ax.plot(dataline['time'],dataline['PT_pred'],
                color=cm.Pastel2(colorNumdata[a]),
                label="RF prediction",
                lw=3,
                zorder=1,
                alpha=0.8
            )
        if plot_interval:
            ax.fill_between(dataline['time'],dataline['err_down'],dataline['err_up'],
                color=cm.Pastel2(colorNumdata[a]),
                alpha=0.2,
                label="Prediction interval",
                zorder=0
            )
        if plot_est:
            ax.plot(dataline['time'],dataline['PT_est'],
                color="gray",
                label="Model fitting",
                lw=3,
                zorder=2,
                alpha=0.8
            )
        if plot_real:
            ax.scatter(dataPT['Date'],dataPT['PT_true'],
                marker="o",
                c=actcolor,
                s=40,
                edgecolors="gray",
                alpha=0.8,
                label="Data"
            )
        
        
        if a == 304:
            ax.legend(bbox_to_anchor=(-0.2,-.85), ncol=len(plot_num), loc='upper left', frameon=False, borderaxespad=1,fontsize=13)
        if thresh == True:
            ax.axhline(PT_thresh, linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
        # 軸ラベル
        # ax.set_title(id)
        ax.set_xlim(-1,8)
        ax.set_ylim(0,150)
        ax.set_xticks([0,1,2,3,7])
        ax.set_yticks([0,50,100,150])
        # x軸目盛りのサイズを統一
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        #ax.legend(fontsize=12)
    fig.supxlabel("Days post-admission",fontsize=14,y=0.08)
    fig.supylabel("Prothrombin time activity percentage (%)",fontsize=14,x=0.08)
    fig.savefig(path+"/results/"+filename, bbox_inches='tight')
    plt.close()
    return fig

if __name__ == "__main__":
    print("___ Data loading ___")
    # 代表として取ってくる患者の症例数を決める
    n_pat = 319
    ClustData = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv")
    ClustData = ClustData.iloc[0:n_pat]
    
    # Clustering labelの追加
    y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']].iloc[0:n_pat,:]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_1.csv").drop(["Unnamed: 0"],axis=1).iloc[0:n_pat,:]
    Life = X["InternalMedicineSurvival"].reset_index(drop=True)
    #X = X.loc[:,["Age","MELD.score","WBC","APTT","ALP","gGTP","Che","CRP","Ferritin","AFP","ALTunder200.1","ALTdevideLDH_new","Sex_M","diagnosis_AcuteHepatitis","diagnosis_AcuteHepatitis_non_coma","diagnosis_AcuteLiverFailure_coma","diagnosis_AcuteLiverFailure_non_coma","diagnosis_LOHF","diagnosis_SubacuteLiverFailure_coma","Etiology_detail_AIH","Etiology_detail_AOSD","Etiology_detail_Acute.HCV","Etiology_detail_Alcohol","Etiology_detail_Alcohol.Or.DIC","Etiology_detail_Amyloidosis","Etiology_detail_Aucte.AIH","Etiology_detail_Budd.Chiari","Etiology_detail_CMV","Etiology_detail_DILI","Etiology_detail_EBV","Etiology_detail_HAV","Etiology_detail_HBV","Etiology_detail_HCV","Etiology_detail_HSV","Etiology_detail_Ischemic","Etiology_detail_Wilson","Etiology_detail_acetaminophen","Etiology_detail_iron.addiction","Etiology_detail_ischemic.hepatitis","Etiology_detail_unknown","LiverAtrophy_d0_Assessor_A_..","LiverAtrophy_d0_Assessor_A_.","LiverAtrophy_d0_Assessor_B_.","LiverAtrophy_d0_Assessor_B_v","ALTdevideLDH_new_under_1.5_Y","FDP_over5_low","Plt_d0","FDP_d0","PT_percentage_d0","Alb_d0","BUN_d0","Cre_d0","AST_d0","ALT_d0","LDH_d0","NH3_d0","TDratio_d0"]]

    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]

    plotDataFrame_tree = pd.read_csv(path+"/data/ChanpionData/addinfo/plotDataFrame_tree_0.csv")
    plotDataFrame_tree = plotDataFrame_tree.iloc[0:n_pat*100,:]
    
    y_pred = pd.read_csv(path+"/data/ChanpionData/addinfo/y_pred_LOO_RF_0.csv").iloc[0:n_pat,:]

    allPat = get_allPat(y,y_pred,X_PTdat_L)
    allPat = allPat.iloc[0:n_pat*5,:]

    print("___ Plot tree per cluster ___")
    logiR = LogisticRegression(random_state=0#,penalty="elasticnet",solver="saga",l1_ratio=0.8
                           )
    logiR.fit(X_PTdat[["PT_percentage_d7"]],Life)
    d7Data = plotDataFrame_tree[plotDataFrame_tree["time"]==7].rename(columns={"PT_pred":"PT_percentage_d7"}).reset_index(drop=True)
    print(d7Data)
    print(logiR.classes_)
    print(logiR.predict_proba(X_PTdat[["PT_percentage_d7"]]))

    print("Logi: Tru val, ROC_AUC", roc_auc_score(Life,logiR.predict_proba(X_PTdat[["PT_percentage_d7"]])[:,1]))
    print("Logi: Pred val, ROC_AUC", roc_auc_score(Life,logiR.predict_proba(d7Data[["PT_percentage_d7"]].astype(float))[:,1]))

    pred_d7Data = d7Data[["PT_percentage_d7"]] <= 51.29
    true_d7data = X_PTdat[["PT_percentage_d7"]] <= 51.29

    print("Tru val, ROC_AUC", roc_auc_score(Life,true_d7data))
    print("Pred val, ROC_AUC", roc_auc_score(Life,pred_d7Data))    

    print("___ Run main ___")
    # 元id 110を消す
    plotDataFrame_tree = plotDataFrame_tree.drop(plotDataFrame_tree[plotDataFrame_tree["id_again"]==224].index,axis=0).reset_index(drop=True)
    Life = Life.drop([223],axis=0).reset_index(drop=True)
    allPat = allPat.drop(allPat[allPat["id_again"]==224].index,axis=0).reset_index(drop=True)
    
    pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_0_all.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True)
    # pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_0_all_thresh.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True,thresh=True)

    print("___ Run until 2 days ___")

    y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']].iloc[0:n_pat,:]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_1.csv").drop(["Unnamed: 0"],axis=1).iloc[0:n_pat,:]
    Life = X["InternalMedicineSurvival"].reset_index(drop=True)

    # PTの実際のデータ作成
    plotDataFrame_tree = pd.read_csv(path+"/data/ChanpionData/addinfo/plotDataFrame_tree_2.csv").reset_index(drop=True)
    plotDataFrame_tree = plotDataFrame_tree.iloc[0:n_pat*100,:]
    
    y_pred = pd.read_csv(path+"/data/ChanpionData/addinfo/y_pred_LOO_RF_2.csv").iloc[0:n_pat,:]

    allPat = get_allPat(y,y_pred,X_PTdat_L)
    allPat = allPat.iloc[0:n_pat*5,:]
    
    # 元id 110を消す
    plotDataFrame_tree = plotDataFrame_tree.drop(plotDataFrame_tree[plotDataFrame_tree["id_again"]==224].index,axis=0).reset_index(drop=True)
    Life = Life.drop([223],axis=0).reset_index(drop=True)
    allPat = allPat.drop(allPat[allPat["id_again"]==224].index,axis=0).reset_index(drop=True)

    pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_2_all.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True)
    # pred_plot(plotDataFrame=plotDataFrame_tree, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_2_all_thresh.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True,thresh=True)
    # pred_plot(plotDataFrame=plotDataFrame_tree, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_0_black.png",plot_real=True, plot_est=False, plot_pred=False, plot_interval=False)
    # pred_plot(plotDataFrame=plotDataFrame_tree, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_0_Blue.png",plot_real=True, plot_est=True, plot_pred=False, plot_interval=False)
    # pred_plot(plotDataFrame=plotDataFrame_tree, allPat=allPat,filename="ChanpionData/Plot_tsPred/LOOCV_gd_PredEstData_plot_tree_0_Green.png",plot_real=True, plot_est=False, plot_pred=True, plot_interval=True)