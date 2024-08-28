print("___ Import Module ___")
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from Fig3.gdP0prediction import get_allPat, pred_plot

if os.path.exists('./R-Yoshimura_ALI'):
    os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.size"] = 15
plt.rcParams["font.family"] = "sans-serif"


if __name__ == "__main__":
    print("___ Data loading ___")
    # 代表として取ってくる患者の症例数を決める
    n_pat = 319
    ClustData = pd.read_csv(path+"/Data/DTWclustLabel.csv")
    ClustData = ClustData.iloc[0:n_pat]
    
    # Clustering labelの追加
    y = pd.read_csv(path+'/monolix/PT_NLMEMfit/IndividualParameters/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']].iloc[0:n_pat,:]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv").drop(["Unnamed: 0"],axis=1).iloc[0:n_pat,:]
    Life = X["InternalMedicineSurvival"].reset_index(drop=True)
    #X = X.loc[:,["Age","MELD.score","WBC","APTT","ALP","gGTP","Che","CRP","Ferritin","AFP","ALTunder200.1","ALTdevideLDH_new","Sex_M","diagnosis_AcuteHepatitis","diagnosis_AcuteHepatitis_non_coma","diagnosis_AcuteLiverFailure_coma","diagnosis_AcuteLiverFailure_non_coma","diagnosis_LOHF","diagnosis_SubacuteLiverFailure_coma","Etiology_detail_AIH","Etiology_detail_AOSD","Etiology_detail_Acute.HCV","Etiology_detail_Alcohol","Etiology_detail_Alcohol.Or.DIC","Etiology_detail_Amyloidosis","Etiology_detail_Aucte.AIH","Etiology_detail_Budd.Chiari","Etiology_detail_CMV","Etiology_detail_DILI","Etiology_detail_EBV","Etiology_detail_HAV","Etiology_detail_HBV","Etiology_detail_HCV","Etiology_detail_HSV","Etiology_detail_Ischemic","Etiology_detail_Wilson","Etiology_detail_acetaminophen","Etiology_detail_iron.addiction","Etiology_detail_ischemic.hepatitis","Etiology_detail_unknown","LiverAtrophy_d0_Assessor_A_..","LiverAtrophy_d0_Assessor_A_.","LiverAtrophy_d0_Assessor_B_.","LiverAtrophy_d0_Assessor_B_v","ALTdevideLDH_new_under_1.5_Y","FDP_over5_low","Plt_d0","FDP_d0","PT_percentage_d0","Alb_d0","BUN_d0","Cre_d0","AST_d0","ALT_d0","LDH_d0","NH3_d0","TDratio_d0"]]

    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]

    plotDataFrame_tree = pd.read_csv(path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_0.csv")
    plotDataFrame_tree = plotDataFrame_tree.iloc[0:n_pat*100,:]

    y_pred = pd.read_csv(path+"/Data/Fig3/predicted/y_pred_LOO_RF_0.csv").iloc[0:n_pat,:]

    allPat = get_allPat(y,y_pred,X_PTdat_L)
    allPat = allPat.iloc[0:n_pat*5,:]

    print("___ Run main ___")

    pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename=path+"/Output/Fig3/Fig3A.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True)
    pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename=path+"/Output/Fig3/Fig3B.png",plot_real=True, plot_est=True, plot_pred=False, plot_interval=False)


    print("___ Run until 2 days ___")
    
    y = pd.read_csv(path+'/monolix/PT_NLMEMfit/IndividualParameters/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv").drop(["Unnamed: 0"],axis=1).iloc[0:n_pat,:]
    Life = X["InternalMedicineSurvival"].reset_index(drop=True)

    plotDataFrame_tree = pd.read_csv(path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_2.csv").reset_index(drop=True)
    plotDataFrame_tree = plotDataFrame_tree.iloc[0:n_pat*100,:]
    
    y_pred = pd.read_csv(path+"/Data/Fig3/predicted/y_pred_LOO_RF_2.csv").iloc[0:n_pat,:]

    allPat = get_allPat(y,y_pred,X_PTdat_L)
    allPat = allPat.iloc[0:n_pat*5,:]

    pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename=path+"/Output/Fig3/EDFig3.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True)

    # print("___ Run until 1 days ___")

    # y = pd.read_csv(path+'/monolix/PT_NLMEMfit/IndividualParameters/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]
    # y["Clustering"] = ClustData
    # X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv").drop(["Unnamed: 0"],axis=1).iloc[0:n_pat,:]
    # Life = X["InternalMedicineSurvival"].reset_index(drop=True)

    # plotDataFrame_tree = pd.read_csv(path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_1.csv").reset_index(drop=True)
    # plotDataFrame_tree = plotDataFrame_tree.iloc[0:n_pat*100,:]
    
    # y_pred = pd.read_csv(path+"/Data/Fig3/predicted/y_pred_LOO_RF_1.csv").iloc[0:n_pat,:]

    # allPat = get_allPat(y,y_pred,X_PTdat_L)
    # allPat = allPat.iloc[0:n_pat*5,:]

    # pred_plot(plotDataFrame=plotDataFrame_tree, colorNumdata=Life, allPat=allPat,filename="/Output/Fig3/Fig3A.png",plot_real=True, plot_est=True, plot_pred=True, plot_interval=True)