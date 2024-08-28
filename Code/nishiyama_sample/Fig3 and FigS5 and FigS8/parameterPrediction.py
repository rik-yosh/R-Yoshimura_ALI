print("___ Import Module ___")
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from onAdmission_d7_Blood import shapBeeWarmplot

import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from tsPrediction import predPT_tsDF
from PTintegral import cul_area
from Plot_tsPred import get_allPat, pred_plot


path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"


# Function to get each tree prediction values
# https://blog.datadive.net/prediction-intervals-for-random-forests/
def pred_dist(model, X):
    X = X.reset_index(drop=True)
    x_reshape = np.array(list(X.iloc[0,:])).reshape(-1,1).T# predictにデータフレームを入れるために、np.arrayにしてreshapeして転置する必要がある

    every_paraDist = []# 各決定木が予測したp,d,P0のarrayを出力（これを経験分布として用いる）    
    for pred in model.estimators_:# 各決定木毎に予測値を出力
        every_paraDist.append(pred.predict(x_reshape)[0])

    np_every = np.array(every_paraDist)

    looTree_paraDist = []# 各決定木が予測したp,d,P0のarrayを出力（これを経験分布として用いる）
    for one_pred in every_paraDist:
        length =len(every_paraDist) -1
        g = (sum(np_every[:,0]) - one_pred[0])/length
        d = (sum(np_every[:,1]) - one_pred[1])/length
        P = (sum(np_every[:,2]) - one_pred[2])/length
        new_pred = [g,d,P]
        looTree_paraDist.append(new_pred)
    
    return every_paraDist, looTree_paraDist

def LOO_RF(X,y,n_tree,rf_seed,shapfilename="sample.pdf"):
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

        # 予測値の分布を取得する
        one_paraDist, one_looDist = pred_dist(rfr, X_test) # one_paraDist[0][0][0]（[0]:一人目の、[0]:一つ目のパラメータ（g_mean）の、[0]:一つ目の木の予測）
        DF_one_paraDist = pd.DataFrame(one_paraDist,columns=['g_mean','d_mean','P0_mean'])
        DF_one_paraDist.to_csv(path+'/data/Distribution/treeDistribution_patient_No'+str(test_index)+'.csv', index=False)
        # pd.DataFrame(one_looDist,columns=['g_mean','d_mean','P0_mean']).to_csv(path+'/data/Distribution/looDistribution_patient_No'+str(test_index)+'.csv', index=False)
        print(test_index)
    
    shapvalDF = shapBeeWarmplot(model=rfr,X=X,y=y,figsize=(6,10.5),max_display=5,fileName=shapfilename)
    
    return y_pred, shapvalDF

if __name__=="__main__":
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
    LifeData = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_d0all.csv")
    Life = LifeData["InternalMedicineSurvival"].reset_index(drop=True)

    # Clustering labelの追加
    ClustData = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv")["dtw_6"]

    y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_1.csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={"TDratio_bil":"TDratio_d0"})

    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]

    print("___ Calculate each data ___")
    rf_seed = 111
    n_tree=500

    # print("day0_data")
    # x_data = Nd0_data
    # SHAPname = "/results/ChanpionData/addinfo/Permutation_all_0.pdf"
    # y_pred_0, perm_imp_dfs_sort_0 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    # y_pred_0.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_0.csv', index=False)
    # perm_imp_dfs_sort_0.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_0.csv')
    # y_pred_0 = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_0.csv')
    # plotDataFrame_tree_0 = predPT_tsDF(y=y,y_pred=y_pred_0, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_0.csv")
    # allPat_0 = get_allPat(y,y_pred_0,X_PTdat_L)
    # pred_plot(plotDataFrame=plotDataFrame_tree_0,colorNumdata=Life,allPat=allPat_0,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_0.pdf")
    # area_0 = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_0[['g_mean','d_mean','P0_mean']])

    # print("day0_data + med")
    # x_data = x_data+Nd0_med
    # SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0m.pdf"
    # y_pred_0m, perm_imp_dfs_sort_0m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    # y_pred_0m.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_0m.csv', index=False)
    # perm_imp_dfs_sort_0m.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_0m.csv')
    # y_pred_0m = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_0m.csv')
    # plotDataFrame_tree_0m = predPT_tsDF(y=y,y_pred=y_pred_0m, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_0m.csv")
    # allPat_0m = get_allPat(y,y_pred_0m,X_PTdat_L)
    # pred_plot(plotDataFrame=plotDataFrame_tree_0m,colorNumdata=Life,allPat=allPat_0m,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_0m.pdf")
    # area_0m = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_0m[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data")
    # x_data = x_data+Nd1_data
    x_data = Nd0_data+Nd0_med+Nd1_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_1.pdf"
    y_pred_1, perm_imp_dfs_sort_1 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_1.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_1.csv', index=False)
    perm_imp_dfs_sort_1.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_1.csv')
    y_pred_1 = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_1.csv')
    plotDataFrame_tree_1 = predPT_tsDF(y=y,y_pred=y_pred_1, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_1.csv")
    allPat_1 = get_allPat(y,y_pred_1,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_1,colorNumdata=Life,allPat=allPat_1,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_1.pdf")
    area_1 = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_1[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med")
    x_data = x_data+Nd1_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_1m.pdf"
    y_pred_1m, perm_imp_dfs_sort_1m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_1m.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_1m.csv', index=False)
    perm_imp_dfs_sort_1m.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_1m.csv')
    y_pred_1m = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_1m.csv')
    plotDataFrame_tree_1m = predPT_tsDF(y=y,y_pred=y_pred_1m, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_1m.csv")
    allPat_1m = get_allPat(y,y_pred_1m,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_1m,colorNumdata=Life,allPat=allPat_1m,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_1m.pdf")
    area_1m = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_1m[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med ")
    x_data = x_data+Nd2_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2.pdf"
    y_pred_2, perm_imp_dfs_sort_2 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_2.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_2.csv', index=False)
    perm_imp_dfs_sort_2.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_2.csv')
    y_pred_2 = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_2.csv')
    plotDataFrame_tree_2 = predPT_tsDF(y=y,y_pred=y_pred_2, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_2.csv")
    allPat_2 = get_allPat(y,y_pred_2,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_2,colorNumdata=Life,allPat=allPat_2,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_2.pdf")
    area_2 = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_2[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med + day2_data + med")
    x_data = x_data+Nd2_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2m.pdf"
    y_pred_2m, perm_imp_dfs_sort_2m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_2m.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_2m.csv', index=False)
    perm_imp_dfs_sort_2m.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_2m.csv')
    y_pred_2m = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_2m.csv')
    plotDataFrame_tree_2m = predPT_tsDF(y=y,y_pred=y_pred_2m, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_2m.csv")
    allPat_2m = get_allPat(y,y_pred_2m,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_2m,colorNumdata=Life,allPat=allPat_2m,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_2m.pdf")
    area_2m = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_2m[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med + day2_data + med + day3_data")
    x_data = x_data+Nd3_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_3.pdf"
    y_pred_3, perm_imp_dfs_sort_3 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_3.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_3.csv', index=False)
    perm_imp_dfs_sort_3.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_3.csv')
    y_pred_3 = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_3.csv')
    plotDataFrame_tree_3 = predPT_tsDF(y=y,y_pred=y_pred_3, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_3.csv")
    allPat_3 = get_allPat(y,y_pred_3,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_3,colorNumdata=Life,allPat=allPat_3,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_3.pdf")
    area_3 = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_3[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med")
    x_data = x_data+Nd3_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_3m.pdf"
    y_pred_3m, perm_imp_dfs_sort_3m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_3m.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_3m.csv', index=False)
    perm_imp_dfs_sort_3m.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_3m.csv')
    y_pred_3m = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_3m.csv')
    plotDataFrame_tree_3m = predPT_tsDF(y=y,y_pred=y_pred_3m, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_3m.csv")
    allPat_3m = get_allPat(y,y_pred_3m,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_3m,colorNumdata=Life,allPat=allPat_3m,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_3m.pdf")
    area_3m = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_3m[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data")
    x_data = x_data+Nd7_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_7.pdf"
    y_pred_7, perm_imp_dfs_sort_7 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_7.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_7.csv', index=False)
    perm_imp_dfs_sort_7.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_7.csv')
    y_pred_7 = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_7.csv')
    plotDataFrame_tree_7 = predPT_tsDF(y=y,y_pred=y_pred_7, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_7.csv")
    allPat_7 = get_allPat(y,y_pred_7,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_7,colorNumdata=Life,allPat=allPat_7,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_7.pdf")
    area_7 = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_7[['g_mean','d_mean','P0_mean']])

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data + med")
    x_data = x_data+Nd7_med+Ndlast
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_7m.pdf"
    y_pred_7m, perm_imp_dfs_sort_7m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)
    y_pred_7m.to_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_7m.csv', index=False)
    perm_imp_dfs_sort_7m.to_csv(path+'/data/ChanpionData/addinfo/importance_LOO_RF_7m.csv')
    y_pred_7m = pd.read_csv(path+'/data/ChanpionData/addinfo/y_pred_LOO_RF_7m.csv')
    plotDataFrame_tree_7m = predPT_tsDF(y=y,y_pred=y_pred_7m, read_prefix="treeDistribution_patient_No",write_fileName="ChanpionData/addinfo/plotDataFrame_tree_7m.csv")
    allPat_7m = get_allPat(y,y_pred_7m,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_7m,colorNumdata=Life,allPat=allPat_7m,filename="ChanpionData/addinfo/LOOCV_gd_PredEstData_plot_tree_7m.pdf")
    area_7m = cul_area(y[['g_mean','d_mean','P0_mean']],y_pred_7m[['g_mean','d_mean','P0_mean']])

    Areadata = pd.DataFrame({
        "d0":area_0,
        "d0m":area_0m,
        "d1":area_1,
        "d1m":area_1m,
        "d2":area_2,
        "d2m":area_2m,
        "d3":area_3,
        "d3m":area_3m,
        "d7":area_7,
        "d7m":area_7m
    })
    Areadata.to_csv(path+"/data/ChanpionData/addinfo/area_eachData.csv", index=False)

    print("rf_seed",rf_seed,", n_tree",n_tree)
    print("___ finish! ___")