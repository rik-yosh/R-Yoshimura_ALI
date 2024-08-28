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

def RF_shap(X,y,n_tree,rf_seed,figsize=(10.5,10.5),colorbar=True,shapfilename="sample.pdf"):
    loo = LeaveOneOut()
    rfr = RandomForestRegressor(n_estimators=n_tree, random_state=rf_seed)
    shapvalDF = shapBeeWarmplot(model=rfr,X=X,y=y,figsize=figsize,colorBar=colorbar,max_display=50,fileName=shapfilename)
    
    return shapvalDF

if __name__=="__main__":
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
    LifeData = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_d0all.csv")
    Life = LifeData["InternalMedicineSurvival"].reset_index(drop=True)

    # Clustering labelの追加
    ClustData = pd.read_csv(path+"/data/ChanpionData/DTWclustLabel.csv")["dtw_6"]

    y = pd.read_csv(path+'/data/ChanpionData/estimatedIndividualParameters.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]
    y["Clustering"] = ClustData
    X = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_1.csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={'TDratio_bil':'TDratio_d0'})
    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]

    print("___ Calculate each data ___")
    rf_seed = 111
    n_tree=500


    print("day0_data")
    x_data = Nd0_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,figsize=(5,10.5),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0_g.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['g_mean']],n_tree,rf_seed,colorbar=False,figsize=(4,10.5),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0_d.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['d_mean']],n_tree,rf_seed,colorbar=False,figsize=(4,10.5),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0_P0.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['P0_mean']],n_tree,rf_seed,colorbar=False,figsize=(4,10.5),shapfilename=SHAPname)

    print("day0_data + med")
    x_data = x_data+Nd0_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_0m.pdf"
    perm_imp_dfs_sort_0m = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data")
    x_data = x_data+Nd1_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_1.pdf"
    perm_imp_dfs_sort_1 = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med")
    x_data = x_data+Nd1_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_1m.pdf"
    perm_imp_dfs_sort_1m = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data")
    x_data = x_data+Nd2_data
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,figsize=(5,10.5),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2_g.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['g_mean']],n_tree,rf_seed,colorbar=False,figsize=(4.4,11.55),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2_d.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['d_mean']],n_tree,rf_seed,colorbar=False,figsize=(4.4,11.55),shapfilename=SHAPname)
    
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2_P0.pdf"
    perm_imp_dfs_sort_0 = RF_shap(X[x_data],y[['P0_mean']],n_tree,rf_seed,colorbar=False,figsize=(4.4,11.55),shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data + med")
    x_data = x_data+Nd2_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_2m.pdf"
    perm_imp_dfs_sort_2m = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data + med + day3_data")
    x_data = x_data+Nd3_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_3.pdf"
    perm_imp_dfs_sort_3 = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med")
    x_data = x_data+Nd3_med
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_3m.pdf"
    perm_imp_dfs_sort_3m = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data")
    x_data = x_data+Nd7_data
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_7.pdf"
    perm_imp_dfs_sort_7 = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data + med")
    x_data = x_data+Nd7_med+Ndlast
    SHAPname = "/results/ChanpionData/addinfo/SHAP_all_7m.pdf"
    perm_imp_dfs_sort_7m = RF_shap(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,shapfilename=SHAPname)