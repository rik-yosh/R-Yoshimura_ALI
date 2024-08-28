print("___ Import Module ___")
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Fig1.Fig1AB import shapBeeWarmplot

import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
import math
from sklearn.metrics import r2_score


if os.path.exists('./R-Yoshimura_ALI'):
    os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()

plt.rcParams["font.family"] = "sans-serif"

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


def ErrList_PTcalculator(g,d,P0,percentile,DistrData, timeList):
    
    PT_randCurve_list = pd.DataFrame()
    errDown_list = []
    errUp_list = []
    
    for i in range(len(DistrData)):
        g_rand = DistrData.iloc[i,0]
        d_rand = DistrData.iloc[i,1]
        P0_rand = DistrData.iloc[i,2]
        PT_randCurve = pd.DataFrame({'S'+str(i)+'':PTcalculator(g_rand, d_rand, P0_rand, timeList)})
        PT_randCurve_list = pd.concat([PT_randCurve_list, PT_randCurve],axis=1)
    
    # たくさん描画したPTの中から95%区間のデータ
    for i in range(len(PT_randCurve_list.index)):
        err_down= np.percentile(PT_randCurve_list.iloc[i,:], (100 - percentile) / 2. )
        err_up = np.percentile(PT_randCurve_list.iloc[i,:], 100 - (100 - percentile) / 2.)
        errDown_list.append(err_down)
        errUp_list.append(err_up)
    return errDown_list,errUp_list

def predPT_tsDF(y=None,y_pred=None,devi_num=100, int_percentile = 95, read_folderName=None, write_fileName="/plotDataFrame_loo.csv"):
    T_plot = np.linspace(0,7,devi_num)
    plotDataFrame = pd.DataFrame(index=[],columns=['id_again','time','PT_est','PT_pred','Clustering'])

    for i in range(0,len(y)):
        print("No.", i)
        idnew = np.repeat(i+1,len(T_plot))
        estList = PTcalculator(y['g_mean'][i],y['d_mean'][i],y['P0_mean'][i],T_plot)
        predList = PTcalculator(y_pred['g_mean'][i],y_pred['d_mean'][i],y_pred['P0_mean'][i],T_plot)

        every_paraDist_tree = pd.read_csv(read_folderName+"/RF_treeDistribution_"+str(i)+".csv")

        downList, upList = ErrList_PTcalculator(
            g=y_pred['g_mean'][i],
            d=y_pred['d_mean'][i],
            P0=y_pred['P0_mean'][i],
            percentile = int_percentile,
            DistrData =every_paraDist_tree,
            timeList=T_plot
        )

        timepoint = pd.DataFrame({'id_again':idnew,'time':T_plot,'PT_est':estList,'PT_pred':predList, 'err_down':downList,'err_up':upList})
        timepoint['Clustering'] = y['Clustering'][i]
        plotDataFrame = pd.concat([plotDataFrame,timepoint],axis=0)
    plotDataFrame.to_csv(write_fileName, index=False)
    return plotDataFrame

# Function to get each tree prediction values
# https://blog.datadive.net/prediction-intervals-for-random-forests/
def pred_dist(model, X):
    X = X.reset_index(drop=True)
    x_reshape = np.array(list(X.iloc[0,:])).reshape(-1,1).T# predictにデータフレームを入れるために、np.arrayにしてreshapeして転置する必要がある

    every_paraDist = []# 各決定木が予測したp,d,P0のarrayを出力（これを経験分布として用いる）    
    for pred in model.estimators_:# 各決定木毎に予測値を出力
        every_paraDist.append(pred.predict(x_reshape)[0])

    return every_paraDist

def LOO_RF(X,y,n_tree,rf_seed,write_folderName="/default",shapfilename="sample.pdf"):
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
        one_paraDist = pred_dist(rfr, X_test) # one_paraDist[0][0][0]（[0]:一人目の、[0]:一つ目のパラメータ（g_mean）の、[0]:一つ目の木の予測）
        DF_one_paraDist = pd.DataFrame(one_paraDist,columns=['g_mean','d_mean','P0_mean'])
        DF_one_paraDist.to_csv(write_folderName+"/RF_treeDistribution_"+str(test_index[0])+'.csv', index=False)
        print(test_index[0])
    
    shapvalDF = shapBeeWarmplot(model=rfr,X=X,y=y,figsize=(6,10.5),max_display=5,fileName=shapfilename)
    
    return y_pred, shapvalDF

def pred_plot(plotDataFrame=None, colorNumdata=None,filename=None,allPat=None, plot_real=True, plot_est=True, plot_pred=True, plot_interval=True,actcolor="gray"):

    fig, axes = plt.subplots(
        3,8,
        sharex=True,
        sharey=True,
        figsize=(15,6),
        dpi=200
    )
    # color
    plot_num = [plot_real, plot_est, plot_pred, plot_interval]
    
    good = [5,158,215,3,188,271,21,39,306,203,239,266]
    bad = [102,214,288,4,78,120,198,204,309,125,138,62]
    selected = sorted(good+bad)

    for a, id in enumerate(selected):
        dataline =  plotDataFrame[plotDataFrame['id_again'] == id]
        dataPT = allPat[allPat['id_again']== id]
        # 12で割った商と余りを出す
        row,col = a // 8, a % 8
        ax = axes[row,col]

        # sample毎に描画
        if plot_pred:
            ax.plot(dataline['time'],dataline['PT_pred'],
                color=cm.Pastel2(colorNumdata[id-1]),
                label="RF prediction",
                lw=3,
                zorder=1,
                alpha=1
            )
        if plot_interval:
            ax.fill_between(dataline['time'],dataline['err_down'],dataline['err_up'],
                color=cm.Pastel2(colorNumdata[id-1]),
                alpha=0.4,
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
        
        if a == 16:
            ax.legend(bbox_to_anchor=(-0.2,-.4), loc='upper left',ncol=len(plot_num), frameon=False, borderaxespad=1,fontsize=13)
        
        # 軸ラベル
        ax.set_xlim(-1,8)
        ax.set_ylim(0,150)
        ax.set_xticks([0,1,2,3,7])
        # x軸目盛りのサイズを統一
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        #ax.legend(fontsize=12)
    fig.supxlabel("Days post-admission",fontsize=14)
    fig.supylabel("Prothrombin time activity percentage (%)",fontsize=14,x=0.08)
    fig.savefig(filename, bbox_inches='tight',dpi=200)
    plt.close()
    return fig


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
    # Clustering labelの追加
    ClustData = pd.read_csv(path+"/Data/DTWclustLabel.csv")["dtw_6"]

    y = pd.read_csv(path+'/monolix/PT_NLMEMfit/IndividualParameters/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]
    X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv").drop(["Unnamed: 0","TDratio_d0"],axis=1).rename(columns={"TDratio_bil":"TDratio_d0"})
    Life = X["InternalMedicineSurvival"]
    y["Clustering"] = ClustData

    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]
    # X_PTdat_L.to_csv(path+"/monolix/timeSeriesPT.csv",index=False)


    print("___ Calculate each data ___")
    rf_seed = 111
    n_tree=500

    print("day0_data")
    x_data = Nd0_data
    SHAPname = "/Output/Fig3/shap/SHAP_all_0.pdf"
    y_pred, perm_imp_dfs_sort_0 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_0.csv', index=False)
    perm_imp_dfs_sort_0.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_0.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_0.csv')
    plotDataFrame_tree_0 = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_0")
    allPat_0 = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_0,colorNumdata=Life,allPat=allPat_0,filename="Output/Fig3/LOOCV_gd_PredEstData_plot_tree_0.pdf")

    print("day0_data + med")
    x_data = x_data+Nd0_med
    SHAPname = "/Output/Fig3/shap/SHAP_all_0m.pdf"
    y_pred, perm_imp_dfs_sort_0m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_0m.csv', index=False)
    perm_imp_dfs_sort_0m.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_0m.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_0m.csv')
    plotDataFrame_tree_0m = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_0m.csv")
    allPat_0m = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_0m,colorNumdata=Life,allPat=allPat_0m,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_0m.pdf")

    print("day0_data + med + day1_data")
    # x_data = x_data+Nd1_data
    x_data = Nd0_data+Nd0_med+Nd1_data
    SHAPname = "/Output/Fig3/shap/SHAP_all_1.pdf"
    y_pred, perm_imp_dfs_sort_1 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_1.csv', index=False)
    perm_imp_dfs_sort_1.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_1.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_1.csv')
    plotDataFrame_tree_1 = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_1.csv")
    allPat_1 = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_1,colorNumdata=Life,allPat=allPat_1,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_1.pdf")

    print("day0_data + med + day1_data + med")
    x_data = x_data+Nd1_med
    SHAPname = "/Output/Fig3/shap/SHAP_all_1m.pdf"
    y_pred, perm_imp_dfs_sort_1m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_1m.csv', index=False)
    perm_imp_dfs_sort_1m.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_1m.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_1m.csv')
    plotDataFrame_tree_1m = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_1m.csv")
    allPat_1m = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_1m,colorNumdata=Life,allPat=allPat_1m,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_1m.pdf")

    print("day0_data + med + day1_data + med ")
    x_data = x_data+Nd2_data
    SHAPname = "/Output/Fig3/shap/SHAP_all_2.pdf"
    y_pred, perm_imp_dfs_sort_2 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_2.csv', index=False)
    perm_imp_dfs_sort_2.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_2.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_2.csv')
    plotDataFrame_tree_2 = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_2.csv")
    allPat_2 = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_2,colorNumdata=Life,allPat=allPat_2,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_2.pdf")

    print("day0_data + med + day1_data + med + day2_data + med")
    x_data = x_data+Nd2_med
    SHAPname = "/Output/Fig3/shap/SHAP_all_2m.pdf"
    y_pred, perm_imp_dfs_sort_2m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_2m.csv', index=False)
    perm_imp_dfs_sort_2m.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_2m.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_2m.csv')
    plotDataFrame_tree_2m = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_2m.csv")
    allPat_2m = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_2m,colorNumdata=Life,allPat=allPat_2m,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_2m.pdf")

    print("day0_data + med + day1_data + med + day2_data + med + day3_data")
    x_data = x_data+Nd3_data
    SHAPname = "/Output/Fig3/shap/SHAP_all_3.pdf"
    y_pred, perm_imp_dfs_sort_3 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_3.csv', index=False)
    perm_imp_dfs_sort_3.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_3.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_3.csv')
    plotDataFrame_tree_3 = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_3.csv")
    allPat_3 = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_3,colorNumdata=Life,allPat=allPat_3,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_3.pdf")

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med")
    x_data = x_data+Nd3_med
    SHAPname = "/Output/Fig3/shap/SHAP_all_3m.pdf"
    y_pred, perm_imp_dfs_sort_3m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_3m.csv', index=False)
    perm_imp_dfs_sort_3m.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_3m.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_3m.csv')
    plotDataFrame_tree_3m = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_3m.csv")
    allPat_3m = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_3m,colorNumdata=Life,allPat=allPat_3m,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_3m.pdf")

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data")
    x_data = x_data+Nd7_data
    SHAPname = "/Output/Fig3/shap/SHAP_all_7.pdf"
    y_pred, perm_imp_dfs_sort_7 = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_7.csv', index=False)
    perm_imp_dfs_sort_7.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_7.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_7.csv')
    plotDataFrame_tree_7 = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_7.csv")
    allPat_7 = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_7,colorNumdata=Life,allPat=allPat_7,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_7.pdf")

    print("day0_data + med + day1_data + med + day2_data + med + day3_data + med + day7_data + med")
    x_data = x_data+Nd7_med+Ndlast
    SHAPname = "/Output/Fig3/shap/SHAP_all_7m.pdf"
    y_pred, perm_imp_dfs_sort_7m = LOO_RF(X[x_data],y[['g_mean','d_mean','P0_mean']],n_tree,rf_seed,write_folderName=path+"/Data/Fig3/predictedDistribution",shapfilename=SHAPname)
    y_pred.to_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_7m.csv', index=False)
    perm_imp_dfs_sort_7m.to_csv(path+'/Data/Fig3/shap/importance_LOO_RF_7m.csv')
    y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_7m.csv')
    plotDataFrame_tree_7m = predPT_tsDF(y=y,y_pred=y_pred, read_folderName=path+"/Data/Fig3/predictedDistribution", write_fileName=path+"/Data/Fig3/DF_forPlot/plotDataFrame_tree_7m.csv")
    allPat_7m = get_allPat(y,y_pred,X_PTdat_L)
    pred_plot(plotDataFrame=plotDataFrame_tree_7m,colorNumdata=Life,allPat=allPat_7m,filename="Output/Fig3/LOOCV/LOOCV_gd_PredEstData_plot_tree_7m.pdf")

    print("rf_seed",rf_seed,", n_tree",n_tree)
    print("___ finish! ___")