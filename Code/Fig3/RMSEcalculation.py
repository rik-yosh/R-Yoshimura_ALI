print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Fig3.gdP0prediction import get_allPat


if os.path.exists('./R-Yoshimura_ALI'):
    os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()

plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"

# PTの実際のデータ作成
def get_X_PTdat_L(X):
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]
    X_PTdat['id_again'] = range(1,len(X_PTdat)+1)
    X_PTdat_L = X_PTdat.melt(id_vars='id_again',var_name='dayName',value_name='PT_percentage')
    X_PTdat_L['Date'] = X_PTdat_L['dayName'].str.split('_', expand=True).iloc[:,2]
    return X_PTdat_L


def RMSEchange(num,y,X_PTdat_L,clustData):
    DF_RMSE = pd.DataFrame(columns=["id_again","d_num","RMSE_TE","RMSE_TP","RMSE_EP"],index=[])
    for i, num_x in enumerate(num):
        # get predicted Parameters
        y_pred = pd.read_csv(path+'/Data/Fig3/predicted/y_pred_LOO_RF_'+num_x+'.csv')
        # get True, Est and Pred PT values at each time point
        print("DFlength",len(y_pred),len(y),X_PTdat_L)
        allPat = get_allPat(y=y,y_pred=y_pred,X_PTdat_L=X_PTdat_L)
        
        # Calculate RMSE per each Dataset(d0, d0m, d1, etc.)
        x_dnum_RMSE = pd.DataFrame(columns=["id_again","d_num","RMSE_TE","RMSE_TP","RMSE_EP"],index=[])
        for j,idx in enumerate(set(allPat["id_again"])):
            
            # Extract each Patient
            x_Pat = allPat[allPat["id_again"]==idx]
            x_clust = clustData[j]

            # Calculate RMSE per each patient
            RMSE_TE = (sum((x_Pat['PT_true'] - x_Pat['PT_est'])**2)/len(x_Pat.index))**(1/2)
            RMSE_TP = (sum((x_Pat['PT_true'] - x_Pat['PT_pred'])**2)/len(x_Pat.index))**(1/2)
            RMSE_EP = (sum((x_Pat['PT_est'] - x_Pat['PT_pred'])**2)/len(x_Pat.index))**(1/2)
            
            # Concatenation all Patient
            x_pat_RMSE = pd.DataFrame({"id_again":idx,"d_num":num_x,"RMSE_TE":RMSE_TE,"RMSE_TP":RMSE_TP,"RMSE_EP":RMSE_EP,"Clustering":x_clust},index=[j])
            x_dnum_RMSE = pd.concat([x_dnum_RMSE,x_pat_RMSE],axis=0)
        
        # Concatenation all dataset
        DF_RMSE = pd.concat([DF_RMSE,x_dnum_RMSE],axis=0)
    
    return DF_RMSE.reset_index(drop=True)


if __name__ == "__main__":
    ClustData = pd.read_csv(path+"/Data/DTWclustLabel.csv")
    y = pd.read_csv(path+'/monolix/PT_NLMEMfit/IndividualParameters/estimatedIndividualParameters_new.txt',sep=",").loc[:,['g_mean','d_mean','P0_mean']]

    X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")

    num = ["0","0m","1","1m","2","2m","3","3m","7","7m"]
    X_PTdat_L = get_X_PTdat_L(X)
    
    DF_RMSE = RMSEchange(num,y,X_PTdat_L,ClustData.iloc[:,0])
    DF_RMSE.to_csv(path+"/Data/Fig3/RMSEchange.csv",index=False)