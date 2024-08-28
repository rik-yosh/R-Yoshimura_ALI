import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from cmath import nan


os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()

plt.rcParams["font.family"] = "sans-serif"
# Get raw dataset
df = pd.read_csv(path+"/Data/rawData.csv")

# Define acceptable NA rate in each data
threshRate = 0.8

# Get day0 data
df_d0all = df[df['BloodColectionDate']==0.0].reset_index(drop=True)
df_d0all = df_d0all.dropna(axis=1, how='all')

# Remove id data columns from day0 data
df_d0all = df_d0all.drop(["id"],axis=1)
df_d0all['id_new'] = range(1,len(df_d0all)+1)

# Replace Japanese word to English
df_d0all['Etiology'] =df_d0all['Etiology'].replace(dict(zip(['AIH acute on chronic','AIH acute on chronic ','AIH再燃','Acute AIH','acute AIH','Alcohol ','Alchol','HAV\u3000','HBV ','HBV Exacebation','HBV acute','HBV acute on chronic','HBV de novo','HBV exacerbation', 'de novo HBV', 'denovo HBV','HCV acute','Unknown','unknown\n',"Alcohol？Or DIC"],['AIH','AIH','AIH','AIH','AIH','Alcohol','Alcohol','HAV','HBV','HBV','HBV','HBV','HBV','HBV','HBV','HBV','HCV','unknown','unknown',"Alcohol"])))
df_d0all['Etiology'] = df_d0all['Etiology'].replace(dict(zip(['DILI、ミトタン','鉄中毒','Did 3回'], ['DILI', 'iron addiction', 'Did_3times'])))
df_d0all['InternalMedicineSurvival'] = df_d0all['InternalMedicineSurvival'].replace(dict(zip(['Survival(Internal)','Death(Internal)'], [int(0),int(1)])))
df_d0all['ANA_over40'] = df_d0all['ANA_over40'].replace({320:nan})
df_d0all["mAST"] = df_d0all["mAST"].replace({'1月8日':nan})
df_d0all["mAST"] = df_d0all["mAST"].astype(float)
# Replace the typo?
df_d0all= df_d0all.replace(dict(zip(['Not Done','Not　Done','DIｄ','Did 3回'], ['NotDone','NotDone', 'Did', 'Did_3times'])))
# df_d0all['FFPtotalquantity'] = df_d0all['FFPtotalquantity'].replace('初日8',8).astype(float)
df_d0all = df_d0all.replace('#NUM!',nan)
# Calculate Tbil/Dbil
df_d0all['Dbil'] = pd.to_numeric(df_d0all['Dbil'], errors='coerce')
df_d0all['TDratio_bil'] = df_d0all['Dbil']/df_d0all['Tbil']
# Calculate MELD score
meld_list = []
for i,(cre,tbil,inr) in enumerate(zip(df_d0all['Cre'],df_d0all['Tbil'],df_d0all['PTINR'])):
    if (cre>0)&(tbil>0)&(inr>0):
        meld_value = 9.57*math.log(cre) + 3.78*math.log(tbil) + 11.2*math.log(inr) + 6.43
    else:
        meld_value = nan
    meld_list.append(meld_value)
df_d0all['MELDscore'] = meld_list
# Drop too much missing data, etc...
df_d0all = df_d0all.dropna(thresh=len(df_d0all.index)*threshRate, axis=1)
df_d0all = df_d0all.drop(["SurvivalOrDeath","BloodColectionDate"],axis=1)
# Remove time series data...
df_d0all = df_d0all.drop(['Plt','FDP','DD','PT_percentage','Alb','BUN','Cre','Tbil','Dbil','AST','ALT','LDH','NH3','FFPprescribing_U','rhTM '],axis=1)
# Save
# df_d0all.to_csv(path+"/Data/df_d0all.csv",index=False)

# Sefparate data by data type
df_sup = df_d0all.select_dtypes(include='int64')
df_chalacter = df_d0all.select_dtypes(include='object')
df_numeric = df_d0all.select_dtypes(include='float64')


# Convert string data to dummy variables
df_dummy = pd.get_dummies(df_chalacter,drop_first=True)
# Concatnate each data
df_d0all_re = pd.concat([df_sup, df_numeric,df_dummy], axis=1)


# Get time series data
ts_columns = df[df['BloodColectionDate']!=0.0].dropna(axis=1, how='all').columns.to_list()
df_TS = df[ts_columns].dropna(axis=0, how='all').reset_index(drop=True)
# Label with id_new
BCdate = [0,1,2,3,7]
id_new_forTS = [i for i in range(1,len(df_d0all)+1) for _ in range(len(BCdate))]
df_TS['id_new'] = id_new_forTS
# Time series Data preparing
df_TS['Dbil'] = pd.to_numeric(df_TS['Dbil'], errors='coerce')
df_TS = df_TS.replace(dict(zip(['Not Done','Not　Done','DIｄ','Did 3回'], ['NotDone','NotDone', 'Did', 'Did_3times'])))

ts_columns = df_TS.columns
### Shift the date of treatment information
# The FFPprescribing and rhTM values are the treatment procedures performed on that day, 
# so they are shifted to a form that is similar to the test items of that day, which were processed the day before.
FFPlist = []
rhTMlist = []
for i in range(len(df_TS['id_new'])):
    if i ==0:
        FFPlist.append(0.0)
        rhTMlist.append('NotDone')
    elif np.isnan(df_TS['FFPprescribing_U'].iloc[i-1]):
        FFPlist.append(0.0)
        rhTMlist.append('NotDone')
    else:
        FFPlist.append(float(df_TS['FFPprescribing_U'].iloc[i-1]))
        rhTMlist.append(df_TS['rhTM '].iloc[i-1])

df_TS = df_TS.drop(['FFPprescribing_U', 'rhTM '],axis=1)
df_TS['Yesterday_FFPprescribing_U'] = FFPlist
df_TS['Yesterday_rhTM'] = rhTMlist

# Drop too much missing data
df_TS = df_TS.dropna(axis=1, thresh=len(df_TS)*threshRate)# 大体3割以上の欠損は削除

# Separate data by data type
df_chalacter = df_TS.select_dtypes(include='object').reset_index(drop=True)
df_numeric = df_TS.select_dtypes(include='float64').reset_index(drop=True)
df_int = df_TS.select_dtypes(include='int64').reset_index(drop=True)
# Change string data to dummy
df_dummy = pd.get_dummies(df_chalacter,drop_first=True)
df_TS = pd.concat([df_int,df_numeric,df_dummy], axis=1).reset_index(drop=True)

#
df_TS_Wide = pd.DataFrame({"id_new":range(1,320)})
for i in sorted(set(df_TS['BloodColectionDate'].astype(int))):
    dat = df_TS[df_TS['BloodColectionDate']==i].drop(['BloodColectionDate'],axis=1).reset_index(drop=True)
    # dat[dat['Tbil'].isna()]['Tbil'] = 0.0000000001
    dat.rename(columns={
        'Plt':'Plt_d'+str(i)+'',
        'FDP':'FDP_d'+str(i)+'',
        'DD':'DD_d'+str(i)+'', 
        'PT_percentage':'PT_percentage_d'+str(i)+'',
        'Alb':'Alb_d'+str(i)+'',
        'BUN':'BUN_d'+str(i)+'',
        'Cre':'Cre_d'+str(i)+'', 
        'Tbil':'Tbil_d'+str(i)+'', 
        'Dbil':'Dbil_d'+str(i)+'',
        'AST':'AST_d'+str(i)+'',
        'ALT':'ALT_d'+str(i)+'',
        'LDH':'LDH_d'+str(i)+'',
        'NH3':'NH3_d'+str(i)+'',
        'Yesterday_FFPprescribing_U':'Yesterday_FFPprescribing_U_d'+str(i)+'',
        'Yesterday_rhTM_NotDone':'Yesterday_rhTM_NotDone_d'+str(i)+''
        }, inplace=True)
    dat['TDratio_d'+str(i)+''] = dat['Dbil_d'+str(i)+'']/dat['Tbil_d'+str(i)+'']

    dat = dat.drop(["id_new"],axis=1)
    df_TS_Wide = pd.concat([df_TS_Wide,dat.drop(['Dbil_d'+str(i)+'','Tbil_d'+str(i)+''],axis=1)],axis=1).reset_index(drop=True)
    df_TS_Wide = df_TS_Wide.dropna(thresh=len(df_TS_Wide)*threshRate,axis=1)
 
df_forMice = pd.concat([df_d0all_re,df_TS_Wide.drop(['id_new'],axis=1)], axis=1)
df_forMice.to_csv(path+"/Data/df_forMice.csv",index=None)
