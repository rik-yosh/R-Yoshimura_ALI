print("___ Import Module ___")
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from onAdmission_d7_Blood import DrawROC

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

path = os.getcwd()

print("___ data loading ___")

data = pd.read_csv(path+"/data/ChanpionData/miceData/240214_df_afterMice_1.csv")

# # be same label number
# data_1 = data[data["InternalMedicineSurvival"]==1].reset_index(drop=True)
# print(data_1.shape)
# data_0 = data[data["InternalMedicineSurvival"]==0].reset_index(drop=True)
# data_0 = data_0.iloc[random.sample(range(len(data_0.index)),len(data_1.index)),:]
# data = pd.concat([data_0,data_1],axis=0)

# shaffule
def get_lr_y(coef,intercept,x):
    y = 1/(1+math.exp(-(coef*x+intercept)))
    return y
def get_lr_x(coef,intercept,y):
    x = (math.log(y/(1-y))-intercept)/coef
    return x



X = data[["PT_percentage_d7"]]
LifeData = pd.read_csv(path+"/data/ChanpionData/dataloading/240214_df_d0all.csv")
y = LifeData["InternalMedicineSurvival"].reset_index(drop=True)

rf_seed =111
np.random.seed(rf_seed)
logiR = LogisticRegression(random_state=0#,penalty="elasticnet",solver="saga",l1_ratio=0.8
                           )
DrawROC(X,y=y,model=logiR,n_cv=4,fileName="/results/ChanpionData/logistic_d7_ROC.pdf")

logiR.fit(X,y)
print("w: ",logiR.coef_,"b:",logiR.intercept_)


with open(path+'/data/ChanpionData/logiR.pickle', mode='wb') as f:
    pickle.dump(logiR,f,protocol=2)


w = logiR.coef_[0][0]
b = logiR.intercept_
y_test = get_lr_y(w,b,80)
print(y_test)
x_test = get_lr_x(w,b,y_test)
print(x_test)


PT_thresh = get_lr_x(w,b,0.5)
print(PT_thresh)

f = open(path+'/data/ChanpionData/logiR_thresh.txt', 'wb')
pickle.dump(PT_thresh, f)

pt_x = np.arange(0., 150., 0.1)
LD_y = [get_lr_y(w,b,x) for x in pt_x]

D_index = y[y==1].index
L_index = y[y==0].index

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(X[y==0], y[y==0], color=[cm.Pastel2(0) for i in range(len(y[y==0]))],label="Severe Patients")
ax.scatter(X[y==1], y[y==1], color=[cm.Pastel2(1) for i in range(len(y[y==1]))],label="Restored Patients")
ax.plot(pt_x, LD_y,c="black",lw=3)
thresh = get_lr_x(w,b,0.5)
ax.axvline(get_lr_x(w,b,0.5), linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (thresh))
# plt.axhline(0.5, linewidth = 1, ls = ":", color = "black")
ax.tick_params()
ax.set(
    # title="Logistic regression",
    xlabel="Prothrombin time activity percentage (%)",
    ylabel="Probability of requiring transplantation"
)
# plt.legend(fontsize=13)
plt.savefig(path+"/results/ChanpionData/logisticR_d7_fit.pdf",bbox_inches="tight")
plt.close()