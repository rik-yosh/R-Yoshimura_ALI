print("___ Import Module ___")
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.chdir('./R-Yoshimura_ALI')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 15

path = os.getcwd()

# shaffule
def get_lr_y(coef,intercept,x):
    y = 1/(1+math.exp(-(coef*x+intercept)))
    return y
def get_lr_x(coef,intercept,y):
    x = (math.log(y/(1-y))-intercept)/coef
    return x

if __name__ == "__main__":
    print("___ data loading ___")

    data = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")
    X = data[["PT_percentage_d7"]]
    y = data["InternalMedicineSurvival"]

    seed =  1234
    np.random.seed(seed)
    logiR = LogisticRegression(random_state=seed)
    # DrawROC(X,y=y,model=logiR,n_cv=4,fileName="/Output/Fig1D_roc.pdf")

    logiR.fit(X,y)
    print("w: ",logiR.coef_,"b:",logiR.intercept_)


    with open(path+'/Data/logiR.pickle', mode='wb') as f:
        pickle.dump(logiR,f,protocol=2)


    w = logiR.coef_[0][0]
    b = logiR.intercept_

    PT_thresh = get_lr_x(w,b,0.5)
    print(PT_thresh)

    f = open(path+'/Data/logiR_thresh.txt', 'wb')
    pickle.dump(PT_thresh, f)

    pt_x = np.arange(0., 150., 0.1)
    LD_y = [get_lr_y(w,b,x) for x in pt_x]

    D_index = y[y==1].index
    L_index = y[y==0].index

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(X[y==0], y[y==0], color=[cm.Pastel2(0) for i in range(len(y[y==0]))],label="Severe Patients")
    ax.scatter(X[y==1], y[y==1], color=[cm.Pastel2(1) for i in range(len(y[y==1]))],label="Restored Patients")
    ax.plot(pt_x, LD_y,c="black",lw=3)
    ax.axvline(PT_thresh, linewidth = 4,ls = ":", color = cm.Accent(5),label="PT = %0.2f" % (PT_thresh))
    # plt.axhline(0.5, linewidth = 1, ls = ":", color = "black")
    ax.tick_params()
    ax.set(
        # title="Logistic regression",
        xlabel="Prothrombin time activity percentage (%)",
        ylabel="Probability of requiring transplantation"
    )
    # plt.legend(fontsize=13)
    plt.savefig(path+"/Output/Fig1/Fig1D.png",bbox_inches="tight")
    plt.close()