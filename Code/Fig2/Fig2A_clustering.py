print("___ Import Module ___")
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from tslearn.clustering import TimeSeriesKMeans

os.chdir('./R-Yoshimura_ALI')
path = os.getcwd()
plt.rcParams["font.family"] = "sans-serif"


print("___ data loading ___")

def elbowClustering(matrix=None,clustMetric="euclidean",maxClustNum = 30,seed=111,fileName="/results/SAMPLE.pdf"):
    distortions = []
    for i  in range(1,maxClustNum+1):                # 1~10クラスタまで一気に計算 
        print("clust:",i)
        km = TimeSeriesKMeans(n_clusters=i,
                    init='k-means++',     # k-means++法によりクラスタ中心を選択
                    n_init=10,
                    metric=clustMetric,
                    max_iter=100,
                    random_state=seed)
        km.fit(matrix)                         # クラスタリングの計算を実行
        number = str(i)
        distortions.append(km.inertia_)   # km.fitするとkm.inertia_が得られる

    plt.figure()
    plt.plot(range(1,maxClustNum+1),distortions,marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.savefig(path+fileName)
    plt.close()

if __name__ == "__main__":
    X = pd.read_csv(path+"/Data/miceData/df_afterMice_1.csv")

    # PTの実際のデータ作成
    X_PTdat = X.loc[:,['PT_percentage_d0','PT_percentage_d1','PT_percentage_d2','PT_percentage_d3','PT_percentage_d7']]

    maxClustNum = 30
    seed = 42
    clustMetric = "dtw"
    elbowClustering(matrix=X_PTdat,clustMetric=clustMetric,maxClustNum = maxClustNum,seed=seed,fileName="/Output/Fig1A_"+clustMetric+"_elbow.png")