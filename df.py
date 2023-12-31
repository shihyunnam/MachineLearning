import pandas as pd
import numpy as np
#공백
dfLoad = pd.read_csv("dataframe.csv", sep="\s+")
x = np.array(dfLoad["X"])
y = np.array(dfLoad["Y"])
#row indexing iloc
# print(dfLoad.iloc[1])
# print(dfLoad.iloc[1]["X"])


#grouping by 
# dfLoad.groupby("")
N = len(x)
np.random.seed(3)
k = np.round(np.random.rand(N))

npCluster = np.c_[x,y,k]
# print(npCluster.shape)
dfCluster = pd.DataFrame(npCluster)
# print(dfCluster)
dfCluster.columns = ["X", "Y", "K"]
dfGroup = dfCluster.groupby("K")
for (cluster, dataInCluster) in dfGroup:
    print(cluster)
    print(dataInCluster)