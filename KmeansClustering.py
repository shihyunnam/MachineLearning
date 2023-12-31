import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
#판을 생성 (그림)
# f1 = plt.figure(1)
#축을 추가
#ax1 = f1.add_subplot(231)
#plotting
#ax1.plot(X,Y, "b")
# plt.show()
plt.close("all")

dfLoad = pd.read_csv("kmeans.csv", sep="\s+")
# print(dfLoad.shape)
samples = np.array(dfLoad)
x = samples[:, 0]
y = samples[:, 1]
#curr plot
N = len(x)
numCluster = 2 #cluster 개수
f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(x, y, "b.")


#전체평균, 표준편차를 이용해서 initialize latent variable 
#INITIALIZE Z
[mx, sx] = [np.mean(x), np.std(x)]
[my, sy] = [np.mean(y), np.std(y)]
z0 = np.array([mx + sx, my + sy]).reshape(1,2)
z1 = np.array([mx - sx, my - sy]).reshape(1,2)
Z = np.r_[z0, z1]
ax1.plot(Z[:,0], Z[:,1], "r*", markersize="20")
# plt.show()

#mapping nearest Z for individual data
k = np.zeros(N)
numOfUpdates = 0
while(True):
    numOfUpdates += 1
    kOld = np.copy(k)
    for i in np.arange(N):
        z0D = np.linalg.norm(samples[i, :] - Z[0, :])
        z1D = np.linalg.norm(samples[i, :] - Z[1, :])
        k[i] = z0D > z1D#더 작은걸로 매핑
    if(np.alltrue(kOld == k)):
        break
    dfCluster = pd.DataFrame(np.c_[x,y,k])
    dfCluster.columns = ["X","Y","K"]
    dfGroup = dfCluster.groupby("K")
    # for (cluster, dataInCluster) in dfGroup:
    #     print(cluster)
    #     print(dataInCluster)
    for cluster in range(numCluster):
        Z[cluster,:] = dfGroup.mean().iloc[cluster] 

f2 = plt.figure(2)
ax2 = f2.add_subplot(111)
for (cluster, dataInCluster) in dfGroup:
    ax2.plot(dataInCluster.X, dataInCluster.Y,".", label = cluster)

# ax1.plot(x, y, "b.")     
ax2.plot(Z[:,0], Z[:,1], "r*", markersize="20")
ax2.legend()
print("num of updates is ", numOfUpdates)
plt.show()

