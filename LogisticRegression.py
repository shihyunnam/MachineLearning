import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close("all")
def sigmoid(x):
    return 1.0 /(1 + np.exp(-x))
# xxTest = np.linspace(-10, 10, num=101)
# plt.plot(xxTest, sigmoid(xxTest), "k. ")
# plt.show()



dfload = pd.read_csv("Logistic.txt", "\s+")
xxRaw = np.array(dfload.values[:,0])#working hours
# print(xxRaw)
yyRaw = np.array(dfload.values[:,1])#pass or fail
plt.xlabel("working hours")
plt.ylabel("pass or fail")
plt.plot(xxRaw, yyRaw, "k. ")
# plt.show()

#gradient of NLL = X^T(mu - y)

#Declaring design matrix
N = len(xxRaw)
x_bias = np.c_[np.ones([N,1]), xxRaw].T
# print(x_bias.shape)
# print(x_bias)
Y = yyRaw.reshape(N , 1)#1개의 value를 가진 N개의 벡터들의 집합
X = x_bias.T
# print(X.shape)
# print(Y)

eta = 0.1
nIterations = 1000
WGD = np.zeros([2,1])
WGDbuffer = np.zeros([2,nIterations + 1])#어떻게 변화하는지 보려고 w0 와 w1의 값이
# print(WGDbuffer.shape)

for iteration in range(nIterations):
    mu = sigmoid(WGD.T.dot(x_bias)).T
    gradients = X.T.dot(mu - Y)
    WGD = WGD - eta*gradients
    WGDbuffer[0, iteration + 1] = WGD[0][0]  # 첫 번째 가중치 값
    WGDbuffer[1, iteration + 1] = WGD[1][0]  # 두 번째 가중치 값
    # WGDbuffer[:, iteration + 1] = [WGD[0][0], WGD[1][0]]
    # WGDbuffer[:,iteration + 1] = [WGD[0], WGD[1]]

xxTest = np.linspace(0,10,num=N).reshape(N,1)
xxTest_bias = np.c_[np.ones([N,1]), xxTest]
# print(xxTest_bias.shape)
print(xxTest_bias.shape)
plt.plot(xxTest, sigmoid(WGD.T.dot(xxTest_bias.T)).T, "r-.")
plt.show()
