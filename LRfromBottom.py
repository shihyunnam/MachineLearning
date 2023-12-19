import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data loading and preparation
dfLoad = pd.read_csv("https://raw.githubusercontent.com/hanwoolJeong/lectureUniv/main/testData_LinearRegression.txt", sep='\t')
xxraw = dfLoad['xx']
yyraw = dfLoad['yy']
yyrawNP = np.array(yyraw)

# Scatter plot of the raw data
plt.plot(xxraw, yyraw, 'r.')

# Normal Equation to find the best fit line
NData = len(xxraw)
X = np.c_[np.ones([NData , 1]), xxraw]
wOLS = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(yyrawNP.reshape(NData, 1))
print("wols is ", wOLS)
# Prediction for the regression line
xPredict = np.linspace(0, 2, num=101)
xPredictPadding = np.c_[np.ones([101,1]), xPredict]
yPredict = wOLS.T.dot(xPredictPadding.T)

# Plot the regression line
plt.plot(xPredict, yPredict.reshape(-1), "b.-", label='OLS Prediction')  # Ensure yPredict is a flat array
# # Display the combined plot
# plt.show()




#USING Gradient Descent Method
learning_rate = 0.1
NumIteration = 20
WGD = np.zeros((2,1))  # Initial parameters set to 0
for iter in np.arange(NumIteration):
    gradient = -(2/NData) * X.T.dot(yyrawNP.reshape(NData, 1) - X.dot(WGD))
    WGD = WGD - learning_rate * gradient
    print(iter)
print(WGD)
print("WGD is ", WGD)
yNewPredict = WGD.T.dot(xPredictPadding.T)
plt.plot(xPredict, yNewPredict.reshape(-1), "g.-", label='Gradient Descent Prediction')  # using W val from Gradient Descent method
plt.legend()
plt.show()
# #RSS minimized equation using normal equation
# #Finding the optimized w hat value (Wols)= (X^T*X)*X^T*Y 