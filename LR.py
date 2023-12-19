import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import csv


#####Data Extraction#####
xdata = []
ydata = []
with open('LinearRegressionData.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    # Skip the first row (header)
    next(reader)
    for row in reader:
        xdata.append(float(row[0]))
        ydata.append(float(row[1]))
        # print(row[0])
        # print(row[1])
xdata = np.array(xdata, dtype=np.float32)
ydata = np.array(ydata, dtype=np.float32)


#####Data plotting#####
# plt.plot(xdata, ydata, 'ro')#red line plot
# plt.xlabel('Hours of Studying')
# plt.ylabel('Scores according to hours of studying')
# plt.title('Relationship between hours corresponding to scores')
# plt.show()


#######DEFINING MODEL###########
#Defining Hyper parameter 
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.01
#LR model -> y = wx + b format
model = nn.Linear(input_size, output_size)
#Loss and optimizer
criterion = nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(), lr=learning_rate)
# x_data와 y_data의 형태(차원)를 출력
print(xdata.shape, ydata.shape)  
# 만약 x_data와 y_data가 1차원이라면, 차원을 확장하여 2차원으로 만듦
#대부분의 파이토치 모델은 입력 데이터의 형태가 (배치 크기, 데이터의 차원)이 되길 요구해서
if len(xdata.shape) == 1 and len(ydata.shape) == 1:
    x_data = np.expand_dims(xdata, axis=-1)  # x_data의 마지막 축에 차원을 추가
    y_data = np.expand_dims(ydata, axis=-1)  # y_data의 마지막 축에 차원을 추가
print(x_data.shape, y_data.shape)  # 차원이 변경된 후의 x_data와 y_data의 형태를 출력

#######Training Model#######
for epoch in range(num_epochs):
    #Convert numpy arrays to torch tensors
    input = torch.from_numpy(x_data)
    realTarget = torch.from_numpy(y_data)
    #predict output with linear model
    predictionOutput = model(input)
    loss = criterion(predictionOutput, realTarget)#예측값, 실제값
    # 그래디언트 계산 및 파라미터 업데이트
    optimizer.zero_grad()  # 이전 그래디언트를 0으로 설정
    loss.backward()  # 손실에 대해 역전파를 수행하여 그래디언트를 계산
    optimizer.step()  # 계산된 그래디언트를 사용하여 모델의 가중치를 업데이트

    # 일정 에폭마다 손실을 출력
    if (epoch+1) % 10 == 0:  # 매 10번째 에폭마다
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  # 에폭 번호와 손실을 출력



# 그래프 출력
        #model이 학습된 상태 위로부터
predicted = model(torch.from_numpy(x_data)).detach().numpy()  # 모델을 사용하여 예측값을 계산하고, PyTorch 텐서에서 numpy 배열로 변환
plt.plot(x_data, y_data, 'ro', label='Original data')  # 원본 데이터를 빨간색 점으로 플롯
plt.plot(x_data, predicted, label='Fitted Line')  # 모델에 의해 예측된 값을 선으로 플롯
plt.legend()  # 범례를 플롯에 추가
plt.show()  # 플롯을 보여줌
