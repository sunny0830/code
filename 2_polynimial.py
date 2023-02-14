import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
data = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_X.csv', encoding='big5')
test = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_T.csv', encoding='big5')

# 把test拆成train和testing arrays
test_list = []
test_list.append(test['PM2.5'].tolist())
test_array = np.array(test_list)
test_training_array = test_array[:,:766] 
test_testing_array = test_array[:,766:]

##############################

# 把data拆成train和testing arrays
data_list = []
list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
        'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
for i in list:
    data_list.append(data[i].tolist())
data_array = np.array(data_list)
training_array = data_array[:, :766]
testing_array = data_array[:,766:]

##############################

M2_training_array = []
for i in range(766):
    training_array_i = training_array[:,i]
    for j in range(17):
        training_j = training_array_i[j]
        for k in range(17):
            if k >= j:
                training_k = training_array_i[k]
                training_data = training_j * training_k
                M2_training_array.append(training_data)
M2_training_array = np.reshape(M2_training_array,(766, 153))
M2_training_array = M2_training_array.T 


##############################

M2_testing_array = []
for i in range(330):
    testing_array_i = testing_array[:,i]
    for j in range(17):
        testing_j = testing_array_i[j]
        for k in range(17):
            if k >= j:
                testing_k = testing_array_i[k]
                testing_data = testing_j * testing_k
                M2_testing_array.append(testing_data)
M2_testing_array = np.reshape(M2_testing_array,(330, 153))
M2_testing_array = M2_testing_array.T

##############################

x_training_array = np.ones(766)
x_testing_array = np.ones(330)

#垂直串接
training_array = np.vstack([x_training_array, training_array]) 
training_array = np.vstack([training_array, M2_training_array])
testing_array = np.vstack([x_testing_array, testing_array])
testing_array = np.vstack([testing_array, M2_testing_array])

def linear_regression(x, y):
    # np.matmul 矩陣相乘
    # np.linalg.pinv 無反矩陣會找近似值
    w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T,x)),x.T),y)
    return w

x = training_array.T
y = test_training_array.T
w = linear_regression(x, y)
weight = w.T
y_train_predict = np.dot(weight, training_array)
# 計算MSE和RMSE
E_train = y_train_predict - test_training_array
E_train_2 = E_train **2
train_RMSE = ((1/(2*766))*E_train_2.sum())**0.5
print(train_RMSE)

##############################

# 對訓練數據集的比較
plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.title('Linear Regression,M=2(train)', fontsize=12)
plt.xlabel('The nth data', fontsize=10)
plt.ylabel('PM2.5', rotation = 90, fontsize=10)
y_train_predict = y_train_predict.flatten() # 摺疊成一維的array
test_training_array = test_training_array.flatten()
# np.linspace 在一定範圍內均勻撒點
plt.plot(np.linspace(1,766,766),y_train_predict, "c-", linewidth=1,label="training_predict value")
plt.plot(np.linspace(1,766,766), test_training_array, "m-", linewidth=1,label="target value")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()

##############################

y_test_predict = np.dot(weight, testing_array)
E_test = y_test_predict - test_testing_array
E_test_2 = E_test**2
test_RMSE = ((1/(2*330))*E_test_2.sum())**0.5
print(test_RMSE)

##############################

# 對驗證數據集的比較
plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.title('Linear Regression,M=2(test)', fontsize=12)
plt.xlabel('The nth data', fontsize=10)
plt.ylabel('PM2.5', rotation = 90, fontsize=10)
y_test_predict = y_test_predict.flatten()
test_testing_array = test_testing_array.flatten()
# np.linspace 在一定範圍內均勻撒點
plt.plot(np.linspace(1,330,330),y_test_predict, "y-", linewidth=1,label="testing_predict value")
plt.plot(np.linspace(1,330,330), test_testing_array, "b-", linewidth=1,label="target value")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()

##############################