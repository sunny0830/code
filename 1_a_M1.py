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
test_training_array = test_array[:,:876] # 1*876
test_testing_array = test_array[:,876:] # 1*220

##############################

# 把data拆成train和testing arrays
data_list = []
list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
        'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
for i in list:
    data_list.append(data[i].tolist())
data_array = np.array(data_list) # 17*1096
training_array = data_array[:, :876] # 17*876
testing_array = data_array[:,876:] # 17*220

x_training_array = np.ones(876)
x_testing_array = np.ones(220)
#垂直串接
training_array = np.vstack([x_training_array, training_array]) # 18*876
testing_array = np.vstack([x_testing_array, testing_array]) # 18*220


def linear_regression(x, y):
    # np.matmul 矩陣相乘
    # np.linalg.inv 反矩陣
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T,x)),x.T),y)
    return w

x = training_array.T # 876*18
y = test_training_array.T # 876*1
w = linear_regression(x, y) # 18*1
weight = w.T # 1*18
y_train_predict = np.dot(weight, training_array) # 1*876
# 計算MSE和RMSE
E_train = y_train_predict - test_training_array
E_train_2 = E_train **2
train_RMSE = ((1/(2*876))*E_train_2.sum())**0.5
print(train_RMSE)



##############################

# 對訓練數據集的比較
plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.title('Linear Regression,M=1(train)', fontsize=12)
plt.xlabel('The nth data', fontsize=10)
plt.ylabel('PM2.5', rotation = 90, fontsize=10)
y_train_predict = y_train_predict.flatten() # 摺疊成一維的array
test_training_array = test_training_array.flatten()
# np.linspace 在一定範圍內均勻撒點
plt.plot(np.linspace(1,876,876),y_train_predict, "c-", linewidth=1,label="training_predict value")
plt.plot(np.linspace(1,876,876), test_training_array, "m-", linewidth=1,label="target value")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()

##############################

y_test_predict = np.dot(weight, testing_array)
E_test = y_test_predict - test_testing_array
E_test_2 = E_test**2
test_RMSE = ((1/(2*220))*E_test_2.sum())**0.5
print(test_RMSE)

##############################

# 對驗證數據集的比較
plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.title('Linear Regression,M=1(test)', fontsize=12)
plt.xlabel('The nth data', fontsize=10)
plt.ylabel('PM2.5', rotation = 90, fontsize=10)
y_test_predict = y_test_predict.flatten()
test_testing_array = test_testing_array.flatten()
# np.linspace 在一定範圍內均勻撒點
plt.plot(np.linspace(1,220,220),y_test_predict, "y-", linewidth=1,label="testing_predict value")
plt.plot(np.linspace(1,220,220), test_testing_array, "b-", linewidth=1,label="target value")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()

##############################