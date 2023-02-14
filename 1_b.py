import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
data = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_X.csv', encoding='big5')
test = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_T.csv', encoding='big5')
without_RMSE = []
all_RMSE = []
a = 0

# 把test拆成train和testing arrays
test_list = []
test_list.append(test['PM2.5'].tolist())
test_array = np.array(test_list)
test_training_array = test_array[:,:876] # 1*876
test_testing_array = test_array[:,876:] # 1*220

for i in range(18):
    ##############################
    # 把data拆成train和testing arrays
    data_list = []
    list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
            'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
    a +=1
    if a < 18:
        list.pop(i)
    for i in list:
        data_list.append(data[i].tolist())
    data_array = np.array(data_list)
    training_array = data_array[:, :876]
    testing_array = data_array[:,876:]
    x_training_array = np.ones(876)
    x_testing_array = np.ones(220)
    #垂直串接
    training_array = np.vstack([x_training_array, training_array])
    testing_array = np.vstack([x_testing_array, testing_array])


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
    if a < 18:
        without_RMSE.append(train_RMSE)
    else:
        all = train_RMSE
for i in range(17):
    all_RMSE.append(all)

list_p = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
            'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']

##############################

# 對訓練數據集的比較
plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.rcParams["xtick.labelsize"] = 6
plt.title('Linear Regression,M=1', fontsize=12)
plt.xlabel('The feature', fontsize=10)
plt.ylabel('Training RMSE', rotation = 90, fontsize=10)
plt.plot(list_p, without_RMSE, "ob", color = 'green', linewidth=1,label="The RMSE without x feature")
plt.plot(all_RMSE, "ob", color = 'orange', linewidth=1,label="RMSE with all features")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()