import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
data = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_X.csv', encoding='big5')
test = pd.read_csv('/Users/sunny/Documents/linear regression/dataset_T.csv', encoding='big5')
original_RMSE =[]
d5_RMSE = []
d12_RMSE =[]
num = [1, 2, 3, 4, 5, 6, 7, 8, 9] # x座標

test_list = []
test_list.append(test['PM2.5'].tolist())
test_array = np.array(test_list)

##############################

data_array = []
list = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10',
        'RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']
for i in list:
    data_array.append(data[i].to_numpy())
data_array = np.reshape(data_array, (17, 1096))
data_array_d5 = np.vstack([data_array[2:3,:], data_array[8:11,:],data_array[13:14,:]])
data_array_d12 = np.vstack([data_array[1:4,:],data_array[7:12,:],data_array[13:,:]])

def linear_regression(x, y):
        # np.matmul 矩陣相乘
        # np.linalg.pinv 無反矩陣會找近似值
        w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T,x)),x.T),y)
        return w

##############################
# N-fold : 9
num_test_samples = 1096//9
# original
for i in range(9):
    testing_array = data_array[:,i*num_test_samples:(i+1)*num_test_samples] # 17*121
    test_testing_array = test_array[:,i*num_test_samples:(i+1)*num_test_samples]
    training_array = np.hstack([data_array[:,:i*num_test_samples],data_array[:,(i+1)*num_test_samples:]]) # 17*975
    test_training_array = np.hstack([test_array[:,:i*num_test_samples],test_array[:,(i+1)*num_test_samples:]])

    ##############################
    data_x = np.zeros(17)
    sb = np.zeros(17)
    for i in range(17):
        x_data_array = training_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/975 # 平均值
        sb[i] = (x2_data_array.sum()/975 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(17,1)
    sb = sb.reshape(17,1)

    ##############################

    x_train_sigmoidal = np.zeros((17, 975))
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(17):
            training_j = (training_array_i[j] - x_mean[j])/sb[j]
            x_train_sigmoidal[j,i] = 1/(1+np.exp(-(training_j)))

    ##############################

    data_x = np.zeros(17)
    sb = np.zeros(17)
    for i in range(17):
        x_data_array = testing_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/121 # 平均值
        sb[i] = (x2_data_array.sum()/121 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(17,1)
    sb = sb.reshape(17,1)

    ##############################
    x_test_sigmoidal = np.zeros((17, 121))
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(17):
            testing_j = (testing_array_i[j] - x_mean[j])/sb[j]
            x_test_sigmoidal[j,i] = 1/(1+np.exp(-(testing_j)))
    
    ##############################
    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)
    #垂直串接
    x_train_sigmoidal = np.vstack([x_training_array, x_train_sigmoidal])
    x_test_sigmoidal = np.vstack([x_testing_array, x_test_sigmoidal]) 

    x = x_train_sigmoidal.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    x_test_predict = np.dot(weight, x_test_sigmoidal)
    # 計算MSE和RMSE
    E_test = x_test_predict - test_testing_array
    E_test_2 = E_test**2
    test_RMSE = ((1/(2*121))*E_test_2.sum())**0.5
    original_RMSE.append(test_RMSE)
# D=5
for i in range(9):
    testing_array = data_array_d5[:,i*num_test_samples:(i+1)*num_test_samples]
    test_testing_array = test_array[:,i*num_test_samples:(i+1)*num_test_samples]
    training_array = np.hstack([data_array_d5[:,:i*num_test_samples],data_array_d5[:,(i+1)*num_test_samples:]])
    test_training_array = np.hstack([test_array[:,:i*num_test_samples],test_array[:,(i+1)*num_test_samples:]])

    ##############################
    data_x = np.zeros(5)
    sb = np.zeros(5)
    for i in range(5):
        x_data_array = training_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/975 # 平均值
        sb[i] = (x2_data_array.sum()/975 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(5,1)
    sb = sb.reshape(5,1)

    ##############################

    x_train_sigmoidal = np.zeros((5, 975))
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(5):
            training_j = (training_array_i[j] - x_mean[j])/sb[j]
            x_train_sigmoidal[j,i] = 1/(1+np.exp(-(training_j)))

    ##############################
    data_x = np.zeros(5)
    sb = np.zeros(5)
    for i in range(5):
        x_data_array = testing_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/121 # 平均值
        sb[i] = (x2_data_array.sum()/121 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(5,1)
    sb = sb.reshape(5,1)

    ##############################
    x_test_sigmoidal = np.zeros((5, 121))
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(5):
            testing_j = (testing_array_i[j] - x_mean[j])/sb[j]
            x_test_sigmoidal[j,i] = 1/(1+np.exp(-(testing_j)))
    
    ##############################
    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)
    #垂直串接
    x_train_sigmoidal = np.vstack([x_training_array, x_train_sigmoidal])
    x_test_sigmoidal = np.vstack([x_testing_array, x_test_sigmoidal]) 

    x = x_train_sigmoidal.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    x_test_predict = np.dot(weight, x_test_sigmoidal)
    # 計算MSE和RMSE
    E_test = x_test_predict - test_testing_array
    E_test_2 = E_test**2
    test_RMSE = ((1/(2*121))*E_test_2.sum())**0.5
    d5_RMSE.append(test_RMSE)
# D=12
for i in range(9):
    testing_array = data_array_d12[:,i*num_test_samples:(i+1)*num_test_samples]
    test_testing_array = test_array[:,i*num_test_samples:(i+1)*num_test_samples]
    training_array = np.hstack([data_array_d12[:,:i*num_test_samples],data_array_d12[:,(i+1)*num_test_samples:]])
    test_training_array = np.hstack([test_array[:,:i*num_test_samples],test_array[:,(i+1)*num_test_samples:]])

    ##############################
    data_x = np.zeros(12)
    sb = np.zeros(12)
    for i in range(12):
        x_data_array = training_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/975 # 平均值
        sb[i] = (x2_data_array.sum()/975 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(12,1)
    sb = sb.reshape(12,1)

    ##############################

    x_train_sigmoidal = np.zeros((12, 975))
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(12):
            training_j = (training_array_i[j] - x_mean[j])/sb[j]
            x_train_sigmoidal[j,i] = 1/(1+np.exp(-(training_j)))

    ##############################

    data_x = np.zeros(12)
    sb = np.zeros(12)
    for i in range(12):
        x_data_array = testing_array[i,:]
        x2_data_array = x_data_array**2
        data_x[i] = x_data_array.sum()/121 # 平均值
        sb[i] = (x2_data_array.sum()/121 - (data_x[i])**2)**0.5
    x_mean = data_x.reshape(12,1)
    sb = sb.reshape(12,1)

    ##############################
    x_test_sigmoidal = np.zeros((12, 121))
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(12):
            testing_j = (testing_array_i[j] - x_mean[j])/sb[j]
            x_test_sigmoidal[j,i] = 1/(1+np.exp(-(testing_j)))
    
    ##############################
    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)
    #垂直串接
    x_train_sigmoidal = np.vstack([x_training_array, x_train_sigmoidal])
    x_test_sigmoidal = np.vstack([x_testing_array, x_test_sigmoidal]) 

    x = x_train_sigmoidal.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    x_test_predict = np.dot(weight, x_test_sigmoidal)
    # 計算MSE和RMSE
    E_test = x_test_predict - test_testing_array
    E_test_2 = E_test**2
    test_RMSE = ((1/(2*121))*E_test_2.sum())**0.5
    d12_RMSE.append(test_RMSE)


plt.rcParams["figure.figsize"] = (18,8) # 顯示圖像的最大範圍
plt.title('Linear Regression,M=2(test)', fontsize=12)
plt.xlabel('The nth fold', fontsize=10)
plt.ylabel('RMSE', rotation = 90, fontsize=10)
plt.plot(num, original_RMSE, "y-", linewidth=1,label="Original Model")
plt.plot(num, d5_RMSE, "b-", linewidth=1,label="Model. D=5")
plt.plot(num, d12_RMSE, "g-", linewidth=1,label="Model. D=12")
plt.legend(loc = "best", fontsize=10) # 加圖例
plt.show()