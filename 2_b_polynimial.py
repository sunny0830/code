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
    M2_training_array = []
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(17):
            training_j = training_array_i[j]
            for k in range(17):
                if k >= j:
                    training_k = training_array_i[k]
                    training_data = training_j * training_k
                    M2_training_array.append(training_data)

    M2_training_array = np.reshape(M2_training_array,(975, 153))
    M2_training_array = M2_training_array.T 

    M2_testing_array = []
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(17):
            testing_j = testing_array_i[j]
            for k in range(17):
                if k >= j:
                    testing_k = testing_array_i[k]
                    testing_data = testing_j * testing_k
                    M2_testing_array.append(testing_data)
    M2_testing_array = np.reshape(M2_testing_array,(121, 153))
    M2_testing_array = M2_testing_array.T

    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)

    #垂直串接
    training_array = np.vstack([x_training_array, training_array]) 
    training_array = np.vstack([training_array, M2_training_array])
    testing_array = np.vstack([x_testing_array, testing_array])
    testing_array = np.vstack([testing_array, M2_testing_array])

    x = training_array.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    ##############################
    y_test_predict = np.dot(weight, testing_array)
    E_test = y_test_predict - test_testing_array
    E_test_2 = E_test**2
    test_RMSE = ((1/(2*121))*E_test_2.sum())**0.5
    original_RMSE.append(test_RMSE)

# D=5
for i in range(9):
    testing_array = data_array_d5[:,i*num_test_samples:(i+1)*num_test_samples] # 5*121
    test_testing_array = test_array[:,i*num_test_samples:(i+1)*num_test_samples]
    training_array = np.hstack([data_array_d5[:,:i*num_test_samples],data_array_d5[:,(i+1)*num_test_samples:]]) #5*975
    test_training_array = np.hstack([test_array[:,:i*num_test_samples],test_array[:,(i+1)*num_test_samples:]])
    ##############################
    M2_training_array = []
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(5):
            training_j = training_array_i[j]
            for k in range(5):
                if k >= j:
                    training_k = training_array_i[k]
                    training_data = training_j * training_k
                    M2_training_array.append(training_data)

    M2_training_array = np.reshape(M2_training_array,(975, 15))
    M2_training_array = M2_training_array.T 

    M2_testing_array = []
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(5):
            testing_j = testing_array_i[j]
            for k in range(5):
                if k >= j:
                    testing_k = testing_array_i[k]
                    testing_data = testing_j * testing_k
                    M2_testing_array.append(testing_data)
    M2_testing_array = np.reshape(M2_testing_array,(121, 15))
    M2_testing_array = M2_testing_array.T

    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)

    #垂直串接
    training_array = np.vstack([x_training_array, training_array]) 
    training_array = np.vstack([training_array, M2_training_array])
    testing_array = np.vstack([x_testing_array, testing_array])
    testing_array = np.vstack([testing_array, M2_testing_array])

    x = training_array.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    ##############################
    y_test_predict = np.dot(weight, testing_array)
    E_test = y_test_predict - test_testing_array
    E_test_2 = E_test**2
    test_RMSE = ((1/(2*121))*E_test_2.sum())**0.5
    d5_RMSE.append(test_RMSE)

# D=12
for i in range(9):
    testing_array = data_array_d12[:,i*num_test_samples:(i+1)*num_test_samples] # 5*121
    test_testing_array = test_array[:,i*num_test_samples:(i+1)*num_test_samples]
    training_array = np.hstack([data_array_d12[:,:i*num_test_samples],data_array_d12[:,(i+1)*num_test_samples:]]) #5*975
    test_training_array = np.hstack([test_array[:,:i*num_test_samples],test_array[:,(i+1)*num_test_samples:]])
    ##############################
    M2_training_array = []
    for i in range(975):
        training_array_i = training_array[:,i]
        for j in range(12):
            training_j = training_array_i[j]
            for k in range(12):
                if k >= j:
                    training_k = training_array_i[k]
                    training_data = training_j * training_k
                    M2_training_array.append(training_data)
    M2_training_array = np.reshape(M2_training_array,(975, 78))
    M2_training_array = M2_training_array.T 

    M2_testing_array = []
    for i in range(121):
        testing_array_i = testing_array[:,i]
        for j in range(12):
            testing_j = testing_array_i[j]
            for k in range(12):
                if k >= j:
                    testing_k = testing_array_i[k]
                    testing_data = testing_j * testing_k
                    M2_testing_array.append(testing_data)
    M2_testing_array = np.reshape(M2_testing_array,(121, 78))
    M2_testing_array = M2_testing_array.T

    x_training_array = np.ones(975)
    x_testing_array = np.ones(121)

    #垂直串接
    training_array = np.vstack([x_training_array, training_array]) 
    training_array = np.vstack([training_array, M2_training_array])
    testing_array = np.vstack([x_testing_array, testing_array])
    testing_array = np.vstack([testing_array, M2_testing_array])

    x = training_array.T
    y = test_training_array.T
    w = linear_regression(x, y)
    weight = w.T
    ##############################
    y_test_predict = np.dot(weight, testing_array)
    E_test = y_test_predict - test_testing_array
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

##############################