import numpy as np
from sklearn.model_selection import train_test_split
import math
import pandas as pd
from data import *
# Generate all sample data
def generate_attribute_value(data_size, dimensional):  # data_size 是样本大小 dimensional 是有几个属性
    Data = np.multiply(np.random.random((data_size, dimensional)), np.random.uniform(0, 1, (1, dimensional)))
    np.savetxt('./simulation_data.csv', Data, delimiter=',')
    return Data

# Assume polynomial marginal functions with some degrees
def marginalvalue(Data, degree):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree) # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        for j in range(1, degree + 1):    # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            New_Data.append(np.array(Data[:, i] ** j))
    New_Data = (np.array(New_Data).T) # 按照degree取幂
    # print(Data[:,0] ** 3 == New_Data[:,2]) is true
    # Data[:,0] ** 2 == New_Data[:,1] is true
    # Data[:,1] ** 1 == New_Data[:,3]
    global_value = np.dot(New_Data, para)
    # global_value = global_value + np.random.random_sample([Data.shape[0],])
    global_value = global_value + np.random.normal(loc=0.0, scale=.50, size=[Data.shape[0], ])
    np.savetxt('./global_value.csv', global_value, delimiter=',')
    return New_Data, global_value.reshape(-1, 1), para # New_data: 取幂后的边际效用 global_value: regression的目标
# For regression
def marginalvalue_with_interaction(Data, degree, pair_num):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree) # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    marginal_value = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        marginalvalue_without_para = []
        for j in range(1, degree + 1):    # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            New_Data.append(np.array(Data[:, i] ** j))
            # marginal values
            marginalvalue_without_para.append(Data[:, i] ** j)
        marginal_value_temp = np.dot(np.array(marginalvalue_without_para).T, para[i * degree : (i+1) * degree])
        marginal_value.append(marginal_value_temp)
    New_Data = (np.array(New_Data).T) # 按照degree取幂
    marginal_value = np.array(marginal_value)
    # print(marginal_value.T.shape)
    # Interactions
    marginal_value = marginal_value.T
    for inter_index in range(pair_num):
        rand_1 = np.random.randint(0, marginal_value.shape[1], 1)
        rand_2 = np.random.randint(0, marginal_value.shape[1], 1)
        inter_temp = marginal_value[:,rand_1] * marginal_value[:, rand_2]
        marginal_value = np.concatenate((marginal_value, inter_temp), axis=1)
    # print(marginal_value.shape)
    global_value = np.sum(marginal_value, axis=1)
    # global_value = global_value + np.random.random_sample([Data.shape[0], ])
    global_value = global_value + np.random.normal(loc=0.0, scale=.50, size=[Data.shape[0], ])
    # global_value = np.dot(New_Data, para)
    # global_value = global_value + np.random.random_sample([Data.shape[0], ])
    # global_value = global_value + np.random.random_sample([Data.shape[0],]) * (np.random.random_sample([Data.shape[0],])-0.5)
    np.savetxt('./global_value.csv', global_value, delimiter=',')
    return New_Data, global_value.reshape(-1, 1), para # New_data: 取幂后的边际效用 global_value: regression的目标

# For classification
def marginalvalue_with_interaction_classifcation(Data, degree, class_num, pair_num = None):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree) # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    marginal_value = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        marginalvalue_without_para = []
        for j in range(1, degree + 1):    # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            New_Data.append(np.array(Data[:, i] ** j))
            # marginal values
            marginalvalue_without_para.append(Data[:, i] ** j)
        marginal_value_temp = np.dot(np.array(marginalvalue_without_para).T, para[i * degree : (i+1) * degree])
        marginal_value.append(marginal_value_temp)
    New_Data = (np.array(New_Data).T) # 按照degree取幂
    marginal_value = np.array(marginal_value)
    # print(marginal_value.T.shape)
    # Interactions
    marginal_value = marginal_value.T
    if pair_num is not None:
        for inter_index in range(pair_num):
            rand_1 = np.random.randint(0, marginal_value.shape[1], 1)
            rand_2 = np.random.randint(0, marginal_value.shape[1], 1)
            inter_temp = marginal_value[:,rand_1] * marginal_value[:, rand_2]
            marginal_value = np.concatenate((marginal_value, inter_temp), axis=1)
        # print(marginal_value.shape)
    global_value = np.sum(marginal_value, axis=1)
    all_data = np.concatenate((np.array(Data), marginal_value, np.reshape(global_value, (-1, 1))), axis=1)
    all_data = all_data[np.lexsort(all_data.T)] # Last COL is global value
    classes = np.arange(0, class_num)
    class_list = classes.repeat(math.ceil(all_data.shape[0]/class_num))
        # print(class_list)
        # print(len(class_list))
    class_list = class_list[0:all_data.shape[0]]
    # all_data = np.concatenate((all_data, np.reshape(class_list, (-1,1))), axis=1)
    Simulation_data = np.concatenate(( np.reshape(class_list, (-1,1)), all_data[:,0:np.array(Data).shape[1]],), axis=1)
    # class_list = class_list[:, np.newaxis]
    # Data = np.concatenate((np.array(Data), np.reshape(class_list, (-1,1))), axis= 1)
    # global_value = np.dot(New_Data, para)
    # global_value = global_value + np.random.random_sample([Data.shape[0], ])
    # global_value = global_value + np.random.random_sample([Data.shape[0],]) * (np.random.random_sample([Data.shape[0],])-0.5)
    np.savetxt('./global_value.csv', Simulation_data[:,0], delimiter=',')
    np.savetxt('./simulation.csv', Simulation_data[:,1:], delimiter=',')
    return Simulation_data[:,1:],  Simulation_data[:,0], para# New_data: 取幂后的边际效用 global_value: regression的目标

# For binary classification
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def marginalvalue_with_interaction_4_binary_classifcation(Data, degree,pair_num = None):
    para = np.random.uniform(-1, 1, Data.shape[1] * degree)  # 生成多项式函数的参数 一共有 dimension * degree 个
    New_Data = []
    marginal_value = []
    for i in range(Data.shape[1]):  # i is from 0 to dimensions
        marginalvalue_without_para = []
        for j in range(1, degree + 1):  # 用低微数据拟合高维多项式  for j in range(1, degree - 2): degree should be greater than 3
            New_Data.append(np.array(Data[:, i] ** j))
            # marginal values
            marginalvalue_without_para.append(Data[:, i] ** j)
        marginal_value_temp = np.dot(np.array(marginalvalue_without_para).T, para[i * degree: (i + 1) * degree])
        marginal_value.append(marginal_value_temp)
    New_Data = (np.array(New_Data).T)  # 按照degree取幂
    marginal_value = np.array(marginal_value)
    # print(marginal_value.T.shape)
    # Interactions
    marginal_value = marginal_value.T
    if pair_num is not None:
        for inter_index in range(pair_num):
            rand_1 = np.random.randint(0, marginal_value.shape[1], 1)
            rand_2 = np.random.randint(0, marginal_value.shape[1], 1)
            inter_temp = marginal_value[:, rand_1] * marginal_value[:, rand_2]
            marginal_value = np.concatenate((marginal_value, inter_temp), axis=1)
        # print(marginal_value.shape)
    global_value = np.sum(marginal_value, axis=1) + np.random.normal(loc=0.0, scale=.50, size = [Data.shape[0], ])
    global_value = np.reshape(global_value, (-1, 1))
    for item in range(len(global_value)):
        if sigmoid(global_value[item]) >=0.5:
            global_value[item] = 1
        else:
            global_value[item] = 0

    all_data = np.concatenate((np.array(Data), marginal_value, np.reshape(global_value, (-1, 1))), axis=1)

    # all_data = np.concatenate((all_data, np.reshape(class_list, (-1,1))), axis=1)
    Simulation_data = np.concatenate((np.reshape(global_value, (-1, 1)), all_data[:, 0:np.array(Data).shape[1]],), axis=1)
    # class_list = class_list[:, np.newaxis]
    # Data = np.concatenate((np.array(Data), np.reshape(class_list, (-1,1))), axis= 1)
    # global_value = np.dot(New_Data, para)
    # global_value = global_value + np.random.random_sample([Data.shape[0], ])
    # global_value = global_value + np.random.random_sample([Data.shape[0],]) * (np.random.random_sample([Data.shape[0],])-0.5)
    np.savetxt('./global_value.csv', Simulation_data[:, 0], delimiter=',')
    np.savetxt('./simulation.csv', Simulation_data[:, 1:], delimiter=',')
    return Simulation_data[:, 1:], Simulation_data[:, 0], para  # New_data: 取幂后的边际效用 global_value: regression的目标


# Piecewise linear interpolation
def piecewise_linear(Data_path, Global_value_path, gamma, task):

    Data = pd.read_csv(Data_path, delimiter=',', header=None)

    # If the first ROW is label
    # data = Data[:, 1:] # Attribute Data
    # label = Data[:, 0] # Label Data
    # If not, block above
    data = Data
    label = np.loadtxt(Global_value_path, delimiter=',')
    row, col = data.shape
    interval_vector = []
    nonlinear_col_name = []
    for j in range(col):  # 对每一列进行操作
        # For binear vars
        # if set(data[:, j]).union({0, 1}) == {0, 1}:
        #     interval_vector.append(np.array([0.0, 1.0]))
        # else:
        max = np.max(data.iloc[:,j])
        min = np.min(data.iloc[:,j])
        # print(min, max)
        interval_vector.append(np.linspace(min, max, gamma +1, endpoint=True))  # for numeric attributes
        nonlinear_col_name.append('Ori_'+str(j+1))
    interval_vector = np.array(interval_vector)

    # Vectorization
    alternative_matrix = []
    linear_col_name = []
    for j in range(col):
        for k in range(len(interval_vector[j]) - 1):
            linear_col_name.append('Vectorized_' + str(j+1) + '_' + str(k+1))
    for i in range(row):  # i - index of alternatives
        alternative_i = []
        for j in range(col):  # j - index of criteria
            # if len(interval_vector[j]) < gamma:
            #     alternative_i.append(data[i][j])
            # else:
            for k in range(len(interval_vector[j]) - 1):  # k - thresholds for j-th criterion
                if interval_vector[j][k + 1] < data.iloc[i, j]:
                    alternative_i.append(1.0)
                elif interval_vector[j][k] <= data.iloc[i, j] <= interval_vector[j][k + 1]:
                    if data.iloc[i, j] == 0:
                        alternative_i.append(1.0)
                    else:
                        alternative_i.append(
                            (data.iloc[i, j] - interval_vector[j][k]) / (interval_vector[j][k + 1] - interval_vector[j][k]))
                else:
                    alternative_i.append(0.0)

        # alternative_i.extend(data.iloc[i,:])
        # alternative_i.append(label[i])
        alternative_matrix.append(np.array(alternative_i))
    df_num_linear = pd.DataFrame(np.array(alternative_matrix), columns=linear_col_name)
    data.columns = nonlinear_col_name
    df_label = pd.DataFrame(label, columns=['Label'])
    df_input = pd.concat([df_num_linear, data, df_label], axis=1)
    predictor_headers = df_input.columns.values[:].tolist()
    predictor_headers.remove('Label')

    all_folds = cv_split_data(
        df_input,
        predictor_headers,
        outcome_header='Label',
        task=task,
        dummy_cols=None,
        num_folds=5,
    )

    return interval_vector, all_folds  # alternative_matrix 是所有的数据 sample_data是preference information 最后一列是label

# if __name__ == '__main__':
#     Data = generate_attribute_value(1000, 20)
#     New_Data, global_value, para = marginalvalue(Data, degree=3)
#     interval_vector, all_folds  = piecewise_linear('./simulation_data.csv', './global_value.csv',5, 0.2)

    ass = 1
# Data = generate_attribute_value(1000, 20)
# New_Data, global_value, para = marginalvalue_with_interaction(Data, 3, 20)
