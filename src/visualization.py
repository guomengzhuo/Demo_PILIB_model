import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import torch
from models import *
sys.path.append("../../src")
plt.style.use('classic')
import seaborn
def bank_marketing(path, gamma):
    df = pd.read_csv(path + "bank_marketing/bank-additional-full-preprocess-centralized.csv",delimiter=',')
    outcome_header = "y"
    binary_predictor = []  # 所有二元变量列名
    cate_predictor = [] # 所有类别变量列名
    orig_predictior_name = list(df)
    df2 = df[orig_predictior_name]
    orig_predictior_name.remove(outcome_header)
    num_predictor = list(df)
    num_predictor.remove(outcome_header) # 所有连续变量列名
    for col_name in orig_predictior_name:
        if len(list(set(df[col_name])))<=2:  # 记录所二元0-1变量
            if sorted(list(set(df[col_name]))) != [0.0, 1.0]:
                df[col_name] = df[col_name].map({sorted(list(set(df[col_name])))[0]: 0.0, sorted(list(set(df[col_name])))[1]: 1.0})
            binary_predictor.append(col_name)
            num_predictor.remove(col_name)
        elif len(list(set(df[col_name])))<=10: # 指定所有小于10个unique values的为cate变量
            # df[col_name] = df[col_name].replace(get_mapping(df, col_name))
            cate_predictor.append(col_name)
            num_predictor.remove(col_name)
    # print(df)
    print(binary_predictor)
    print(cate_predictor)
    print(num_predictor)
    print(orig_predictior_name)
    # 对于二元变量 没有操作 训练的bias是为0时的marginal value weight为1时的marginal value
    if binary_predictor is not None:
        vectorized_bin_col_name = []
        for col_name in binary_predictor:
            vectorized_bin_col_name.append(col_name+'_bin')
        df_linear_input_bin =  pd.DataFrame(df[binary_predictor], columns=vectorized_bin_col_name)


    # For categorical variables
    if cate_predictor is not None:
        interval_vector_cat = []
        vectorized_cate_col_name = []  # new col names
        # df3 = df[cate_predictor]
        vectorized_cate_col_name_num_list = []  # 记录每个类别变量被几个点分割
        alternative_matrix = []
        for col_name in cate_predictor:
            col_val = sorted(list(set(df[col_name])))  # USED ORIGINAL DATA
            print(col_val)
            vectorized_cate_col_name_num_list.append(len(col_val) - 1)
            interval_vector_cat.append(col_val)
            for index in range(len(col_val) - 1):
                vectorized_cate_col_name.append(col_name + '_cat_vec_' + str(index))
        for i in range(df.shape[0]):
            alternative_i = []
            for j, col_name in enumerate(cate_predictor):
                for k in range(len(interval_vector_cat[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_cat[j][k + 1] < df[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_cat[j][k] <= df[col_name][i] <= interval_vector_cat[j][k + 1]:
                        if df[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df[col_name][i] - interval_vector_cat[j][k]) / (
                                            interval_vector_cat[j][k + 1] - interval_vector_cat[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        for col_name in cate_predictor:
            orig_predictior_name.remove(col_name)  # 从df3中把categorical attribute移除
        df_linear_input_cat = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_cate_col_name)
    print(vectorized_cate_col_name)
    print(vectorized_cate_col_name_num_list)

    # 对于连续变量
    if num_predictor is not None:
        interval_vector_num = []
        vectorized_num_col_name = []
        alternative_matrix = []

        for col_name in num_predictor:
            max = np.max(df[col_name])
            min = np.min(df[col_name])
            interval_vector_num.append(np.linspace(min, max, gamma + 1, endpoint=True))
            for index in range(gamma):
                vectorized_num_col_name.append(col_name + '_num_vec_' + str(index))
        interval_vector_num = np.array(interval_vector_num)
        # Vectorization
        for i in range(df.shape[0]):  # i - index of alternatives
            alternative_i = []
            for j, col_name in enumerate(num_predictor):  # j - index of criteria
                for k in range(len(interval_vector_num[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_num[j][k + 1] < df[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_num[j][k] <= df[col_name][i] <= interval_vector_num[j][k + 1]:
                        if df[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df[col_name][i] - interval_vector_num[j][k]) / (
                                        interval_vector_num[j][k + 1] - interval_vector_num[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        df_linear_input_num = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_num_col_name)

    # df2 是放入block模块的 df_linear_input 是放入线性模块的
    if num_predictor is not None and cate_predictor is not None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_cat, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is not None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_cat, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_num, df2], axis=1)
    if num_predictor is None and cate_predictor is not None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_cat, df2], axis=1)
    if num_predictor is None and cate_predictor is None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df2], axis=1)
    if num_predictor is None and cate_predictor is not None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_cat, df_linear_input_num, df2], axis=1)
    predictor_headers = list(df_input) #.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    # print(len(vectorized_bin_col_name), len(vectorized_cate_col_name), len(vectorized_num_col_name))
    return binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num


def spambase(path, gamma):
    col_name_list = ['word_freq_make',
                     'word_freq_address',
                     'word_freq_all',
                     'word_freq_3d',
                     'word_freq_our',
                     'word_freq_over',
                     'word_freq_remove',
                     'word_freq_internet',
                     'word_freq_order',
                     'word_freq_mail',
                     'word_freq_receive',
                     'word_freq_will',
                     'word_freq_people',
                     'word_freq_report',
                     'word_freq_addresses',
                     'word_freq_free',
                     'word_freq_business',
                     'word_freq_email',
                     'word_freq_you',
                     'word_freq_credit',
                     'word_freq_your',
                     'word_freq_font',
                     'word_freq_000',
                     'word_freq_money',
                     'word_freq_hp',
                     'word_freq_hpl',
                     'word_freq_george',
                     'word_freq_650',
                     'word_freq_lab',
                     'word_freq_labs',
                     'word_freq_telnet',
                     'word_freq_857',
                     'word_freq_data',
                     'word_freq_415',
                     'word_freq_85',
                     'word_freq_technology',
                     'word_freq_1999',
                     'word_freq_parts',
                     'word_freq_pm',
                     'word_freq_direct',
                     'word_freq_cs',
                     'word_freq_meeting',
                     'word_freq_original',
                     'word_freq_project',
                     'word_freq_re',
                     'word_freq_edu',
                     'word_freq_table',
                     'word_freq_conference',
                     'char_freq_colon',
                     'char_freq_brackets',
                     'char_freq_square_brackets',
                     'char_freq_exclamation_mark',
                     'char_freq_dollar_mark',
                     'char_freq_hashtag',
                     'capital_run_length_average',
                     'capital_run_length_longest',
                     'capital_run_length_total',
                     'label']
    df = pd.read_csv(path + 'spambase/spambase.data', delimiter=',', header=None, names=col_name_list)
    outcome_header = "label"
    binary_predictor = []  # 所有二元变量列名
    cate_predictor = [] # 所有类别变量列名
    orig_predictior_name = list(df)
    df2 = df[orig_predictior_name]
    orig_predictior_name.remove(outcome_header)
    num_predictor = list(df)
    num_predictor.remove(outcome_header) # 所有连续变量列名
    for col_name in orig_predictior_name:
        if len(list(set(df[col_name])))<=2:  # 记录所二元0-1变量
            if sorted(list(set(df[col_name]))) != [0.0, 1.0]:
                df[col_name] = df[col_name].map({sorted(list(set(df[col_name])))[0]: 0.0, sorted(list(set(df[col_name])))[1]: 1.0})
            binary_predictor.append(col_name)
            num_predictor.remove(col_name)
        elif len(list(set(df[col_name])))<=3: # 指定所有小于10个unique values的为cate变量
            # df[col_name] = df[col_name].replace(get_mapping(df, col_name))
            cate_predictor.append(col_name)
            num_predictor.remove(col_name)
    # print(df)
    print(binary_predictor)
    print(cate_predictor)
    print(num_predictor)
    print(orig_predictior_name)
    # 对于二元变量 没有操作 训练的bias是为0时的marginal value weight为1时的marginal value
    if binary_predictor is not None:
        vectorized_bin_col_name = []
        for col_name in binary_predictor:
            vectorized_bin_col_name.append(col_name+'_bin')
        df_linear_input_bin =  pd.DataFrame(df[binary_predictor], columns=vectorized_bin_col_name)


    # For categorical variables
    if cate_predictor is not None:
        interval_vector_cat = []
        vectorized_cate_col_name = []  # new col names
        # df3 = df[cate_predictor]
        vectorized_cate_col_name_num_list = []  # 记录每个类别变量被几个点分割
        alternative_matrix = []
        for col_name in cate_predictor:
            col_val = sorted(list(set(df[col_name])))  # USED ORIGINAL DATA
            print(col_val)
            vectorized_cate_col_name_num_list.append(len(col_val) - 1)
            interval_vector_cat.append(col_val)
            for index in range(len(col_val) - 1):
                vectorized_cate_col_name.append(col_name + '_cat_vec_' + str(index))
        for i in range(df.shape[0]):
            alternative_i = []
            for j, col_name in enumerate(cate_predictor):
                for k in range(len(interval_vector_cat[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_cat[j][k + 1] < df[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_cat[j][k] <= df[col_name][i] <= interval_vector_cat[j][k + 1]:
                        if df[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df[col_name][i] - interval_vector_cat[j][k]) / (
                                            interval_vector_cat[j][k + 1] - interval_vector_cat[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        for col_name in cate_predictor:
            orig_predictior_name.remove(col_name)  # 从df3中把categorical attribute移除
        df_linear_input_cat = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_cate_col_name)
    print(vectorized_cate_col_name)
    print(vectorized_cate_col_name_num_list)

    # 对于连续变量
    if num_predictor is not None:
        interval_vector_num = []
        vectorized_num_col_name = []
        alternative_matrix = []

        for col_name in num_predictor:
            max = np.max(df[col_name])
            min = np.min(df[col_name])
            interval_vector_num.append(np.linspace(min, max, gamma + 1, endpoint=True))
            for index in range(gamma):
                vectorized_num_col_name.append(col_name + '_num_vec_' + str(index))
        interval_vector_num = np.array(interval_vector_num)
        # Vectorization
        for i in range(df.shape[0]):  # i - index of alternatives
            alternative_i = []
            for j, col_name in enumerate(num_predictor):  # j - index of criteria
                for k in range(len(interval_vector_num[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_num[j][k + 1] < df[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_num[j][k] <= df[col_name][i] <= interval_vector_num[j][k + 1]:
                        if df[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df[col_name][i] - interval_vector_num[j][k]) / (
                                        interval_vector_num[j][k + 1] - interval_vector_num[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        df_linear_input_num = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_num_col_name)

    # df2 是放入block模块的 df_linear_input 是放入线性模块的
    if num_predictor is not None and cate_predictor is not None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_cat, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is not None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_cat, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_num, df2], axis=1)
    if num_predictor is not None and cate_predictor is None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_num, df2], axis=1)
    if num_predictor is None and cate_predictor is not None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df_linear_input_cat, df2], axis=1)
    if num_predictor is None and cate_predictor is None and binary_predictor is not None:
        df_input = pd.concat([df_linear_input_bin, df2], axis=1)
    if num_predictor is None and cate_predictor is not None and binary_predictor is None:
        df_input = pd.concat([df_linear_input_cat, df_linear_input_num, df2], axis=1)
    predictor_headers = list(df_input) #.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    # print(len(vectorized_bin_col_name), len(vectorized_cate_col_name), len(vectorized_num_col_name))
    return binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num


def bank_marketing_main_effect(model,num_binary,
               num_categorical,
               num_numerical,
               gamma,
               binary_predictor,
               cate_predictor,
               num_predictor,
               interval_vector_cat,
               interval_vector_num,
               categorical_list=None):
    if categorical_list is not None:
        assert np.sum(np.array(categorical_list)) == num_categorical
    model_para = torch.load(model)
    all_para_weight = model_para['model_pwlnet.layer_piecewise.weight'].numpy().ravel()
    all_para_bias = model_para['model_pwlnet.layer_piecewise.bias'].numpy()

    # consider weights
    attr_weights = model_para['model_pwlnet.layer_summation.weight'].numpy().ravel()
    total_cnt_attr = 0
    # seperate different kinds of attributes
    binary_vari_weight = all_para_weight[0:num_binary]
    binary_vari_bias = all_para_bias[0:num_binary]

    categorical_vari_weight = all_para_weight[num_binary:num_categorical+num_binary]
    categorical_vari_bias = all_para_bias[num_binary:num_categorical+num_binary]

    numerical_vari_weight = all_para_weight[num_binary+num_categorical:]
    numerical_vari_bias = all_para_bias[num_binary+num_categorical:]

    # For binary variable
    if num_binary != 0 :
        _binary_variable = []
        _useful_binary_variable = []
        _useful_binary_variable2 = []
        for id, zero_value in enumerate( binary_vari_bias):
            temp_vector = zero_value+binary_vari_weight[id]
            temp_vector2 = temp_vector * attr_weights[total_cnt_attr]
            # attr + 1
            total_cnt_attr += 1
            # temp_vector = binary_vari_weight[id]
            if abs(temp_vector) >= 0.0001:
                _useful_binary_variable.append({binary_predictor[id]:(id, temp_vector)})
            _binary_variable.append(temp_vector)

            if abs(temp_vector2) >= 0.0001:
                _useful_binary_variable2.append({binary_predictor[id]: (id, temp_vector2)})

        print(_useful_binary_variable)

        fig, ax = plt.subplots()
        ax.bar(np.linspace(0,num_binary*2, num_binary,endpoint=True), _binary_variable,0.1)
        if _useful_binary_variable is not None:
            bbox = dict(boxstyle="round", fc="0.8")
            offset = 12
            arrowprops = dict(
                arrowstyle="->",
                connectionstyle="angle,angleA=0,angleB=90,rad=10")
            for item in _useful_binary_variable:
                for key in item:
                    print(key)
            #         ax.annotate(key, xy=(item[key][0],item[key][1]),xytext=(-2*offset, offset),xycoords='data',textcoords='offset points',
            # bbox=bbox, arrowprops=arrowprops)
                    ax.text(2*item[key][0], item[key][1], key, fontsize=8,rotation=25)
        plt.show()

        fig, ax = plt.subplots()
        ax.bar(np.linspace(0, num_binary * 2, num_binary, endpoint=True), _binary_variable, 0.1)
        if _useful_binary_variable2 is not None:
            bbox = dict(boxstyle="round", fc="0.8")
            offset = 12
            arrowprops = dict(
                arrowstyle="->",
                connectionstyle="angle,angleA=0,angleB=90,rad=10")
            for item in _useful_binary_variable2:
                for key in item:
                    print(key)
                    #         ax.annotate(key, xy=(item[key][0],item[key][1]),xytext=(-2*offset, offset),xycoords='data',textcoords='offset points',
                    # bbox=bbox, arrowprops=arrowprops)
                    ax.text(2 * item[key][0], item[key][1], key, fontsize=8, rotation=25)
        plt.show()

    print(total_cnt_attr)
    # For categorical variable
    if num_categorical != 0:
        _categorical_variable = []
        _categorical_variable2 = []
        count = 0
        for id, intervals in enumerate(categorical_list):
            temp_vector = []
            bias = np.sum(categorical_vari_bias[count:count+intervals])
            _weight_vector = categorical_vari_weight[count:count+intervals]
            count += intervals
            temp_vector.append(bias)
            _temp = 0
            for i in range(intervals):
                temp_vector.append(bias+_weight_vector[i]+_temp)
                _temp += _weight_vector[i]
            temp_vector2 = [x * attr_weights[total_cnt_attr] for x in temp_vector]
            _categorical_variable.append(temp_vector)
            _categorical_variable2.append(temp_vector2)
            total_cnt_attr += 1
        fig = plt.figure()
        for id, y in enumerate(_categorical_variable):
            ax = fig.add_subplot(len(_categorical_variable), 1, id+1)
            if id == 2:
                ax.plot(interval_vector_cat[id], y, 'xb')
            else:
                ax.bar(interval_vector_cat[id], y, 0.1 / len(interval_vector_cat[id]))
            ax.set_xlabel(cate_predictor[id])
        # plt.subplots_adjust(hspace = 0.5)
        plt.tight_layout()
        plt.show()

        fig = plt.figure()
        for id, y in enumerate(_categorical_variable2):
            ax = fig.add_subplot(len(_categorical_variable2), 1, id + 1)
            if id == 2:
                ax.plot(interval_vector_cat[id], y, 'xb')
            else:
                ax.bar(interval_vector_cat[id], y, 0.1 / len(interval_vector_cat[id]))
            ax.set_xlabel(cate_predictor[id])
        # plt.subplots_adjust(hspace = 0.5)
        plt.tight_layout()
        plt.show()
    print(total_cnt_attr)

    # For numerical variable
    if num_numerical != 0:
        assert num_numerical % gamma == 0
        _attr = int(num_numerical/gamma)
        _numerical_variable = []
        _numerical_variable2 = []
        count = 0
        for id in range(_attr):
            temp_vector = []
            bias = np.sum(numerical_vari_bias[count:count+gamma])
            _weight_vector = numerical_vari_weight[count:count+gamma]
            count += gamma
            temp_vector.append(bias)
            _temp=0
            for i in range(gamma):
                temp_vector.append(bias+_weight_vector[i]+_temp)
                _temp += _weight_vector[i]
            temp_vector2 = [x * attr_weights[total_cnt_attr] for x in temp_vector]
            total_cnt_attr += 1
            _numerical_variable.append((temp_vector))
            _numerical_variable2.append((temp_vector2))
        fig = plt.figure()
        for id, y in enumerate(_numerical_variable):
            if len(_numerical_variable)%2 == 0:
                ax = fig.add_subplot(len(_numerical_variable)//2, 2, id+1)
                ax.plot(interval_vector_num[id], y)
                ax.set_xlabel(num_predictor[id])
            else:
                ax = fig.add_subplot(len(_numerical_variable) // 2 + 1, 2, id + 1)
                ax.plot(interval_vector_num[id], y)
                ax.set_xlabel(num_predictor[id])
        plt.tight_layout()
        # plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        plt.show()

        fig = plt.figure()
        for id, y in enumerate(_numerical_variable2):
            if len(_numerical_variable) % 2 == 0:
                ax = fig.add_subplot(len(_numerical_variable2) // 2, 2, id + 1)
                ax.plot(interval_vector_num[id], y)
                ax.set_xlabel(num_predictor[id])
            else:
                ax = fig.add_subplot(len(_numerical_variable2) // 2 + 1, 2, id + 1)
                ax.plot(interval_vector_num[id], y)
                ax.set_xlabel(num_predictor[id])
        plt.tight_layout()
        # plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        plt.show()
    print(total_cnt_attr)

def skill(path, gamma):
    col_name_list = ['response', 'Age', 'HoursPerWeek', 'TotalHours', 'APM', 'SelectByHotkeys', 'AssignToHotkeys',
                     'UniqueHotkeys', 'MinimapAttacks', 'MinimapRightClicks', 'NumberOfPACs', 'GapBetweenPACs',
                     'ActionLatency', 'ActionsInPAC', 'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade',
                     'ComplexUnitsMade', 'ComplexAbilitiesUsed'
                     ]
    df = pd.read_csv(path + "skill/skill_bin.csv", delimiter=',', header=None, names=col_name_list)  # iloc[57] is label
    # df = pd.read_csv("./dataset/spambase.data", delimiter=',', header=None, names=col_name_list)
    outcome_header = "response"
    # print(df.describe())
    orig_predictior_name = list(df)
    df2 = df[orig_predictior_name]
    orig_predictior_name.remove(outcome_header)
    binary_predictor = []
    cate_predictor = []
    num_predictor = list(df)
    num_predictor.remove(outcome_header)  # 所有连续变量列名
    interval_vector_num = []
    vectorized_num_col_name = []
    alternative_matrix = []
    if cate_predictor != []:
        interval_vector_cat = []
        vectorized_cate_col_name = []  # new col names
        # df3 = df[cate_predictor]
        vectorized_cate_col_name_num_list = []  # 记录每个类别变量被几个点分割
        alternative_matrix = []
        for col_name in cate_predictor:
            col_val = sorted(list(set(df[col_name])))  # USED ORIGINAL DATA
            print(col_val)
            vectorized_cate_col_name_num_list.append(len(col_val) - 1)
            interval_vector_cat.append(col_val)
            for index in range(len(col_val) - 1):
                vectorized_cate_col_name.append(col_name + '_cat_vec_' + str(index))
        for i in range(df.shape[0]):
            alternative_i = []
            for j, col_name in enumerate(cate_predictor):
                for k in range(len(interval_vector_cat[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_cat[j][k + 1] < df[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_cat[j][k] <= df[col_name][i] <= interval_vector_cat[j][k + 1]:
                        if df[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df[col_name][i] - interval_vector_cat[j][k]) / (
                                        interval_vector_cat[j][k + 1] - interval_vector_cat[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        for col_name in cate_predictor:
            orig_predictior_name.remove(col_name)  # 从df3中把categorical attribute移除
        df_linear_input_cat = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_cate_col_name)
    else:
        vectorized_cate_col_name_num_list = []

    for col_name in num_predictor:
        max = np.max(df[col_name])
        min = np.min(df[col_name])
        interval_vector_num.append(np.linspace(min, max, gamma + 1, endpoint=True))
        for index in range(gamma):
            vectorized_num_col_name.append(col_name + '_num_vec_' + str(index))
    interval_vector_num = np.array(interval_vector_num)
    # Vectorization
    for i in range(df.shape[0]):  # i - index of alternatives
        alternative_i = []
        for j, col_name in enumerate(num_predictor):  # j - index of criteria
            for k in range(len(interval_vector_num[j]) - 1):  # k - thresholds for j-th criterion
                if interval_vector_num[j][k + 1] < df[col_name][i]:
                    alternative_i.append(1.0)
                elif interval_vector_num[j][k] <= df[col_name][i] <= interval_vector_num[j][k + 1]:
                    if df[col_name][i] == 0:
                        alternative_i.append(1.0)
                    else:
                        alternative_i.append(
                            (df[col_name][i] - interval_vector_num[j][k]) / (
                                    interval_vector_num[j][k + 1] - interval_vector_num[j][k]))
                else:
                    alternative_i.append(0.0)
        alternative_matrix.append(np.array(alternative_i))
    df_linear_input_num = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_num_col_name)
    df_input = pd.concat([df_linear_input_num, df2], axis=1)
    predictor_headers = list(df_input)  # .columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    interval_vector_cat = None
    return binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num

def get_subnets(trained_block_model: BlockMainEffectsMLP):
    n_group = trained_block_model._n_group

    layers = defaultdict(list)  # type: dict # 防止产生不存在的key报错
    for layer in trained_block_model.layers:
        layers[layer.__class__.__name__].append(layer)

    block_layers = layers[GroupBlockLayer.__name__]
    pnets_params = [
        [(l.block_weight[i].data, l.block_bias[i].data) for l in block_layers]
        for i in range(n_group)
    ]

    assert len(layers[GroupL0Linear.__name__]) == 1
    base_net = layers[GroupL0Linear.__name__][0]
    mask = base_net.trained_mask   #  layer_l0.py : self.register_buffer("trained_mask", torch.zeros(self._origin.weight.size()))???
    w = base_net._origin.weight.data
    w = torch.mul(mask, w) # 权重*Gate variable
    try:
        b = base_net._origin.bias.data
    except Exception:
        b = None

    size = w.size()[0] // n_group
    for i in range(n_group):
        param = (
            w[size * i : size * (i + 1)],
            None if b is None else b[size * i : size * (i + 1)],
        )
        pnets_params[i].insert(0, param)

    _subnets = [FixedParaMLP(pnets_params[i]) for i in range(n_group)]
    return _subnets


class FixedParaMLP(nn.Module):
    def __init__(self, params):
        super(FixedParaMLP, self).__init__()

        self._ws = []
        self._bs = []
        for i in range(len(params)):
            w, b = params[i]
            if i == 0:
                w = w.numpy()
                b = b.numpy()
            else:
                w = w[0]
                b = b[0]
            self._ws.append(Variable(  torch.FloatTensor(w), requires_grad=False))
            self._bs.append(
                Variable(torch.FloatTensor(b), requires_grad=False)
                if b is not None
                else None
            )

    def forward(self, x):
        for i in range(len(self._ws)):
            x = F.linear(x, self._ws[i], bias=self._bs[i])
            if i != len(self._ws) - 1:
                x = F.relu(x)
        return x



# binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num = bank_marketing("../dataset/", gamma=5)
# binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num = spambase("../dataset/", gamma=5)
binary_predictor, cate_predictor, num_predictor, interval_vector_cat, interval_vector_num = skill("../dataset/", gamma=5)


# Data, bank markeing main effect
# bank_marketing_main_effect("4.pt", num_binary=37, num_categorical=22, num_numerical=40, gamma=5,
#            binary_predictor=binary_predictor,
#            cate_predictor=cate_predictor,
#            num_predictor=num_predictor,
#            interval_vector_cat=interval_vector_cat,
#            interval_vector_num=interval_vector_num,
#            categorical_list=[6,7,9])




def spambase_heatmap(saved_model, n_group, input_attr):
    model = torch.load(saved_model)
    origin_weight = model['model_block.layers.0._origin.weight']
    # print(origin_weight[0])

    loc = model['model_block.layers.0.loc'].data.numpy()
    # print(loc[0])

    mask = model['model_block.layers.0.trained_mask'].data.numpy()
    # print(mask[1])
    mask = np.multiply(mask, origin_weight.data.numpy())
    non_zero_index = []
    _temp = []
    for id, neural in enumerate(mask):  # id: 第id个neural neural: 列表
        index = neural.nonzero()
        _temp.append(list(index))
        if (id + 1) % (800 / n_group) == 0:
            non_zero_index.append(index)
            _temp = []

    final_pair = {} # save pairwise interaction
    for id in range(0, n_group):
        if len(non_zero_index[id][0]) == 2:
            final_pair['block_' + str(id)] = non_zero_index[id][0]
    print(final_pair)
    layers_weight = defaultdict(list)  # type: dict # 防止产生不存在的key报错
    layers_bias = defaultdict(list)
    for key in model:
        if '_block_w' in key:
            layers_weight[key].append(model[key].data)
        if '_block_b' in key:
            layers_bias[key].append(model[key].data)

    pnets_params = []
    for i in range(n_group):  # i: blocks, 1 - 20
        pnets_params_block = []
        for layer_id in [3, 5, 7, 9]:
            weight_name = 'model_block.layers.' + str(layer_id) + '._block_w_' + str(i)
            bias_name = 'model_block.layers.' + str(layer_id) + '._block_b_' + str(i)
            # print(weight_name)
            # print(bias_name)
            pnets_params_block.append((layers_weight[weight_name], layers_bias[bias_name]))
        pnets_params.append(pnets_params_block)
    mask = model['model_block.layers.0.trained_mask']
    w = origin_weight.data
    w = torch.mul(mask, w)
    b = model['model_block.layers.0._origin.bias'].data

    size = w.size()[0] // n_group
    for i in range(n_group):
        param = (
            w[size * i: size * (i + 1)],
            None if b is None else b[size * i: size * (i + 1)],
        )
        pnets_params[i].insert(0, param)
    _subnets = [FixedParaMLP(pnets_params[i]) for i in range(n_group)]
    predictor = binary_predictor + cate_predictor + num_predictor
    for key in final_pair:
        print(predictor[final_pair[key][0]], predictor[final_pair[key][1]], )
        _model = _subnets[int(key[-1])]
        data = torch.rand(1, input_attr)
        result_mat = []
        if np.max(interval_vector_num[final_pair[key][0]]) > 100:
            attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]), 50, 100)
        else:
            attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]),
                                                          np.max(interval_vector_num[final_pair[key][0]]), 100)
        # attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]),
        #                        np.max(interval_vector_num[final_pair[key][0]]), 100)
        print(np.min(interval_vector_num[final_pair[key][0]]), np.max(interval_vector_num[final_pair[key][0]]))
        if np.max(interval_vector_num[final_pair[key][1]]) > 100:
            attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]), 50, 100)
        else:
            attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]),
                                                          np.max(interval_vector_num[final_pair[key][1]]), 100)
        # attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]),
        #                        np.max(interval_vector_num[final_pair[key][1]]), 100)
        # attr2 = torch.linspace(0, 100, 100)
        print(np.min(interval_vector_num[final_pair[key][1]]), np.max(interval_vector_num[final_pair[key][1]]))
        for id1 in attr1:
            data[:, final_pair[key][0]] = id1
            _temp_mat = []
            for id2 in attr2:
                data[:, final_pair[key][1]] = id2
                _temp_mat.append(np.ravel(_model(Variable(data)).data.numpy().tolist())[0])
            result_mat.append(_temp_mat)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = plt.imshow(result_mat)
        # plt.imshow(result_mat)
        # ax.set_xlim(0, len(interval_vector_num[final_pair[key][1]]))
        # ax.set_ylim(0, len(interval_vector_num[final_pair[key][0]]))
        if np.max(interval_vector_num[final_pair[key][1]]) > 100:
            ax.set_xticks(range(0, 100, 10))
            ax.set_xticklabels(np.round(np.linspace(1,
                                                    50, 10), 2),
                               rotation=45)

        else:
            ax.set_xticks(range(0, 100, 10))
            ax.set_xticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][1]]),
                                                np.max(interval_vector_num[final_pair[key][1]]), 10), 2), rotation=45)
        if np.max(interval_vector_num[final_pair[key][0]]) > 100:
            ax.set_yticks(range(0, 100, 10))
            ax.set_yticklabels(np.round(np.linspace(1,
                                                    50, 10), 2),rotation=45)
        else:
            ax.set_yticks(range(0, 100, 10))
            ax.set_yticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][0]]),
                                                np.max(interval_vector_num[final_pair[key][0]]), 10), 2))
        ax.set_xlabel(predictor[final_pair[key][1]])
        ax.set_ylabel(predictor[final_pair[key][0]])
        plt.colorbar()
        plt.savefig('./spambase/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.png',
                    format='png')
        plt.savefig('./spambase/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.eps',
                    format='eps', dpi=1000)
        # plt.savefig(
        #     './bank_marketing/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.png',
        #     format='png')
        # plt.savefig(
        #     './bank_marketing/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.eps',
        #     format='eps', dpi=1000)

        plt.show()

# Plot interacting attributes in Data Spambase
# spambase_heatmap('0.pt', 20, 57)


def skill_heatmap(saved_model, n_group, input_attr):
    model = torch.load(saved_model)
    origin_weight = model['model_block.layers.0._origin.weight']
    # print(origin_weight[0])

    loc = model['model_block.layers.0.loc'].data.numpy()
    # print(loc[0])

    mask = model['model_block.layers.0.trained_mask'].data.numpy()
    # print(mask[1])
    mask = np.multiply(mask, origin_weight.data.numpy())
    non_zero_index = []
    _temp = []
    for id, neural in enumerate(mask):  # id: 第id个neural neural: 列表
        index = neural.nonzero()
        _temp.append(list(index))
        if (id + 1) % (300 / n_group) == 0:
            non_zero_index.append(index)
            _temp = []

    final_pair = {} # save pairwise interaction
    for id in range(0, n_group):
        if len(non_zero_index[id][0]) == 2:
            final_pair['block_' + str(id)] = non_zero_index[id][0]
    print(final_pair)
    layers_weight = defaultdict(list)  # type: dict # 防止产生不存在的key报错
    layers_bias = defaultdict(list)
    for key in model:
        if '_block_w' in key:
            layers_weight[key].append(model[key].data)
        if '_block_b' in key:
            layers_bias[key].append(model[key].data)

    pnets_params = []
    for i in range(n_group):  # i: blocks, 1 - 20
        pnets_params_block = []
        for layer_id in [3, 5, 7,]:
            weight_name = 'model_block.layers.' + str(layer_id) + '._block_w_' + str(i)
            bias_name = 'model_block.layers.' + str(layer_id) + '._block_b_' + str(i)
            # print(weight_name)
            # print(bias_name)
            pnets_params_block.append((layers_weight[weight_name], layers_bias[bias_name]))
        pnets_params.append(pnets_params_block)
    mask = model['model_block.layers.0.trained_mask']
    w = origin_weight.data
    w = torch.mul(mask, w)
    b = model['model_block.layers.0._origin.bias'].data

    size = w.size()[0] // n_group
    for i in range(n_group):
        param = (
            w[size * i: size * (i + 1)],
            None if b is None else b[size * i: size * (i + 1)],
        )
        pnets_params[i].insert(0, param)
    _subnets = [FixedParaMLP(pnets_params[i]) for i in range(n_group)]
    predictor = binary_predictor + cate_predictor + num_predictor
    for key in final_pair:
        print(predictor[final_pair[key][0]], predictor[final_pair[key][1]])
        _model = _subnets[int(key[-1])]
        # data = torch.rand(1, input_attr)
        data = torch.zeros(size=(1, input_attr))
        result_mat = []

        attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]),
                                                          np.max(interval_vector_num[final_pair[key][0]]), 1000)
        # attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]),
        #                        np.max(interval_vector_num[final_pair[key][0]]), 100)
        print(np.min(interval_vector_num[final_pair[key][0]]), np.max(interval_vector_num[final_pair[key][0]]))

        attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]),
                                                          np.max(interval_vector_num[final_pair[key][1]]), 1000)
        # attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]),
        #                        np.max(interval_vector_num[final_pair[key][1]]), 100)
        # attr2 = torch.linspace(0, 100, 100)
        print(np.min(interval_vector_num[final_pair[key][1]]), np.max(interval_vector_num[final_pair[key][1]]))
        for id1 in attr1:
            data[:, final_pair[key][0]] = id1
            _temp_mat = []
            for id2 in attr2:
                data[:, final_pair[key][1]] = id2
                _temp_mat.append(np.ravel(_model(Variable(data)).data.numpy().tolist())[0])
                # print(_temp_mat)
            result_mat.append(_temp_mat)
        # print(result_mat)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        im = plt.imshow(result_mat)
        # plt.imshow(result_mat)
        ax.set_xlim(0, len(interval_vector_num[final_pair[key][1]]))
        ax.set_ylim(0, len(interval_vector_num[final_pair[key][0]]))

        ax.set_xticks(range(0, 100, 10))
        ax.set_xticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][1]]),
                                            np.max(interval_vector_num[final_pair[key][1]]), 10), 2), rotation=45)

        ax.set_yticks(range(0, 100, 10))
        ax.set_yticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][0]]),
                                            np.max(interval_vector_num[final_pair[key][0]]), 10), 2))
        ax.set_xlabel(predictor[final_pair[key][1]])
        ax.set_ylabel(predictor[final_pair[key][0]])
        cbar = plt.colorbar()
        cbar.set_clim(np.min(np.array(result_mat)), np.max(np.array(result_mat)))
        print(np.min(np.array(result_mat)), np.max(np.array(result_mat)))
        plt.savefig('./skill/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.png',
                    format='png')
        plt.savefig('./skill/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.eps',
                    format='eps', dpi=1000)
        # plt.savefig(
        #     './bank_marketing/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.png',
        #     format='png')
        # plt.savefig(
        #     './bank_marketing/Attr_' + predictor[final_pair[key][1]] + '_and_Attr_' + predictor[final_pair[key][0]] + '.eps',
        #     format='eps', dpi=1000)

        # plt.show()

# Plot interacting attributes in Data Skill
skill_heatmap('3.pt', 20, 18)

def spambase_maineffect(model, num_numerical, gamma, num_predictor,interval_vector_num):
    model_para = torch.load(model)
    all_para_weight = model_para['model_pwlnet.layer_piecewise.weight'].numpy().ravel()
    all_para_bias = model_para['model_pwlnet.layer_piecewise.bias'].numpy()
    # seperate different kinds of attributes
    numerical_vari_weight = all_para_weight[0:]
    numerical_vari_bias = all_para_bias[0:]
    if num_numerical != 0:
        assert num_numerical % gamma == 0
        _attr = int(num_numerical/gamma)
        _numerical_variable = []
        count = 0
        for id in range(_attr):
            temp_vector = []
            bias = np.sum(numerical_vari_bias[count:count+gamma])
            _weight_vector = numerical_vari_weight[count:count+gamma]
            count += gamma
            temp_vector.append(bias)
            _temp=0
            for i in range(gamma):
                temp_vector.append(bias+_weight_vector[i]+_temp)
                _temp += _weight_vector[i]
            _numerical_variable.append((temp_vector))

        for id, y in enumerate(_numerical_variable):
            fig = plt.figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            ax.plot(interval_vector_num[id], y)
            ax.set_xlabel(num_predictor[id])
            plt.savefig('./spambase/main_effects/Attr_'+num_predictor[id]+'.png',format='png')
            plt.savefig('./spambase/main_effects/Attr_' + num_predictor[id] + '.eps', format='eps',dpi=1000)


# spambase_maineffect('0.pt',57*5,5,num_predictor,interval_vector_num)


#
# model = torch.load('0.pt')
# 'model_block.layers.0.trained_mask'
# # for key in model:
# #     print(key)
#
# # print(model['model_block.layers.0.trained_mask'].shape)
# # print(np.count_nonzero(model['model_block.layers.0.trained_mask'].data.numpy()))
#
# # print(model['model_block.layers.0.loc'].shape)
# # print(np.count_nonzero(model['model_block.layers.0.loc'].data.numpy()))
#
# # print(model['model_block.layers.0._origin.weight'].shape)
# # print(np.count_nonzero(model['model_block.layers.0._origin.weight'].data.numpy()))
#
# origin_weight = model['model_block.layers.0._origin.weight']
# # print(origin_weight[0])
#
# loc = model['model_block.layers.0.loc'].data.numpy()
# # print(loc[0])
#
# mask = model['model_block.layers.0.trained_mask'].data.numpy()
# # print(mask[1])
# mask = np.multiply(mask, origin_weight.data.numpy())
# non_zero_index = []
# _temp = []
# for id, neural in enumerate(mask): # id: 第id个neural neural: 列表
#     index = neural.nonzero()
#     _temp.append(list(index))
#     if (id+1) % (800/20) == 0:
#         non_zero_index.append(index)
#         _temp = []
#
# # print(len(non_zero_index[3][0]))
# final_pair= {}  # save pairwise interaction
# for id in range(0, 20):
#     if len(non_zero_index[id][0]) == 2:
#         final_pair['block_'+str(id)] = non_zero_index[id][0]
# print(final_pair)
#
# n_group = 20
# layers_weight = defaultdict(list)  # type: dict # 防止产生不存在的key报错
# layers_bias = defaultdict(list)
# for key in model:
#     if '_block_w' in key:
#         layers_weight[key].append(model[key].data)
#     if '_block_b' in key:
#         layers_bias[key].append(model[key].data)
#
# # block_layers = layers[GroupBlockLayer.__name__]
# pnets_params = []
# for i in range(n_group): # i: blocks, 1 - 20
#     pnets_params_block = []
#     for layer_id in [3, 5, 7, 9]:
#         weight_name = 'model_block.layers.'+str(layer_id)+'._block_w_' +str(i)
#         bias_name = 'model_block.layers.'+str(layer_id)+'._block_b_' +str(i)
#         # print(weight_name)
#         # print(bias_name)
#         pnets_params_block.append( (layers_weight[weight_name], layers_bias[bias_name])  )
#     pnets_params.append(pnets_params_block)
# mask = model['model_block.layers.0.trained_mask']
# w = origin_weight.data
# w = torch.mul(mask, w)
# b = model['model_block.layers.0._origin.bias'].data
#
# size = w.size()[0] // n_group
# for i in range(n_group):
#     param = (
#         w[size * i: size * (i + 1)],
#         None if b is None else b[size * i: size * (i + 1)],
#     )
#     pnets_params[i].insert(0, param)
# _subnets = [FixedParaMLP(pnets_params[i]) for i in range(n_group)]
# # print(_subnets[0])
# _model = _subnets[3]
# ws = _model._ws[0].data.numpy()
# print(np.nonzero(ws))
# # 0， 15
#
#
# predictor = binary_predictor+cate_predictor+num_predictor
# for key in final_pair:
#     print(predictor[final_pair[key][0]], predictor[final_pair[key][1]], )
#     _model = _subnets[int(key[-1])]
#     data = torch.rand(1, 57)
#     result_mat = []
#     attr1 = torch.linspace(np.min(interval_vector_num[final_pair[key][0]]), np.max(interval_vector_num[final_pair[key][0]]), 100)
#     print(np.min(interval_vector_num[final_pair[key][0]]), np.max(interval_vector_num[final_pair[key][0]]))
#     attr2 = torch.linspace(np.min(interval_vector_num[final_pair[key][1]]), np.max(interval_vector_num[final_pair[key][1]]), 100)
#     print(np.min(interval_vector_num[final_pair[key][1]]), np.max(interval_vector_num[final_pair[key][1]]))
#     for id1 in attr1:
#         data[:, final_pair[key][0]] = id1
#         _temp_mat = []
#         for id2 in attr2:
#             data[:, final_pair[key][1]] = id2
#             _temp_mat.append(np.ravel(_model(Variable(data)).data.numpy().tolist())[0])
#         result_mat.append(_temp_mat)
#     fig = plt.figure(figsize=(6,6))
#     ax = fig.add_subplot(111)
#     im = plt.imshow(result_mat)
#     # plt.imshow(result_mat)
#     # ax.set_xlim(0, len(interval_vector_num[final_pair[key][1]]))
#     # ax.set_ylim(0, len(interval_vector_num[final_pair[key][0]]))
#     ax.set_xticks(range(0, 100, 10))
#     ax.set_xticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][1]]), np.max(interval_vector_num[final_pair[key][1]]), 10), 2), rotation=45)
#     ax.set_yticks(range(0, 100, 10))
#     ax.set_yticklabels(np.round(np.linspace(np.min(interval_vector_num[final_pair[key][0]]), np.max(interval_vector_num[final_pair[key][0]]), 10),2))
#     ax.set_xlabel(predictor[final_pair[key][1]])
#     ax.set_ylabel(predictor[final_pair[key][0]])
#     plt.colorbar()
#     plt.savefig('Attr_'+predictor[final_pair[key][1]]+'_and_Attr_'+predictor[final_pair[key][0]]+'.png',format='png')
#     plt.savefig('Attr_'+predictor[final_pair[key][1]]+'_and_Attr_'+predictor[final_pair[key][0]]+'.eps',format='eps',dpi=1000)
#
# # for i in range(48):
# #     input = torch.ones(1, 48)*10
# #     input[0, i] = torch.tensor(1000)
# #     print(_model((input)))
# # input[0, 0] = 10000
# # input[0, 15] = 10000
#
#
# # print(_model.parameters)
#
# data = torch.rand(1, 57)
#
# print(_model(Variable(data)))
# result_mat = []
# attr1 = torch.linspace(0, 7, 100)
# attr2 = torch.linspace(0, 7, 100)
# for id1 in attr1:
#     data[:, 5] = id1
#     _temp_mat = []
#     for id2 in attr2:
#         data[:, 54] = id2
#         _temp_mat.append(np.ravel(_model(Variable(data)).data.numpy().tolist())[0])
#     result_mat.append(_temp_mat)
# # data[:, 54] = torch.FloatTensor([1]*10)
# # data[:, 55] = torch.FloatTensor([1]*10)
# # print(_model(Variable(data)))
# print(result_mat)
# plt.imshow(result_mat)
# plt.colorbar()
# plt.show()