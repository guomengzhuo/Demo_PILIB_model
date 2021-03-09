import pandas as pd
from data import *
def bank_marketing(path, gamma, num_folds=5):
    df = pd.read_csv(path + "bank-additional-full-preprocess.csv",delimiter=',')
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
    # print(binary_predictor)
    # print(cate_predictor)
    # print(num_predictor)
    # print(orig_predictior_name)
    # 对于二元变量 没有操作 训练的bias是为0时的marginal value weight为1时的marginal value
    if binary_predictor is not None:
        df_linear_input_bin =  pd.DataFrame(df[binary_predictor].values, columns=binary_predictor)
        vectorized_bin_col_name = binary_predictor
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
    predictor_headers = df_input.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    print(predictor_headers)
    print(len(vectorized_bin_col_name), len(vectorized_cate_col_name), len(vectorized_num_col_name))
    all_folds = cv_split_data(
            df_input,
            predictor_headers,
            outcome_header,
            task="regression",
            dummy_cols=cate_predictor,
            num_folds=num_folds,
        )
    all_folds = scale_data(all_folds)
    if num_predictor is not None and cate_predictor is not None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), len(vectorized_cate_col_name), len(vectorized_num_col_name), vectorized_cate_col_name_num_list
    if num_predictor is not None and cate_predictor is not None and binary_predictor is None:
        return all_folds, None, len(vectorized_cate_col_name), len(vectorized_num_col_name), vectorized_cate_col_name_num_list
    if num_predictor is not None and cate_predictor is None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), None, len(vectorized_num_col_name), None
    if num_predictor is not None and cate_predictor is None and binary_predictor is None:
        return all_folds, None, None, len(vectorized_num_col_name), None
    if num_predictor is None and cate_predictor is not None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), len(vectorized_cate_col_name), None, vectorized_cate_col_name_num_list
    if num_predictor is None and cate_predictor is None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), None, None, None
    if num_predictor is None and cate_predictor is not None and binary_predictor is None:
        return all_folds, None, len(vectorized_cate_col_name), None, vectorized_cate_col_name_num_list
path = "D:/data/bank-marketing/JOC_dataset_ref/bank-additional/bank-additional/"
bank_marketing(path, 5, 5)