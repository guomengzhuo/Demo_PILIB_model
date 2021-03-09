import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import copy


def cv_split_data(
    df2,
    predictor_headers,
    outcome_header,
    task="classification",
    dummy_cols=[],
    synth=False,
    num_folds=5,
):

    only_fold_index = -1

    df2 = df2.sample(frac=1, random_state=0).reset_index(drop=True) # 随机抽取100%的样本 reset_index(True)为了保证第一列不作为索隐列！
    if task == "regression":
        df_X = df2[predictor_headers]
        df_Y = df2[[outcome_header]]
    else:
        # df_Y1 = df2.iloc[:, -2]
        # df_Y2 = df2.iloc[:, -1]
        df_Y1 = df2[outcome_header]
        df_Y2 = df2[outcome_header]
        df_X = df2[predictor_headers]
        df_Y = pd.concat([df_Y1, df_Y2], axis=1)

    all_folds = []
    if task == "regression":
        df_Y_slice = df_Y.iloc[:, 0] # Series 切片 loc索隐label（列名），iloc索隐位置 第0列的所有行
    else:
        df_Y_slice = df_Y.iloc[:, 1]

    if task == "regression":
        kf = KFold(n_splits=num_folds, random_state=0, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)
    fold_idx = -1
    for d1, test in kf.split(df_X, df_Y_slice):
        fold_idx += 1
        if only_fold_index == 0:
            if fold_idx != only_fold_index:
                continue
            else:
                pass
                # print ('\nOnly Using Fold ' + str(fold_idx+1) + '\n')

        d1_X = df_X.iloc[d1] #按照split的索隐d1是训练集的位置索引， test是test set的位置索引
        d1_Y = df_Y.iloc[d1] # df_Y存的是预测值
        te_x = df_X.iloc[test]
        te_y = df_Y.iloc[test]

        if task == "regression":
            d1_Y_slice = d1_Y.iloc[:, 0]
        else:
            d1_Y_slice = d1_Y.iloc[:, 1]

        # skf2 = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

        if synth:
            test_size = 0.5
        elif num_folds == 10:
            test_size = 0.111111111111
        elif num_folds == 5:
            test_size = 0.25
        else:
            raise ValueError("invalid number of folds")

        if task == "classification":

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=test_size, random_state=0
            )
            train, valid = [x for x in sss.split(d1_X, d1_Y_slice)][0]
        elif task == "regression":
            train, valid = train_test_split(
                list(range(len(d1_Y_slice))), test_size=test_size, random_state=0
            ) # 又把training set 分为了validation和training

        # for preserving random seed effects
        if only_fold_index > 0:
            if fold_idx != only_fold_index:
                continue
            else:
                print("\nOnly Using Fold " + str(fold_idx + 1) + "\n")

        tr_x = d1_X.iloc[train]
        tr_y = d1_Y.iloc[train]
        va_x = d1_X.iloc[valid]
        va_y = d1_Y.iloc[valid]

        assert list(tr_x.index.values) == list(tr_y.index.values)
        assert list(va_x.index.values) == list(va_y.index.values)
        assert list(te_x.index.values) == list(te_y.index.values)

        assert df_X.shape[0] == len(tr_x.index.values) + len(va_x.index.values) + len(
            te_x.index.values
        )

        all_folds.append(
            {
                "tr_x": tr_x,
                "tr_y": tr_y,
                "va_x": va_x,
                "va_y": va_y,
                "te_x": te_x,
                "te_y": te_y,
                "tr_indices": tr_y.index.values,
                "va_indices": va_y.index.values,
                "te_indices": te_y.index.values,
                "dummy_cols": dummy_cols,
            }
        )

        if synth:
            break
    return all_folds


def scale_data(all_folds, impute_mean=False, features_to_scale=[]):
    # scale only numeric predictors with standard scaler (removing mean and dividing by variance)
    for fold in all_folds:
        fold["tr_x_orig"] = fold["tr_x"]
        fold["te_x_orig"] = fold["te_x"]
        tr_x = fold["tr_x"].copy()
        tr_y = fold["tr_y"].copy()
        fold["tr_y_orig"] = fold["tr_y"]
        fold["te_y_orig"] = fold["te_y"]
        te_x = fold["te_x"].copy()
        te_y = fold["te_y"].copy()
        if "va_x" in fold:
            valid = True
        else:
            valid = False
        if valid:
            fold["va_x_orig"] = fold["va_x"]
            fold["va_y_orig"] = fold["va_y"]
            va_x = fold["va_x"].copy()
            va_y = fold["va_y"].copy()

        if impute_mean:
            imp = preprocessing.Imputer(
                missing_values="NaN", strategy="mean", axis=0
            ).fit(tr_x)
            tr_x_temp = imp.transform(tr_x)
            tr_x = pd.DataFrame(tr_x_temp, index=tr_x.index, columns=tr_x.columns)
            if valid:
                va_x_temp = imp.transform(va_x)
                va_x = pd.DataFrame(va_x_temp, index=va_x.index, columns=va_x.columns)
            te_x_temp = imp.transform(te_x)
            te_x = pd.DataFrame(te_x_temp, index=te_x.index, columns=te_x.columns)
            fold["imputer"] = imp

        if features_to_scale:
            scaler_x = preprocessing.StandardScaler().fit(
                tr_x[features_to_scale].values
            )
            tr_x_scaled = scaler.transform(tr_x[features_to_scale].copy().values)
            if valid:
                va_x_scaled = scaler.transform(va_x[features_to_scale].copy().values)
            te_x_scaled = scaler.transform(te_x[features_to_scale].copy().values)

            tr_x[features_to_scale] = tr_x_scaled
            if valid:
                va_x[features_to_scale] = va_x_scaled
            te_x[features_to_scale] = te_x_scaled
        else:
            scaler_x = preprocessing.StandardScaler().fit(tr_x)
            tr_x_temp = scaler_x.transform(tr_x)
            tr_x = pd.DataFrame(tr_x_temp, index=tr_x.index, columns=tr_x.columns)
            if valid:
                va_x_temp = scaler_x.transform(va_x)
                va_x = pd.DataFrame(va_x_temp, index=va_x.index, columns=va_x.columns)
            te_x_temp = scaler_x.transform(te_x)
            te_x = pd.DataFrame(te_x_temp, index=te_x.index, columns=te_x.columns)
        if tr_y.shape[1] == 1:
            scaler_y = preprocessing.StandardScaler().fit(tr_y)
            tr_y_temp = scaler_y.transform(tr_y)
            tr_y = pd.DataFrame(tr_y_temp, index=tr_y.index, columns=tr_y.columns)
            if valid:
                va_y_temp = scaler_y.transform(va_y)
                va_y = pd.DataFrame(va_y_temp, index=va_y.index, columns=va_y.columns)
            te_y_temp = scaler_y.transform(te_y)
            te_y = pd.DataFrame(te_y_temp, index=te_y.index, columns=te_y.columns)
            fold["scaler_y"] = scaler_y

        fold["tr_x"] = tr_x
        if valid:
            fold["va_x"] = va_x
        fold["te_x"] = te_x
        fold["tr_y"] = tr_y
        if valid:
            fold["va_y"] = va_y
        fold["te_y"] = te_y
        fold["scaler_x"] = scaler_x
    return all_folds


def oversample_data(
    all_folds, predictor_headers, outcome_header, task="classification"
):
    for fold in all_folds:
        tr_x = fold["tr_x"]
        tr_y = fold["tr_y"]

        # recombine tr_x and tr_y momentarily
        tr_both = pd.concat([tr_x, tr_y], axis=1)
        temp1 = tr_both.loc[tr_y.iloc[:, 0] == 1]
        temp2 = tr_both.loc[tr_y.iloc[:, 0] == 0]

        if len(temp1) > len(temp2):
            tr0 = temp1
            tr1 = temp2
        else:
            tr0 = temp2
            tr1 = temp1
        assert len(tr0) > len(tr1)
        tile1 = tr1.copy()

        while (len(tr1) + len(tile1)) <= len(tr0):
            tr1 = pd.concat([tr1, tile1])
        size_diff = len(tr0) - len(tr1)
        if size_diff != 0:
            # continue oversampling by randomly selecting without replacement
            np.random.seed(0)
            remaining_indices = np.random.choice(len(tile1), size_diff, replace=False)
            tile1_subset = tile1.iloc[remaining_indices, :]
            tr1 = pd.concat([tr1, tile1_subset])
            # rejoin balanced classes and shuffle rows
        tr = pd.concat([tr0, tr1])
        tr = tr.sample(frac=1, random_state=0).reset_index(drop=True)
        # split tr_x and tr_y again
        tr_x = tr[predictor_headers]
        if task == "regression":
            tr_y = tr.iloc[:, -1:]
        else:
            tr_y = tr.iloc[:, -2:]
            # return the splits to the all_folds dict
        fold["tr_x"] = tr_x
        fold["tr_y"] = tr_y
    return all_folds


def get_mapping(df, col):
    mapping = dict() # 空字典
    total = len(df[col]) # 所有行数
    val_cnts = df[col].value_counts() # 统计每个数值出现的次数
    keys = list(val_cnts.keys()) # keys存储每个独特值
    vals = list(val_cnts) # 存储每个独特值的出现次数
    sum_percent = 0
    first = True
    mapped_val = 0
    for k, key in enumerate(keys): # k: list index; key: list value at index k !!!
        percent = vals[k] * 1.0 / total
        sum_percent += percent
        if sum_percent <= 0.95:
            mapping[key] = str(keys[mapped_val])
            mapped_val += 1
        elif first == True:
            mapping[key] = str(keys[mapped_val])
            mapped_val += 1
            first = False
        else:
            if len(keys[k:]) == 1 and mapping[keys[k - 1]] != "other":
                mapping[key] = str(keys[mapped_val])
                mapped_val += 1
            else:
                mapping[key] = "other"
    return mapping


def binarize(df, col):
    col_vals = sorted(list(set(df[col]))) # 从小到大排列列表中唯一值
    if len(col_vals) == 2:
        return df[col].map({col_vals[0]: 0, col_vals[1]: 1}), True # 如果只有两个unique value 使用0,1变量
    else:
        return df[col], False # 多于两个unique value 使用原始值


def dummify(df, dummy_cols, outcome_header_to_dummify=None):
    dummy_cols2 = copy.copy(dummy_cols)
    for col_name in dummy_cols:
        df[col_name] = df[col_name].replace(get_mapping(df, col_name)) # replace(dict) 按照dict中 把键值替换为value值
        df[col_name], binary = binarize(df, col_name)
        if binary:
            dummy_cols2.remove(col_name)
    if outcome_header_to_dummify is not None:
        df = pd.get_dummies(df, columns=dummy_cols2 + [outcome_header_to_dummify])
    else:
        df = pd.get_dummies(df, columns=dummy_cols2)
    return df, dummy_cols2


def get_cal_housing(path, num_folds=5):
    orig_predictor_headers = [
        "longitude",
        "latitude",
        "housingMedianAge",
        "totalRooms",
        "totalBedrooms",
        "population",
        "households",
        "medianIncome",
    ]
    outcome_header = "medianHouseValue"
    used_headers = orig_predictor_headers + [outcome_header]
    df = pd.read_csv(
        path + "cal_housing/cal_housing.data.gz", names=used_headers, compression="gzip"
    )

    df2 = df[used_headers]
    predictor_headers = df2.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    all_folds = cv_split_data(
        df2, predictor_headers, outcome_header, task="regression", num_folds=num_folds
    )
    all_folds = scale_data(all_folds)
    return all_folds

def spambase(path, gamma, num_folds):
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
    df = pd.read_csv(path + 'spambase/spambase.data' , delimiter=',', header=None, names=col_name_list)
    outcome_header = "label"
    # print(df.describe())
    orig_predictior_name = list(df)
    df2 = df[orig_predictior_name]
    orig_predictior_name.remove(outcome_header)
    num_predictor = list(df)
    num_predictor.remove(outcome_header)  # 所有连续变量列名
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
    df_input = pd.concat([df_linear_input_num, df2], axis=1)
    predictor_headers = list(df_input)  # .columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    all_folds = cv_split_data(
        df_input,
        predictor_headers,
        outcome_header,
        task="classification",
        dummy_cols=None,
        num_folds=num_folds,
    )
    return all_folds, 0, None, len(vectorized_num_col_name), None

def bank_marketing(path, gamma, num_folds=5):
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
    all_folds = cv_split_data(
            df_input,
            predictor_headers,
            outcome_header,
            task="classification",
            dummy_cols=[cate_predictor],
            num_folds=num_folds,
        )

    # all_folds = scale_data(all_folds)
    if num_predictor is not None and cate_predictor is not None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), len(vectorized_cate_col_name), len(vectorized_num_col_name), vectorized_cate_col_name_num_list
    if num_predictor is not None and cate_predictor is not None and binary_predictor is None:
        return all_folds, 0, len(vectorized_cate_col_name), len(vectorized_num_col_name), vectorized_cate_col_name_num_list
    if num_predictor is not None and cate_predictor is None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), None, len(vectorized_num_col_name), None
    if num_predictor is not None and cate_predictor is None and binary_predictor is None:
        return all_folds, 0, None, len(vectorized_num_col_name), None
    if num_predictor is None and cate_predictor is not None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), len(vectorized_cate_col_name), None, vectorized_cate_col_name_num_list
    if num_predictor is None and cate_predictor is None and binary_predictor is not None:
        return all_folds, len(vectorized_bin_col_name), None, None, None
    if num_predictor is None and cate_predictor is not None and binary_predictor is None:
        return all_folds, 0, len(vectorized_cate_col_name), None, vectorized_cate_col_name_num_list


def skill(path, gamma, num_folds = 5):
    col_name_list = ['response', 'Age', 'HoursPerWeek', 'TotalHours', 'APM', 'SelectByHotkeys', 'AssignToHotkeys',
                     'UniqueHotkeys', 'MinimapAttacks', 'MinimapRightClicks', 'NumberOfPACs', 'GapBetweenPACs',
                     'ActionLatency', 'ActionsInPAC', 'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade',
                     'ComplexUnitsMade', 'ComplexAbilitiesUsed'
                     ]
    df = pd.read_csv(path+"skill/skill_bin.csv", delimiter=',', header=None, names=col_name_list)  # iloc[57] is label
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
    all_folds = cv_split_data(
        df_input,
        predictor_headers,
        outcome_header,
        task="classification",
        dummy_cols=None,
        num_folds=num_folds,
    )
    return all_folds, 0, None, len(vectorized_num_col_name), None

def get_bike_sharing(path, gamma, num_folds=5):
    # gamma: predefined number of pieces
    df = pd.read_csv(path + "bike_sharing/hour.csv.gz", compression="gzip")
    orig_predictor_headers = list(df) # 读取列名
    outcome_header = "cnt"
    used_headers = orig_predictor_headers
    used_headers.remove("instant")
    used_headers.remove("dteday")
    used_headers.remove("casual")
    used_headers.remove("registered")
    df2 = df[used_headers]

    original_predictor_headers = used_headers
    original_predictor_headers.remove(outcome_header)

    dummy_cols = ["weathersit"]
    df2, dummy_cols2 = dummify(df2, dummy_cols)
    predictor_headers = df2.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)

    # Vectorization!
    # gamma = 3
    # For dummy variables
    interval_vector_cat = []
    vectorized_cate_col_name = []  # new col names

    df3 = df[original_predictor_headers]
    # 处理categorical attribute
    if dummy_cols is not None:
        vectorized_cate_col_name_num_list = [] # 记录每个类别变量被几个点分割
        alternative_matrix = []
        for col_name in dummy_cols:
            col_val = sorted(list(set(df3[col_name])))  # USED ORIGINAL DATA
            print(col_val)
            vectorized_cate_col_name_num_list.append(len(col_val)-1)
            interval_vector_cat.append(col_val)
            for index in range(len(col_val) - 1):
                vectorized_cate_col_name.append(col_name + '_cat_vec_' + str(index))

        for i in range(df3.shape[0]):
            alternative_i = []
            for j, col_name in enumerate(dummy_cols):
                for k in range(len(interval_vector_cat[j]) - 1):  # k - thresholds for j-th criterion
                    if interval_vector_cat[j][k + 1] < df3[col_name][i]:
                        alternative_i.append(1.0)
                    elif interval_vector_cat[j][k] <= df3[col_name][i] <= interval_vector_cat[j][k + 1]:
                        if df3[col_name][i] == 0:
                            alternative_i.append(1.0)
                        else:
                            alternative_i.append(
                                (df3[col_name][i] - interval_vector_cat[j][k]) / (
                                            interval_vector_cat[j][k + 1] - interval_vector_cat[j][k]))
                    else:
                        alternative_i.append(0.0)
            alternative_matrix.append(np.array(alternative_i))
        for col_name in dummy_cols:
            original_predictor_headers.remove(col_name)  # 从df3中把categorical attribute移除

        # print(alternative_matrix)
        df_linear_input_cat = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_cate_col_name)

    # For numerical attribute
    interval_vector_num = []
    vectorized_num_col_name = []
    alternative_matrix = []

    for col_name in original_predictor_headers:
        if col_name not in dummy_cols:
            max = np.max(df3[col_name])
            min = np.min(df3[col_name])
            interval_vector_num.append(np.linspace(min, max, gamma + 1, endpoint=True))
            for index in range(gamma):
                vectorized_num_col_name.append(col_name + '_num_vec_' + str(index))
    interval_vector_num = np.array(interval_vector_num)
    # print(interval_vector.shape)
    # print(len(original_predictor_headers))

    # Vectorization

    for i in range(df3.shape[0]):  # i - index of alternatives
        alternative_i = []
        for j, col_name in enumerate(original_predictor_headers):  # j - index of criteria
            for k in range(len(interval_vector_num[j]) - 1):  # k - thresholds for j-th criterion
                if interval_vector_num[j][k + 1] < df3[col_name][i]:
                    alternative_i.append(1.0)
                elif interval_vector_num[j][k] <= df3[col_name][i] <= interval_vector_num[j][k + 1]:
                    if df3[col_name][i] == 0:
                        alternative_i.append(1.0)
                    else:
                        alternative_i.append(
                            (df3[col_name][i] - interval_vector_num[j][k]) / (
                                        interval_vector_num[j][k + 1] - interval_vector_num[j][k]))
                else:
                    alternative_i.append(0.0)
        alternative_matrix.append(np.array(alternative_i))

    df_linear_input_num = pd.DataFrame(np.array(alternative_matrix), columns=vectorized_num_col_name)
    # df2 是放入block模块的 df_linear_input 是放入线性模块的
    df_input = pd.concat([df_linear_input_cat, df_linear_input_num, df2], axis=1)
    predictor_headers = df_input.columns.values[:].tolist()
    predictor_headers.remove(outcome_header)
    # pretraining
    # X_train = df_input.iloc[:, 0:(len(vectorized_cate_col_name)+len(vectorized_num_col_name))].values
    # Y_train = df_input.loc[:,[outcome_header]].values
    # theta =utils.RLS(X_train, Y_train,0.01)


    all_folds = cv_split_data(
        df_input,
        predictor_headers,
        outcome_header,
        task="regression",
        dummy_cols=dummy_cols2,
        num_folds=num_folds,
    )
    all_folds = scale_data(all_folds)
    if dummy_cols is not None:
        return all_folds,0, len(vectorized_cate_col_name), len(vectorized_num_col_name), vectorized_cate_col_name_num_list # 最终的数据包括三部分,
        # 第0列-第len(vectorized_cate_col_name)列为类别变量
        # 第len(vectorized_cate_col_name)-第(len(vectorized_cate_col_name) + len(vectorized_num_col_name)) 为数字变量
        # 第(len(vectorized_cate_col_name) + len(vectorized_num_col_name)) - (-1） 是block network的输入
        #vectorized_cate_col_name_num_list 记录了类别变量的characterstics points的个数
    else:
        return all_folds, 0, len(vectorized_num_col_name), None


def split_df(df, predictor_headers):
    df_Y1 = df.iloc[:, -2]
    df_Y2 = df.iloc[:, -1]
    df_X = df[predictor_headers]
    df_Y = pd.concat([df_Y1, df_Y2], axis=1)
    return df_X, df_Y


def cv_split_validation(
    dfa,
    dfb,
    predictor_headers,
    outcome_header,
    num_folds,
    task="classification",
    dummy_cols=[],
):
    dfa = dfa.sample(frac=1, random_state=0).reset_index(drop=True)
    dfb = dfb.sample(frac=1, random_state=0).reset_index(drop=True)

    if task == "regression":
        df_X = dfa[predictor_headers]
        df_Y = dfa[[outcome_header]]
    else:
        df_Y1 = dfa.iloc[:, -2]
        df_Y2 = dfa.iloc[:, -1]
        df_X = dfa[predictor_headers]
        df_Y = pd.concat([df_Y1, df_Y2], axis=1)

    all_folds = []
    if task == "regression":
        df_Y_slice = df_Y.iloc[:, 0]
    else:
        df_Y_slice = df_Y.iloc[:, 1]

    if task == "regression":
        kf = KFold(n_splits=num_folds, random_state=0, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=num_folds, random_state=0, shuffle=True)

    te_x, te_y = split_df(dfb, predictor_headers)

    for tr, va in kf.split(df_X, df_Y_slice):
        tr_x = df_X.iloc[tr]
        tr_y = df_Y.iloc[tr]
        va_x = df_X.iloc[va]
        va_y = df_Y.iloc[va]

        fold = {
            "tr_x": tr_x,
            "tr_y": tr_y,
            "va_x": va_x,
            "va_y": va_y,
            "te_x": te_x,
            "te_y": te_y,
            "tr_indices": tr_y.index.values,
            "va_indices": va_y.index.values,
            "te_indices": te_y.index.values,
            "dummy_cols": dummy_cols,
        }
        all_folds.append(fold)
    return all_folds


def get_cifar(path, note="", num_folds=5):
    outcome_header = "label"
    try:
        dfa = pd.read_csv(path + "cifar/train34.csv", delimiter=",")
        dfb = pd.read_csv(path + "cifar/test34.csv", delimiter=",")
    except:
        path = "/home/mtsang/workspace/transparency_v2/dataset/"
        dfa = pd.read_csv(path + "cifar/train34.csv", delimiter=",")
        dfb = pd.read_csv(path + "cifar/test34.csv", delimiter=",")

    dfa[outcome_header] = dfa[outcome_header].map({3: 0, 4: 1})
    dfb[outcome_header] = dfb[outcome_header].map({3: 0, 4: 1})

    dfa = pd.get_dummies(dfa, columns=[outcome_header])
    dfb = pd.get_dummies(dfb, columns=[outcome_header])
    predictor_headers = dfa.columns.values[:].tolist()[:-2]

    if num_folds == 0:
        df_X, df_Y = split_mnist(dfa, predictor_headers)
        df_X2, df_Y2 = split_mnist(dfb, predictor_headers)

        all_folds = [{"tr_x": df_X, "tr_y": df_Y, "te_x": df_X2, "te_y": df_Y2}]
    else:
        all_folds = cv_split_validation(
            dfa, dfb, predictor_headers, outcome_header, num_folds
        )

    all_folds = scale_data(all_folds)
    return all_folds


def get_realworld_data(dataset_name, gamma, num_folds=5, note="49"):
    dataset_path = "../dataset/"

    if dataset_name == "cal_housing":
        return get_cal_housing(dataset_path, num_folds=num_folds)
    elif dataset_name == "bike_sharing":
        return get_bike_sharing(dataset_path, gamma=gamma , num_folds=num_folds)
    elif dataset_name == "cifar":
        return get_cifar(dataset_path)
    elif dataset_name == 'bank_marketing':
        return bank_marketing(dataset_path, gamma=gamma, num_folds=num_folds)
    elif dataset_name == 'spambase':
        return spambase(dataset_path, gamma=gamma, num_folds=num_folds)
    elif dataset_name == 'skill':
        return skill(dataset_path, gamma=gamma, num_folds=num_folds)
    else:
        return None
