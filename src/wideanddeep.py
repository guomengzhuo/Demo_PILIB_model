from data import *
from utils import *
from models import *
import data_process
import tensorboardX
import os
import datetime
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Generate simulation data
np.random.seed(70) # 69 70 1000
BATCH_SIZE = 128
number_of_attr = 50
number_of_object = 20000
gamma = 5 # predefined number of pieces
degree = 10
task = 'regression'
verbose = False

class GMZ(torch.nn.Module):
    def __init__(self, input1,input2, **kwargs):  # input1: gamma * num of attributes, input2 : num of criteria
        super(GMZ, self).__init__()
        self.layer1 = torch.nn.Conv1d(input1,input1,1,1,0,bias=0,groups=input1) # group = number of criteria
        self.layer2 = torch.nn.Conv1d(input2,input2,1,1,0,bias=0,groups=input2)
        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Linear(input2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1)
        )
        self.layer_weight = torch.nn.Sequential(
            torch.nn.Linear(input1, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1),
            torch.nn.Sigmoid()
        )
        # self.layer_sig = torch.nn.Sigmoid() # For binary classification

    def forward(self, input_linear, input_nonlinear, mean, K, train_size, num_criteria, test_weight=None):
        y1 = self.layer1(input_linear) # First layer in linear part: correspond to each marginal score
        # print(y1.shape)
        index = np.linspace(0, K * num_criteria, K * num_criteria, endpoint=False, dtype=int)
        temp_y1 = torch.zeros(size=(train_size, num_criteria, 1))
        for i in range(0, K):
            temp_index = index[i::K]
            temp_y1 += y1[:, temp_index, :]
        y1 = temp_y1 # Sum all pieces for each attribute
        #############
        #############
        # For each attribute, we give an individual weights
        y2 = self.layer2(y1)
        # Parameters in layer2 are weights for each attribute?
        linear_out = torch.sum(y2, dim=1)
        # Sum all marginal scores, the output of linear part
        #############
        # For nonlinear part, the input is the original attribute values
        #############
        nonlinear_out = self.layer3_2(input_nonlinear)
        #############
        # The tradeoff coefficient, range 0-1
        if test_weight is None:
            # weight = self.layer_weight(mean)
            weight = torch.FloatTensor(np.array(0.5))
        else:
            weight = test_weight
        # weight = torch.FloatTensor(np.array(1.0))

        link_out = linear_out * weight + nonlinear_out * (1 - weight)
        # link_out = linear_out  + nonlinear_out
        # By Sigmoid, link_out is bineary classification
        # By sofmax, link_out is multiple classification
        # By identity, link_out is regression
        # link_out = self.layer_sig(link_out)
        return link_out, weight

def report_scores(
    f,
    task,
    val_perf,
    test_perf,
    final=False,
    i=0,
    loss=0,
    base_lr=0,
    crit_idx=0,
    verbose=False,
):

    if val_perf is None:
        raise ValueError("val_perf should not be None")

    if task == "regression":
        print(loss)
        if final is False:
            f.write(
                f"\t{i:3d}\t{loss:.2e}\t{val_perf:.2e}\t{test_perf:.2e}\t{base_lr:.2e}\n"
            )
            if verbose:
                print(
                    f"\t{i:3d}\t{loss:.2e}\t{val_perf:.2e}\t{test_perf:.2e}\t{base_lr:.2e}"
                )
        else:
            f.write("\n\tsaved model\n")
            f.write(f"\t{val_perf:.2e}\t{test_perf:.2e}\n")
            if verbose:
                print("\n\tsaved model\n")
                print(f"\t{crit_idx:3d}\t{val_perf:.2e}\t{test_perf:.2e}\n")

    elif task == "classification":
        print(loss)
        if final is False:
            f.write(
                f"\t{i:3d}\t{loss:.2e}\t{val_perf:.4}\t{test_perf:.4}\t{base_lr:.2e}\n"
            )
            if verbose:
                print(
                    f"\t{i:3d}\t{loss:.2e}\t{val_perf:.4}\t{test_perf:.4}\t{base_lr:.2e}"
                )
        else:
            f.write("\n\tsaved model\n")
            f.write(f"\t{val_perf:.4}\t{test_perf:.4}\n")
            if verbose:
                print("\n\tsaved model\n")
                print(f"\t{crit_idx:3d}\t{val_perf:.4}\t{test_perf:.4}\n")

    f.flush()

Ori_Data = data_process.generate_attribute_value(number_of_object, number_of_attr)
# For regression
New_Data, global_value, coeff = data_process.marginalvalue_with_interaction(Ori_Data, degree=degree, pair_num=900)
interval_vector, all_folds = data_process.piecewise_linear('./simulation_data.csv', './global_value.csv',
                                                                         gamma, task)

base_lrs = [5e-3] * 4 + [3e-3] * 4 + [1e-3] * 4 + [3e-4] * 4

patience = 100

if task == "classification":
    inc = 2
    interval = 1
else:
    inc = 20
    interval = 10

seeds = list(range(len(all_folds)))

max_order_path = (
            "./"+ task + "/number_attr_" + str(number_of_attr) + "/gamma_" + str(gamma)
        )   # .//bike_sharing/max_order_2
save_path = max_order_path + "/saved_models"
tb_path = max_order_path + "/runs"
if not os.path.exists(max_order_path):
    os.makedirs(max_order_path)
if not os.path.exists(tb_path):
    os.makedirs(tb_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

f = open(save_path + "/log.txt", "w")

for trial, seed in enumerate(seeds):
    time_stamp = "{:%Y-%m-%d_%H_%M_%S}".format(datetime.datetime.now())
    tb_dt_path = tb_path + "/" + time_stamp
    if not os.path.exists(tb_dt_path):
        os.makedirs(tb_dt_path)
    d = all_folds[trial]  # d 读取 all_folds列表中每个元素 每个元素是一个字典
    tr_x, va_x, te_x, tr_y, va_y, te_y = (
        d["tr_x"].values,
        d["va_x"].values,
        d["te_x"].values,
        d["tr_y"].values,
        d["va_y"].values,
        d["te_y"].values,
    )
    # 合并va和tr
    tr_x = np.concatenate((tr_x, va_x), axis=0)
    tr_y = np.concatenate((tr_y, va_y), axis=0)

    theta = None
    if task == "classification":
        # tr_y, va_y, te_y = tr_y[:, 1], va_y[:, 1], te_y[:, 1]
        tr_y, va_y, te_y = tr_y[:, 0], va_y[:, 0], te_y[:, 0]
        X_train = tr_x[:, gamma*number_of_attr]
        Y_train = tr_y
        clf = LogisticRegression(C=0.001, fit_intercept=True, max_iter=100).fit(X_train, Y_train)
        theta = clf.coef_

    # pretrain
    if task == 'regression':
        # print('tr_x is',tr_x)
        X_train =tr_x[:, 0:gamma*number_of_attr]
        # print('tr_y is', tr_y)
        Y_train = tr_y
        theta = utils.RLS(X_train, Y_train, 0.01)
        pred_y = np.dot(te_x[:, 0:gamma*number_of_attr], theta)
        linear_reg_MSE = np.square(te_y - np.ravel(pred_y)).mean()
        print('Piecewise Linear Regression:' + str(linear_reg_MSE))
        f.write('Piecewise Linear Regression:' + str(linear_reg_MSE))

        # theta = utils.LS(X_train, Y_train)
        # print(theta)
        X = np.array(tr_x[:, gamma*number_of_attr:])
        X = X.reshape(X.shape[0], -1)
        decision_tree = DecisionTreeRegressor(max_depth=20)
        random_forest = RandomForestRegressor(n_estimators=100)
        svr_rbf = SVR(kernel='rbf', gamma=0.1)
        svr_linear = SVR(kernel='linear')
        svr_poly = SVR(kernel='poly')

        dt_model = decision_tree.fit(X, np.array(Y_train).reshape(-1,1),)
        decision_tree_y = dt_model.predict(np.array(te_x[:, gamma*number_of_attr:]).reshape(-1, X.shape[1]))
        decision_tree_MSE = np.square(decision_tree_y - np.ravel(te_y)).mean()
        print('Decision Tree:'+str(decision_tree_MSE))
        f.write('Decision Tree:'+str(decision_tree_MSE))

        rf_model = random_forest.fit(X, np.array(Y_train).reshape(-1,1))
        random_forest_y = rf_model.predict(np.array(te_x[:, gamma*number_of_attr:]).reshape(-1, X.shape[1]))
        random_forest_MSE = np.square(random_forest_y - np.ravel(te_y)).mean()
        print('Random forest:'+str(random_forest_MSE))
        f.write('Random forest:'+str(random_forest_MSE))

        svr_linear_model = svr_linear.fit(X, np.array(Y_train).reshape(-1,1))
        svr_linear_y = svr_linear_model.predict(np.array(te_x[:, gamma*number_of_attr:]).reshape(-1, X.shape[1]))
        svr_linear_MSE = np.square(svr_linear_y - np.ravel(te_y)).mean()
        print('SVR_linear:'+str(svr_linear_MSE))
        f.write('SVR_linear:'+ str(svr_linear_MSE))

        svr_rbf_model = svr_rbf.fit(X, np.array(Y_train).reshape(-1,1))
        svr_rbf_y = svr_rbf_model.predict(np.array(te_x[:, gamma*number_of_attr:]).reshape(-1, X.shape[1]))
        svr_rbf_MSE = np.square(svr_rbf_y - np.ravel(te_y)).mean()
        print('SVR_rbf:' + str(svr_rbf_MSE))
        f.write('SVR_rbf:' + str(svr_rbf_MSE))

        svr_poly_model = svr_poly.fit(X,np.array(Y_train).reshape(-1,1))
        svr_poly_y = svr_poly_model.predict(np.array(te_x[:, gamma*number_of_attr:]).reshape(-1, X.shape[1]))
        svr_poly_MSE = np.square(svr_poly_y - np.ravel(te_y)).mean()
        print('SVR_poly:'+str(svr_poly_MSE))
        f.write('SVR_poly:'+str(svr_poly_MSE))


    tr_loader = get_data_loader(tr_x, tr_y, BATCH_SIZE)  # torch DataLoader 格式 for x,y in enumerate(tr_loader)
    # va_loader = get_data_loader(va_x, va_y, BATCH_SIZE)
    te_loader = get_data_loader(te_x, te_y, BATCH_SIZE)


    np.random.seed(seed)
    torch.manual_seed(seed)

    model = WideandDeep(input1= gamma*number_of_attr, input2=number_of_attr, n_inp=number_of_attr, reg_lambda=1e-5,
            reg_method="l2",)

    if theta is None:
        print("There are no intializations for linear part.")
    else:
        theta = np.squeeze(theta[:gamma*number_of_attr])
        # print(theta.shape)
        theta = torch.from_numpy(theta)
        theta = theta.type(torch.FloatTensor)
        theta = theta.view(-1, 1, 1)

    for name, params in model.model_pwlnet.named_parameters():
        # print(name)
        # print(params)
        if name == 'layer_piecewise.weight':
            # print('OOOOOO')
            # params.requires_grad = False
            params.data = theta
            # print(params)
            # name.weight
    fit_interactions = True  # Wide and block
    # fit_interactions = False
    base_lr = 5e-3
    Epoch = 150
    early_stop = False
    cnt = 0


    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    es_scores = []
    es_models = []
    step = 0
    es_break = False
    enable_es = True
    for epoch in range(Epoch):

        loss = wide_and_deep_train(model=model, data_loader=tr_loader, optimizer=optimizer, i=epoch, gamma=gamma, task= task , num_cat_variable=None, num_num_variable=gamma*number_of_attr, vectorized_cate_col_name_num_list=None,)
        print(loss)
        if epoch % interval == 0:
            val_perf, test_perf = wide_and_deep_test(
                model=model,
                data_loader=te_loader,
                va_loader=None,
                gamma=gamma,
                num_cat_variable=None,
                num_num_variable=gamma*number_of_attr,
                vectorized_cate_col_name_num_list=None,
                task=task
            )
            report_scores(
                f,
                task,
                val_perf,
                test_perf,
                False,
                epoch,
                loss,
                base_lr,
                verbose=verbose,
            )


    val_perf, test_perf = wide_and_deep_test(
                    model=model,
                    data_loader=te_loader,
                    gamma=gamma,
                    num_cat_variable=None,
                    num_num_variable=gamma*number_of_attr,
                    va_loader=None,
                    vectorized_cate_col_name_num_list=None,
                    task=task,
                )

    report_scores(
        f, task, val_perf, test_perf, final=True, crit_idx=0, verbose=verbose
    )

    torch.save(model.state_dict(), save_path + "/" + str(trial) + ".pt")
f.close()