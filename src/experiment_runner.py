import pickle
import sys
import os
import math
import datetime
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
sys.path.append("../../src")

from data import *
from utils import *
from models import *

import tensorboardX # pytorch Tools for Visualization!


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


def run_experiment(
    dataset,
    l0_lambda,
    max_interaction_order,
    dataset_params,
    root_path,
    gamma,
    num_folds=5,
    verbose=False,
    waiting_fraction=0.1,
    n_blocks=20,
    selected_fold=None,

):

    if type(l0_lambda) == list:
        max_order_path = (
            root_path
            + "/"
            + dataset
            + "/max_order_"
            + str(max_interaction_order)
            + "_noisy"
        )
        save_path = max_order_path + "/saved_models"
        tb_path = max_order_path + "/runs"
        if not os.path.exists(root_path + "/" + dataset):
            os.makedirs(root_path + "/" + dataset)
        if not os.path.exists(max_order_path):
            os.makedirs(max_order_path)
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)

        f = open(max_order_path + "/l0s.txt", "w")
        f.write(str(l0_lambda))
        f.flush()
        f.close()
    else:
        max_order_path = (
            root_path + "/" + dataset + "/max_order_" + str(max_interaction_order)
        )   # .//bike_sharing/max_order_2
        l0_dir = "l0_" + f"{l0_lambda:.1e}"# l0_dir = str(l0_1.0e-02)
        l0_path = max_order_path + "/" + l0_dir #.//bike_sharing/max_order_2/l0_1.0e-02
        save_path = l0_path + "/saved_models"
        tb_path = l0_path + "/runs"
        if not os.path.exists(root_path + "/" + dataset):
            os.makedirs(root_path + "/" + dataset)
        if not os.path.exists(max_order_path):
            os.makedirs(max_order_path)
        if not os.path.exists(l0_path):
            os.makedirs(l0_path)
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)

    n_inp = dataset_params[0]
    n_hidden_units = dataset_params[1] # list
    task = dataset_params[2]
    reg_const = dataset_params[3]
    l0_const = l0_lambda
    max_order = max_interaction_order

    base_lrs = [5e-3] * 4 + [3e-3] * 4 + [1e-3] * 4 + [3e-4] * 4

    patience = 100

    #####  fixed parameters

    batch_size = 128
    n_out = 1
    n_hidden_uni = 10

    if task == "classification":
        inc = 2
        interval = 1
    else:
        inc = 20
        interval = 10

    all_folds, num_bin_variable, num_cat_variable, num_num_variable, vectorized_cate_col_name_num_list = get_realworld_data(dataset, gamma=gamma , num_folds=num_folds)
    # print(num_bin_variable)
    # print(num_cat_variable)
    print(num_num_variable)
    # print(len(vectorized_cate_col_name_num_list))

    seeds = list(range(len(all_folds)))
    use_main_effect_nets = False

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if selected_fold is not None:
        f = open(save_path + "/log" + str(selected_fold) + ".txt", "w")
    else:
        f = open(save_path + "/log.txt", "w")

    for trial, seed in enumerate(seeds):
        if selected_fold is not None:
            if trial != selected_fold:
                continue

        time_stamp = "{:%Y-%m-%d_%H_%M_%S}".format(datetime.datetime.now())
        if selected_fold is not None:
            tb_dt_path = tb_path + "/" + str(selected_fold) + "_" + time_stamp

        else:
            tb_dt_path = tb_path + "/" + time_stamp

        if not os.path.exists(tb_dt_path):
            os.makedirs(tb_dt_path)

        d = all_folds[trial] # d 读取 all_folds列表中每个元素 每个元素是一个字典
        tr_x, va_x, te_x, tr_y, va_y, te_y = (
            d["tr_x"].values,
            d["va_x"].values,
            d["te_x"].values,
            d["tr_y"].values,
            d["va_y"].values,
            d["te_y"].values,
        )
        if np.isnan(tr_x).any():
            tr_x[np.isnan(tr_x)] = np.median(tr_x[~np.isnan(tr_x)])
        if np.isnan(va_x).any():
            va_x[np.isnan(va_x)] = np.median(va_x[~np.isnan(va_x)])
        if np.isnan(te_x).any():
            te_x[np.isnan(te_x)] = np.median(te_x[~np.isnan(te_x)])
        if np.isnan(tr_y).any():
            print('Na in tr_y')
        if np.isnan(va_y).any():
            print('Na in va_y')
        if np.isnan(te_y).any():
            print('Na in te_y')
        if num_cat_variable is None:
            num_cat_variable = 0
        if num_bin_variable is None:
            num_bin_variable = 0
        theta = None
        if task == "classification":
            # tr_y, va_y, te_y = tr_y[:, 1], va_y[:, 1], te_y[:, 1]
            tr_y, va_y, te_y = tr_y[:, 0], va_y[:, 0], te_y[:, 0]
            X_train = tr_x[:, 0:(num_bin_variable + num_cat_variable + num_num_variable)]
            Y_train = tr_y
            clf = LogisticRegression(C=0.001, fit_intercept=True,max_iter=100).fit(X_train, Y_train)
            theta = clf.coef_
        #pretrain
        if task == 'regression':
            # print('tr_x is',tr_x)
            X_train = tr_x[:, 0:(num_cat_variable + num_num_variable)]
            # print('tr_y is', tr_y)
            Y_train = tr_y
            theta =utils.RLS(X_train, Y_train,0.01)
            # theta = utils.LS(X_train, Y_train)
            # print(theta)

        tr_loader = get_data_loader(tr_x, tr_y, batch_size) # torch DataLoader 格式 for x,y in enumerate(tr_loader)
        va_loader = get_data_loader(va_x, va_y, batch_size)
        te_loader = get_data_loader(te_x, te_y, batch_size)

        logger = tensorboardX.SummaryWriter(log_dir=tb_dt_path) # Wide and block
        # logger = None

        if verbose:
            print("trial", trial, "\n")
        f.write("trial " + str(trial) + "\n")

        np.random.seed(seed)
        torch.manual_seed(seed)

        if type(l0_const) == list:
            l0_const2 = l0_const[trial]
        else:
            l0_const2 = l0_const

        # Model1: pwl network; Model2: Block network; Model3: Deep MLP
        # Model Wide and Block: Model1 + Model2
        # Model Wide and Block and Deep: Model1 + Model2 + Model3
        if dataset == 'bank_marketing':
            input1 = num_num_variable + num_cat_variable + num_bin_variable
            input2 = num_bin_variable + int(num_num_variable/gamma) + len(vectorized_cate_col_name_num_list)

        else:
            input1 = num_num_variable + num_cat_variable
            if vectorized_cate_col_name_num_list is None:
                vectorized_cate_col_name_num_list = []
            input2 = int(num_num_variable / gamma) + len(vectorized_cate_col_name_num_list)

            # Wide and block and deep
        # model = WideAndBlockAndDeep(
        #     input1,
        #     input2,
        #     n_inp,
        #     n_out,
        #     n_hidden_units,
        #     n_group=n_blocks,
        #     uni_hidden_unit=n_hidden_uni * use_main_effect_nets,
        #     n_layers_uni=3,
        #     reg_lambda=reg_const,
        #     reg_method="l2",
        #     max_interaction_order=max_order,
        #     l0_lambda=l0_const2,
        #     # use_weight_norm=False,
        #     use_fixed_mask=False,
        #     vectorized_cate_col_name_num_list = vectorized_cate_col_name_num_list
        # )

        # Wide and Block

        model = WideAndBlock(
            input1,
            input2,
            n_inp,
            n_out,
            n_hidden_units,
            n_group=n_blocks,
            uni_hidden_unit=n_hidden_uni * use_main_effect_nets,
            n_layers_uni=3,
            reg_lambda=reg_const,
            reg_method="l2",
            max_interaction_order=max_order,
            l0_lambda=l0_const2,
            # use_weight_norm=False,
            use_fixed_mask=False,
            vectorized_cate_col_name_num_list = vectorized_cate_col_name_num_list
        )

        # Wide and Deep

        # model = WideandDeep(input1,
        #     input2,
        #     n_inp,
        #     n_out,
        #     n_hidden_units,
        #     n_group=n_blocks,
        #     uni_hidden_unit=n_hidden_uni * use_main_effect_nets,
        #     n_layers_uni=3,
        #     reg_lambda=reg_const,
        #     reg_method="l2",
        #     max_interaction_order=max_order,
        #     l0_lambda=l0_const2,
        #     # use_weight_norm=False,
        #     use_fixed_mask=False,
        #     vectorized_cate_col_name_num_list = vectorized_cate_col_name_num_list)

        if theta is None:
            print("There are no intializations for linear part.")
        else:
            theta = np.squeeze(theta[:(num_num_variable + num_cat_variable)])
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
        # nit model
        # model = BlockMainEffectsMLP(
        #     n_inp,
        #     n_out,
        #     n_hidden_units,
        #     n_group=n_blocks,
        #     uni_hidden_unit=n_hidden_uni * use_main_effect_nets,
        #     n_layers_uni=3,
        #     reg_lambda=reg_const,
        #     reg_method="l2",
        #     l0_lambda=l0_const2,
        #     max_interaction_order=max_order,
        # )

        ## make two loops

        fit_interactions = True  # Wide and block
        # fit_interactions = False
        base_lr = 5e-3
        cnt = 0
        waiting_period = False
        waiting_step = 0
        fix_mask = False
        reinitialized = False

        while fit_interactions:
            # print(model.parameters())
            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

            for i in range(cnt, cnt + inc + 1):

                if waiting_period and not fix_mask:
                    if waiting_step >= steps_to_target_order * waiting_fraction :  # Force to stop when 1000 times   or i == 1000
                        if verbose:
                            print("freezing")
                        # if i == 1000: Mark
                        # if i == 1000:
                        #     f.write("Force to stop\n")
                        f.write("freezing\n")
                        fix_mask = True
                    else:
                        waiting_step += 1

                if (
                    hasattr(model.model_block.layers[0], "start_waiting_period")
                ) and not waiting_period:
                    if verbose:
                        print("starting waiting period")

                    steps_to_target_order = i
                    waiting_period = True

                loss = train(
                    model, tr_loader, optimizer, i, logger=logger, fix_mask=fix_mask,
                    pos_linear_input=num_bin_variable+num_num_variable+num_cat_variable,
                    gamma=gamma, num_bin_variable=num_bin_variable,
                    num_cat_variable=num_cat_variable,
                    num_num_variable=num_num_variable, vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list
                )

                if hasattr(model.model_block.layers[0], "fixed_mask") and not reinitialized:
                    if verbose:
                        print("reinitializing")
                    reinitialize_parameters(model)
                    reinitialized = True
                    enable_es = True
                    f.write("reinitialized\n")
                    fit_interactions = False
                    break

                if i % interval == 0:
                    val_perf, test_perf = test(
                        model,
                        te_loader,
                        epoch=i,
                        va_loader=va_loader,
                        logger=logger,
                        task=task,
                        pos_linear_input=num_bin_variable+num_num_variable + num_cat_variable, gamma=gamma,
                        num_bin_variable = num_bin_variable,
                        num_cat_variable=num_cat_variable,
                        num_num_variable=num_num_variable,
                        vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list

                    )
                    report_scores(
                        f,
                        task,
                        val_perf,
                        test_perf,
                        False,
                        i,
                        loss,
                        base_lr,
                        verbose=verbose,
                    )

            if fit_interactions:
                cnt = cnt + inc + 1
            else:
                cnt = i

        es_scores = []
        es_models = []
        step = 0

        assert reinitialized
        assert hasattr(model.model_block.layers[0], "fixed_mask")
        assert fix_mask

        es_break = False

        for lr_idx, base_lr in enumerate(base_lrs):

            interaction_sizes = np.count_nonzero(
                model.model_block.layers[0].mask_raw.data.cpu().numpy(), axis=1
            )
            if verbose:
                print("max interaction size", np.max(interaction_sizes))

            optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

            for i in range(cnt, cnt + inc + 1):

                # loss = train(
                #     model, tr_loader, optimizer, i, logger=logger, fix_mask=fix_mask
                # )  # TODO

                loss = train(
                    model, tr_loader, optimizer, i, logger=logger, fix_mask=fix_mask,
                    pos_linear_input=num_bin_variable+num_num_variable + num_cat_variable, gamma=gamma,
                    num_bin_variable=num_bin_variable,
                    num_cat_variable=num_cat_variable,
                    num_num_variable=num_num_variable,
                    vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list
                )# TODO




                if i % interval == 0:
                    val_perf, test_perf = test(
                        model,
                        te_loader,
                        epoch=i,
                        va_loader=va_loader,
                        logger=logger,
                        task=task,
                        pos_linear_input=num_num_variable + num_cat_variable+num_bin_variable, gamma=gamma,
                        num_bin_variable = num_bin_variable,
                        num_cat_variable=num_cat_variable,
                        num_num_variable=num_num_variable,
                        vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list
                    )

                    report_scores(
                        f,
                        task,
                        val_perf,
                        test_perf,
                        False,
                        i,
                        loss,
                        base_lr,
                        verbose=verbose,
                    )

                    if enable_es:
                        es_scores.append(val_perf)
                        es_models.append(copy.deepcopy(model.state_dict()))
                        crit_idx, es_break = early_stopping_checker(
                            es_scores, es_models, step, task=task, patience=patience
                        )
                        if es_break:
                            break

                    step += 1

            cnt = cnt + inc + 1
            if es_break:
                break

        model.load_state_dict(es_models[crit_idx])
        val_perf, test_perf = test(
            model, te_loader, epoch=i, va_loader=va_loader, logger=logger, task=task,
            pos_linear_input=num_bin_variable+num_num_variable + num_cat_variable, gamma=gamma,
            num_bin_variable = num_bin_variable,
            num_cat_variable=num_cat_variable,
            num_num_variable=num_num_variable,
            vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list

        )

        report_scores(
            f, task, val_perf, test_perf, final=True, crit_idx=0, verbose=verbose
        )

        torch.save(model.state_dict(), save_path + "/" + str(trial) + ".pt")

        if verbose:
            print()
    f.close()
