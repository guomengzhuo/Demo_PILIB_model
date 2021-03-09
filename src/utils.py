import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import pickle

global_epoch = 0
mt_reg = False
inter_order_lambda = 5e-2
reg_lambda = 5e-5
freeze_mask = False


def str_ndarray_1d(arr, fmt="%.2f", sep=","):
    assert len(arr.shape) <= 1

    res = ""
    for i in range(len(arr)):
        res += fmt % arr[i] + sep
    return res[: -len(sep)]


def str_ndarray(arrs, fmt="%.2f", sep=","):
    if len(arrs.shape) <= 1:

        return str_ndarray_1d(arrs, fmt=fmt, sep=sep)

    if len(arrs.shape) == 2:
        return "\n".join([str_ndarray_1d(arr, fmt=fmt, sep=sep) for arr in arrs])

    return str(arrs)


def is_cuda_model(model: nn.Module):
    first_param = next(model.parameters())  # type: Variable
    return first_param.is_cuda


def toTensor(x):  # 转化x为tensor！
    if type(x) == np.ndarray:
        x = torch.from_numpy(x).type(torch.FloatTensor)
    elif type(x) in [torch.Tensor, torch.FloatTensor, torch.DoubleTensor]:
        x = x.type(torch.FloatTensor)
    elif type(x) == Variable:
        x = x.data
    else:
        x = torch.FloatTensor(x)
    return x


def toVariable(x):
    if type(x) == Variable:
        pass
    else:
        x = Variable(toTensor(x).type(torch.FloatTensor))
    return x


def row_l2(x: Variable):
    batch_row_l2 = torch.bmm(x.dot(torch.transpose(x, 1, 2)))
    return torch.sum(batch_row_l2)


def get_data_loader(x, y, batch_size):
    if x is None or y is None:
        return None
    y2 = toTensor(y).view(-1, 1) if len(y.shape) == 1 else toTensor(y)
    data_loader = DataLoader(
        TensorDataset(toTensor(x), y2), batch_size=batch_size, shuffle=True
    )
    return data_loader


def get_accuracy(yp, y):
    yp2 = torch.sigmoid(yp.squeeze()).data.cpu().numpy()
    yp3 = 1 * (yp2 > 0.5)
    correct = np.sum(y.data.cpu().numpy() == yp3)
    return correct / yp.shape[0]


def get_auc(yp, y):
    yp2 = torch.sigmoid(yp.squeeze()).data.cpu().numpy()
    y2 = y.data.cpu().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y2, yp2, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc
def RLS(Phi,samp_y,lamd): #calculate Regularized LS Regression
    Y = samp_y.reshape(-1, 1)
    Phi = Phi.T
    # theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.ones((Phi.shape[0],1)) ), Phi) , Y)
    theta_RLS = np.dot(np.dot( np.linalg.inv(np.dot(Phi, Phi.T) + lamd*np.eye((Phi.shape[0])) ), Phi) , Y)
    return theta_RLS
def LS(Phi,samp_y): #calculate LeastSquare Regression
    Y = samp_y.reshape(-1, 1)  # reshape Y to 50*1
    # based on the equation
    Phi = Phi.T
    theta_LS = np.dot(np.dot(np.linalg.inv(np.dot(Phi,Phi.T)),Phi),Y)
    return theta_LS

def train(
    model,
    data_loader,
    optimizer,
    i,
    gamma,
    logger=None,
    task="regression",
    michael_reg=False,
    fix_mask=False,
    interaction_order_lambda=5e-2,
    l1_lambda=0,
    pos_linear_input = None,
    num_bin_variable = None,
    num_cat_variable = None,
    num_num_variable = None,
vectorized_cate_col_name_num_list=None,
):
    global global_epoch
    global_epoch = i
    global mt_reg
    mt_reg = michael_reg
    global inter_order_lambda
    inter_order_lambda = interaction_order_lambda
    global reg_lambda
    reg_lambda = l1_lambda
    global freeze_mask
    freeze_mask = fix_mask

    model.train() #


    losses = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = toVariable(x), toVariable(y)

        if is_cuda_model(model):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                model.cpu()

        optimizer.zero_grad()
        input_linear = x[:, 0:pos_linear_input]
        input_linear = input_linear.view(x.shape[0], -1, 1)
        # print(x.shape)
        yp = model(input_linear=input_linear, K=gamma,train_size= x.shape[0],
                   num_cat_variable=num_cat_variable,
                   num_num_variable = num_num_variable, num_bin_variable = num_bin_variable,
                   input_nonlinear=x[:,pos_linear_input:], vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list)
        # print(yp)
        if task == "regression":
            loss = nn.functional.mse_loss(yp, y)
        elif task == "classification":
            loss = nn.functional.binary_cross_entropy_with_logits(
                yp.squeeze(), y.squeeze()
            )
        # if model.model_pwlnet.reg_term() != 0:
        #     print("Has reg, GMZZZZZZ")

        reg_loss = model.model_block.reg_term() + model.model_pwlnet.reg_term() # Wide and Block
        # reg_loss = model.model_deep.reg_term() + model.model_pwlnet.reg_term() + model.model_block.reg_term() # Wide and block and Deep
        # reg_loss = model.model_block.reg_term() + model.model_pwlnet.reg_term() + model.sum_layer.reg_term()  # Test for merging layer
        (reg_loss + loss).backward()

        optimizer.step()
        losses.append(loss.data.item())

    if logger is not None:
        for name, param in model.model_block.named_parameters():
            logger.add_histogram(name, param, i, bins="sqrt")

        for name, param in model.model_pwlnet.named_parameters():
            logger.add_histogram(name, param, i, bins="sqrt")

        interaction_sizes = np.count_nonzero(
            model.model_block.layers[0].mask_raw.data.cpu().numpy(), axis=1
        )
        nonzero_interaction_sizes = np.array(
            [inter for inter in interaction_sizes if inter != 0]
        )

        logger.add_scalar("train_loss", loss.data.item(), i)
        logger.add_scalar(
            "average interaction size", np.mean(nonzero_interaction_sizes), i
        )
        print(np.mean(nonzero_interaction_sizes))
        logger.add_scalar("max interaction size", np.max(interaction_sizes), i)
        logger.add_scalar("min interaction size", np.min(nonzero_interaction_sizes), i)
        logger.add_scalar("std interaction size", np.std(nonzero_interaction_sizes), i)
        logger.add_scalar("num active blocks", np.count_nonzero(interaction_sizes), i)

        logger.add_scalar("dense_penalty", model.model_block.layers[0].penalty, i)
        logger.add_scalar(
            "dense_mask_nonzeros",
            np.count_nonzero(model.model_block.layers[0].mask_raw.data.cpu().numpy()),
            i,
        )
        logger.add_histogram(
            "dense_gated",
            model.model_block.layers[0].mask * model.model_block.layers[0]._origin.weight,
            i,
            bins="sqrt",
        )
        logger.add_histogram("interation sizes", interaction_sizes, i, bins="sqrt")

    return np.mean(losses)


def test(
    model, data_loader, gamma ,epoch=None, va_loader=None, logger=None, task="regression",
    num_bin_variable=None,
    pos_linear_input = None,
    num_cat_variable = None,
    num_num_variable = None,
    vectorized_cate_col_name_num_list=None,
):
    model.eval()

    score_id = "mse" if task == "regression" else "auc"

    loaders = [data_loader]
    eval_scores = []
    if va_loader is not None:
        loaders = [va_loader] + loaders

    for loader_idx, loader in enumerate(loaders):

        losses = []
        for x, y in loader:
            x, y = toVariable(x), toVariable(y)

            if is_cuda_model(model):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                else:
                    model.cpu()
            input_linear = x[:, 0:pos_linear_input]
            input_linear = input_linear.view(x.shape[0], -1, 1)
            yp = model(input_linear=input_linear, K=gamma,train_size= x.shape[0],num_bin_variable=num_bin_variable, num_cat_variable=num_cat_variable, num_num_variable = num_num_variable, input_nonlinear=x[:,pos_linear_input:], vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list)
            if task == "regression":
                loss = nn.functional.mse_loss(yp, y)
                losses.append(loss.data.item())

            elif task == "classification":
                if False:
                    score = nn.functional.binary_cross_entropy_with_logits(
                        yp.squeeze(), y.squeeze()
                    ).data.cpu()[0]
                else:
                    score = get_auc(yp, y)  # "loss" is now calucated as accuracy

                losses.append(score)

        eval_score = np.mean(losses)

        if (logger is not None) and (epoch is not None):
            eval_id = "validation" if loader_idx == 1 else "test"
            logger.add_scalar(eval_id + "_" + score_id, eval_score, epoch)
            logger.add_scalar(
                eval_id + "_dense_mask_nonzeros",
                np.count_nonzero(model.model_block.layers[0].mask_raw.data.cpu().numpy()),
                epoch,
            )
            logger.add_scalar(
                eval_id + "_average interaction size",
                np.mean(
                    np.count_nonzero(
                        model.model_block.layers[0].mask_raw.data.cpu().numpy(), axis=1
                    )
                ),
                epoch,
            )

        eval_scores.append(eval_score)

    if va_loader is not None:
        return eval_scores[0], eval_scores[1]
    else:
        return eval_scores[0]

def wide_and_deep_train(model,
    data_loader,
    optimizer,
    i,
    gamma, num_cat_variable, num_num_variable, vectorized_cate_col_name_num_list,task="regression", ):
    model.train()  #
    losses = []
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = toVariable(x), toVariable(y)

        if is_cuda_model(model):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            else:
                model.cpu()

        optimizer.zero_grad()
        input_linear = x[:, 0:num_num_variable]
        input_linear = input_linear.view(x.shape[0], -1, 1)
        # print(x.shape)
        yp = model(input_linear=input_linear, K=gamma, train_size=x.shape[0],
                   num_cat_variable=num_cat_variable,
                   num_num_variable=num_num_variable,
                   num_bin_variable = None,
                   input_nonlinear=x[:,num_num_variable:],
                   vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list)
        # print(yp)
        if task == "regression":
            loss = nn.functional.mse_loss(yp, y)
        elif task == "classification":
            loss = nn.functional.binary_cross_entropy_with_logits(
                yp.squeeze(), y.squeeze()
            )
        # if model.model_pwlnet.reg_term() != 0:
        #     print("Has reg, GMZZZZZZ")

        reg_loss = model.model_deep.reg_term() + model.model_pwlnet.reg_term()  # Wide and Block
        # reg_loss = model.model_deep.reg_term() + model.model_pwlnet.reg_term() + model.model_block.reg_term() # Wide and block and Deep
        # reg_loss = model.model_block.reg_term() + model.model_pwlnet.reg_term() + model.sum_layer.reg_term()  # Test for merging layer
        (reg_loss + loss).backward()

        optimizer.step()
        losses.append(loss.data.item())
    return np.mean(losses)

def wide_and_deep_test(
    model, data_loader, gamma, num_cat_variable, num_num_variable, vectorized_cate_col_name_num_list,task="regression", va_loader=None,
):
    model.eval()

    score_id = "mse" if task == "regression" else "auc"

    loaders = [data_loader]
    eval_scores = []
    if va_loader is not None:
        loaders = [va_loader] + loaders

    for loader_idx, loader in enumerate(loaders):

        losses = []
        for x, y in loader:
            x, y = toVariable(x), toVariable(y)

            if is_cuda_model(model):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                else:
                    model.cpu()
            input_linear = x[:, 0:num_num_variable]
            input_linear = input_linear.view(x.shape[0], -1, 1)
            yp = model(input_linear=input_linear, K=gamma, train_size=x.shape[0],
                       num_cat_variable=num_cat_variable,
                       num_num_variable=num_num_variable,num_bin_variable = None,
                       input_nonlinear=x[:, num_num_variable:],
                       vectorized_cate_col_name_num_list=vectorized_cate_col_name_num_list)
            if task == "regression":
                loss = nn.functional.mse_loss(yp, y)
                losses.append(loss.data.item())

            elif task == "classification":
                if False:
                    score = nn.functional.binary_cross_entropy_with_logits(
                        yp.squeeze(), y.squeeze()
                    ).data.cpu()[0]
                else:
                    score = get_auc(yp, y)  # "loss" is now calucated as accuracy

                losses.append(score)

        eval_score = np.mean(losses)


        eval_scores.append(eval_score)

    if va_loader is not None:
        return eval_scores[0], eval_scores[1]
    else:
        return eval_scores[0]


def early_stopping_checker(
    es_scores, es_models, step, task="regression", patience=5, val_loss=False
):
    if task == "classification":
        if val_loss:
            crit_score = min(es_scores)
        else:
            crit_score = max(es_scores)
    elif task == "regression":
        crit_score = min(es_scores)
    if len(es_scores) > patience + 1:
        es_models[-1 * (patience + 2)] = None
    crit_idx = es_scores.index(crit_score)
    if step - crit_idx >= patience:
        print("early stopping")
        es_break = True
    else:
        es_break = False
    return crit_idx, es_break


def print_mask(mask):
    print("\n\n")
    if mask.shape[1] > 20:
        for x in mask:
            for y in x:
                if y == 0:
                    print("", end=" ")
                else:
                    print("1", end=" ")
            print()
    else:
        for x in mask:
            for y in x:
                if y == 0:
                    print("", end="\t")
                else:
                    print("%.1e" % abs(y), end="\t")
            print()
    print("\n\n")


def reinitialize_parameters(model):

    for n, p in model.named_parameters():
        if "bi_" in n or ".loc" in n or "mask" in n or ".temp" in n:
            continue
        p.data = p.data.normal_(0, 0.1)


def save_mask(mask, path):
    with open(path, "wb") as handle:
        pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
