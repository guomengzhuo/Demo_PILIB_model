import operator as op
from collections import defaultdict
from functools import reduce

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam

import utils
from layer_l0 import GroupL0Linear, PenaltyLayer
from layers import GroupBlockLayer, GroupMergeLayer
# *arg表示任意多个无名参数，类型为tuple;**kwargs表示关键字参数，为dict，使用时需将*arg放在**kwargs之前

REG_FUNCTIONS = {
    "l1": lambda x: nn.functional.l1_loss(x, torch.zeros_like(x), size_average=False),
    "l2": lambda x: nn.functional.mse_loss(x, torch.zeros_like(x), size_average=False),
    "row_l2": utils.row_l2,
}


def createMLP(n_inp, n_out, hidden_units, activation=nn.ReLU): # 建立简单的MLP
    layers = []
    layers_size = [n_inp] + hidden_units
    for i in range(len(layers_size) - 1):
        layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))
        if activation is not None:
            layers.append(activation())
    layers.append(nn.Linear(layers_size[-1], n_out))

    return nn.Sequential(*layers)
# For example: mlp = createLMLP(1,1,[2,4,6],),
# Sequential(
#   (0): Linear(in_features=1, out_features=2, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=2, out_features=4, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=4, out_features=6, bias=True)
#   (5): ReLU()
#   (6): Linear(in_features=6, out_features=1, bias=True)
# )

def initialize_lognormal(layer, dims):  # 初始化层权重
    mu, sigma = 1, 1  # mean and standard deviation
    s = np.random.lognormal(mu, sigma, dims) / 50
    ri = 2 * np.random.randint(2, size=dims) - 1
    s = ri * s
    layer.weight.data = torch.FloatTensor(s)

    return layer


def createMLPWithBlockLayers(
    n_inp,
    n_out,
    hidden_units, # 是一个列表 包含各个层的hidden layer 例如 [10,20,30]
    n_group,
    max_interaction_order=2,
    l0_lambda=1e-3,
    n_dense_layer=1,
    activation=nn.ReLU,
    use_l0=True,
    use_lognormal=False,
    use_fixed_mask=False,
):
    layers = []
    hidden_units = [n_inp] + hidden_units # 将输入层input添加到hidden 层

    penalty_layer = None
    if use_l0:  # 是否用L0 Norm
        assert n_dense_layer == 1 # 判断n_dense_layer是否为1 为1才正常执行
        group_l0_layer = GroupL0Linear(
            hidden_units[0],
            hidden_units[1],
            n_group,
            max_interaction_order=max_interaction_order,
            l0_lambda=l0_lambda,
            use_fixed_mask=use_fixed_mask,
        )
        penalty_layer = PenaltyLayer()
        layers.append(group_l0_layer)
        layers.append(penalty_layer)
        if activation is not None:
            layers.append(activation())
    else:
        for i in range(1, 1 + n_dense_layer):
            dims = [hidden_units[i], hidden_units[i - 1]]
            dense_layer = nn.Linear(hidden_units[i - 1], hidden_units[i])
            if use_lognormal:
                dense_layer = initialize_lognormal(dense_layer, dims)
            layers.append(dense_layer)
            if activation is not None:
                layers.append(activation())

    for i in range(1 + n_dense_layer, len(hidden_units)): # 只在第一层加入L0 Norm？ Group Hidden Layers
        layers.append(
            GroupBlockLayer(
                n_group,
                int(hidden_units[i - 1] / n_group),
                int(hidden_units[i] / n_group),
            )
        )
        if activation is not None:
            layers.append(activation())
    layers.append(GroupBlockLayer(n_group, int(hidden_units[-1] / n_group), n_out)) # Sum Each blocks
    layers.append(GroupMergeLayer(n_group, n_out)) # Sum all blocks

    return nn.Sequential(*layers), penalty_layer


class MainEffectsMLP(nn.Module):
    def __init__(
        self,
        n_inp,
        n_out,
        hidden_units,
        uni_hidden_unit=0,
        n_layers_uni=2,
        reg_lambda=0,
        reg_method="l2",
        **kwargs,
    ):
        super(MainEffectsMLP, self).__init__()

        self.layers = createMLP(n_inp, n_out, hidden_units)

        if uni_hidden_unit > 0:
            uni_layers = [
                createMLP(1, n_out, [uni_hidden_unit] * n_layers_uni)
                for i in range(n_inp)
            ]
            for i in range(n_inp):
                setattr(self, f"_uni_{i}", uni_layers[i])
            self.uni_layers = uni_layers
        else:
            self.uni_layers = None

        self._reg_lambda = reg_lambda
        self._reg_method = REG_FUNCTIONS[reg_method]

    def reg_term(self):
        if abs(self._reg_lambda) < 1e-15:
            return 0

        reg_func = self._reg_method
        reg_loss = 0
        for name, param in self.named_parameters():
            reg_loss += self._reg_method(param)

        # Test GMZ
        # print('Reg_term in MainEffectMLP')

        return reg_loss * self._reg_lambda

    def forward(self, x):
        y = self.layers(x)

        if self.uni_layers is not None:
            uni_nets = [
                uni_layer(x[:, i : i + 1])
                for i, uni_layer in enumerate(self.uni_layers)
            ]

            uni_net = reduce(op.add, uni_nets)
            y = y + uni_net

        return y

    def get_weights(self):
        w_dicts = {}
        cnt = 1
        for layer in self.layers:
            if hasattr(layer, "weight"):
                w_dicts[f"h{cnt}"] = layer.weight.data.cpu().numpy().T
                cnt += 1

        return w_dicts

# Pwl network
class PwlNetwork(torch.nn.Module):
    def __init__(self, input1, input2,reg_lambda=0,reg_method="l2", **kwargs): # input1: num of all intervals, input2 : num of criteria
        super(PwlNetwork, self).__init__()
        self.layer_piecewise = torch.nn.Conv1d(input1, input1, 1, 1, 0, bias=True, groups=input1)  # group = number of criteria
        self.layer_summation = torch.nn.Conv1d(input2, input2, 1, 1, 0, bias=True, groups=input2)
        self._reg_lambda = reg_lambda
        self._reg_method = REG_FUNCTIONS[reg_method]
        # layer_piecewise train for each element in the vectorized data point
        # layer_summation sums all up

    def forward(self, input_linear, K, train_size, num_cat_variable, num_num_variable, num_bin_variable, vectorized_cate_col_name_num_list):
        y1 = self.layer_piecewise(input_linear)  # First layer in linear part: correspond to each marginal score
        # For cate and bin
        if vectorized_cate_col_name_num_list is not None:
            # print("yes")
            y1_bin = y1[:, 0:num_bin_variable, :]
            # print(y1_bin)
            y1_cat = y1[:, num_bin_variable:(num_cat_variable + num_bin_variable),:]
            y1_num = y1[:, num_cat_variable + num_bin_variable:, :]
            temp_y1_cat = torch.zeros(size=(train_size, len(vectorized_cate_col_name_num_list), 1))
            count = 0
            for index, num_of_interval in enumerate(vectorized_cate_col_name_num_list):
                temp_y1_cat[:,index,:] = y1_cat[:,count:count+num_of_interval,:].sum(dim=1)
                count += num_of_interval
            # print(temp_y1_cat)
            # For numerical variables
            index = np.linspace(0, num_num_variable, num_num_variable, endpoint=False, dtype=int)
            assert num_num_variable % K == 0
            temp_y1_num = torch.zeros(size=(train_size, int(num_num_variable/K), 1))
            for i in range(0, K):
                temp_index = index[i::K]
                temp_y1_num += y1_num[:, temp_index, :]
            y1 = torch.cat((y1_bin, temp_y1_cat, temp_y1_num), dim=1)
            # print(y1.shape)
            # print(temp_y1_num)
            # print(y1)
        # For categorical variables
        # if vectorized_cate_col_name_num_list is not None :
        #     y1_cat = y1[:, 0:num_cat_variable,:]
        #     y1_num = y1[:, num_cat_variable:, :]
        #     temp_y1_cat = torch.zeros(size=(train_size, len(vectorized_cate_col_name_num_list), 1))
        #     count = 0
        #     for index, num_of_interval in enumerate(vectorized_cate_col_name_num_list):
        #         temp_y1_cat[:,index,:] = y1_cat[:,count:count+num_of_interval,:].sum(dim=1)
        #         count += num_of_interval
        #     # print(temp_y1_cat)
        #     # For numerical variables
        #     index = np.linspace(0, num_num_variable, num_num_variable, endpoint=False, dtype=int)
        #     assert num_num_variable % K == 0
        #     temp_y1_num = torch.zeros(size=(train_size, int(num_num_variable/K), 1))
        #     for i in range(0, K):
        #         temp_index = index[i::K]
        #         temp_y1_num += y1_num[:, temp_index, :]
        #     y1 = torch.cat((temp_y1_cat, temp_y1_num), dim=1)
        #     # print(temp_y1_num)
        #     # print(y1)
        else:
            index = np.linspace(0, num_num_variable, num_num_variable, endpoint=False, dtype=int)
            assert num_num_variable % K == 0
            temp_y1 = torch.zeros(size=(train_size, int(num_num_variable/K), 1))
            for i in range(0, K):
                temp_index = index[i::K]
                temp_y1 += y1[:, temp_index, :]
            y1 = temp_y1  # Sum all pieces for each attribute
        #############
        # For each attribute, we give an individual weights
        y2 = self.layer_summation(y1)
        linear_out = torch.sum(y2, dim=1)
        return linear_out
    def reg_term(self):
        reg_loss = 0
        if abs(self._reg_lambda) < 1e-15:
            return 0
        # if self._reg_lambda > 0 and utils.freeze_mask:
        # if self._reg_lambda > 0 and not utils.freeze_mask:
        if self._reg_lambda > 0:
            for name, param in self.named_parameters():
                reg_loss += self._reg_method(param) * self._reg_lambda
        else:
            pass
        # Test GMZ
        # print('Reg_term in MainEffectMLP')

        return reg_loss

class DeepMLP(torch.nn.Module):
    def __init__(self, n_inp, reg_lambda=0, reg_method="l2", **kwargs):
        super(DeepMLP,self).__init__()
        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Linear(n_inp, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self._reg_lambda = reg_lambda
        self._reg_method = REG_FUNCTIONS[reg_method]
    def forward(self, input_nonlinear):
        nonlinear_out = self.layer3_2(input_nonlinear)
        return nonlinear_out
    def reg_term(self):
        reg_loss = 0
        if abs(self._reg_lambda) < 1e-15:
            return 0
        # if self._reg_lambda > 0 and utils.freeze_mask:
        # if self._reg_lambda > 0 and not utils.freeze_mask:
        if self._reg_lambda > 0:
            for name, param in self.named_parameters():
                reg_loss += self._reg_method(param) * self._reg_lambda
        else:
            pass
        # Test GMZ
        # print('Reg_term in MainEffectMLP')

        return reg_loss

# BlockMainEffectsMLP <- createMLPWithBlockLayers <- GroupL0Norm <- L0Norm
class BlockMainEffectsMLP(MainEffectsMLP):
    def __init__(
        self,
        n_inp,
        n_out,
        hidden_units,
        n_group=2,
        uni_hidden_unit=0,
        n_layers_uni=2,
        reg_lambda=0,
        reg_method="l2",
        max_interaction_order=2,
        l0_lambda=1e-3,
        # use_weight_norm=False,
        use_fixed_mask=False,
        **kwargs,
    ):

        super(BlockMainEffectsMLP, self).__init__(
            n_inp,
            n_out,
            hidden_units,
            uni_hidden_unit,
            n_layers_uni,
            reg_lambda,
            reg_method,
            **kwargs,
        )

        self.layers, self.penalty_layer = createMLPWithBlockLayers(  # 返回的是划分了block的layers
            n_inp,
            n_out,
            hidden_units,
            n_group,
            max_interaction_order=max_interaction_order,
            l0_lambda=l0_lambda,
            use_l0=True,
            use_fixed_mask=use_fixed_mask,
        )

        self._n_inp = n_inp
        self._n_group = n_group
        self._l0_lambda = l0_lambda
        self.use_fixed_mask = use_fixed_mask

    def reg_term(self):
        reg_loss = 0
        # print(utils.freeze_mask)
        if self._reg_lambda > 0 and utils.freeze_mask:   # Default utils.freeze_mask = False
            for name, param in self.named_parameters():
                if "layers.0._origin.weight" in name:
                    reg_loss += (
                        self._reg_method(param * self.layers[0].mask) * self._reg_lambda
                    )
                elif "_b_" not in name and "_uni_" not in name:
                    reg_loss += self._reg_method(param) * self._reg_lambda

        if self.penalty_layer is not None:
            #Test GMZ, This part is processed
            # print('BP 2 in BlockmaineffectMLP')
            reg_loss += self.penalty_layer.penalty  # * self._l0_lambda

            # print(reg_loss)

        return reg_loss

    def get_block_func(self, block_idx, x):
        assert self.use_fixed_mask, "Freeze mask before running this"

        if self._block_func is None or self._block_func[block_idx] is None:
            self._build_block_funcs()
        return self._block_func[block_idx]

    def _build_block_funcs(self):
        mask = self.layers[0].trained_mask

class WideAndBlockAndDeep(torch.nn.Module):
    def __init__(self,
        input1,
        input2,
        n_inp,
        n_out,
        hidden_units,
        n_group=2,
        uni_hidden_unit=0,
        n_layers_uni=2,
        reg_lambda=0,
        reg_method="l2",
        max_interaction_order=2,
        l0_lambda=1e-3,
        # use_weight_norm=False,
        use_fixed_mask=False,
        **kwargs,):
        super(WideAndBlockAndDeep,self).__init__()
        self.model_pwlnet = PwlNetwork(input1=input1, input2=input2)
        self.model_block = BlockMainEffectsMLP(n_inp=n_inp,
                                               n_out=n_out,
                                               hidden_units=hidden_units,
                                               n_group = n_group,
                                               uni_hidden_unit=uni_hidden_unit,
                                               n_layers_uni=n_layers_uni,
                                               reg_lambda=reg_lambda,
                                               reg_method=reg_method,
                                               max_interaction_order=max_interaction_order,
                                               l0_lambda=l0_lambda,
                                               use_fixed_mask=use_fixed_mask)
        self.model_deep = DeepMLP(n_inp=n_inp)

    def forward(self, input_linear, K, train_size, num_cat_variable ,num_num_variable, input_nonlinear, vectorized_cate_col_name_num_list,):
        x1 = self.model_pwlnet(input_linear, K, train_size, num_cat_variable ,num_num_variable, vectorized_cate_col_name_num_list)
        x2 = self.model_block(input_nonlinear)
        x3 = self.model_deep(input_nonlinear)
        return x1+x2+x3

class WideandDeep(torch.nn.Module):
    def __init__(self,
                 input1,
                 input2,
                 n_inp,
                 reg_lambda=0,
                 reg_method="l2",
                 **kwargs,):
        super(WideandDeep,self).__init__()
        self.model_pwlnet = PwlNetwork(input1=input1, input2=input2, reg_lambda=reg_lambda,
                                       reg_method=reg_method, )
        self.model_deep = DeepMLP(n_inp = n_inp)

    def forward(self, input_linear, K, train_size, num_cat_variable, num_num_variable, input_nonlinear, num_bin_variable,
                vectorized_cate_col_name_num_list ):
        x1 = self.model_pwlnet(input_linear, K, train_size, num_cat_variable ,num_num_variable,num_bin_variable, vectorized_cate_col_name_num_list)
        x2 = self.model_deep(input_nonlinear)
        return x1 + x2

class Sum_layer(torch.nn.Module):
    # A merge layer for two parts.
    def __init__(self,
        reg_lambda=0,
        reg_method="l2",):
        super(Sum_layer, self).__init__()
        self.sum_layer = torch.nn.Linear(2,1,bias=False)
        self._reg_lambda = reg_lambda
        self._reg_method = REG_FUNCTIONS[reg_method]
    def forward(self, sum_input):
        # print('Sum_layer is used')
        return self.sum_layer(sum_input)
    def reg_term(self):
        reg_loss = 0
        if abs(self._reg_lambda) < 1e-15:
            return 0
        # if self._reg_lambda > 0 and utils.freeze_mask:
        # if self._reg_lambda > 0 and not utils.freeze_mask:
        if self._reg_lambda > 0:
            # print('Sum_layer.reg_term is used')
            for name, param in self.named_parameters():
                reg_loss += self._reg_method(param) * self._reg_lambda
        else:
            pass
        return reg_loss

class WideAndBlock(torch.nn.Module):
    def __init__(self,
        input1,
        input2,
        n_inp,
        n_out,
        hidden_units,
        n_group=2,
        uni_hidden_unit=0,
        n_layers_uni=2,
        reg_lambda=0,
        reg_method="l2",
        max_interaction_order=2,
        l0_lambda=1e-3,
        # use_weight_norm=False,
        use_fixed_mask=False,
        vectorized_cate_col_name_num_list = None,
        **kwargs,):
        super(WideAndBlock,self).__init__()
        self.model_pwlnet = PwlNetwork(input1=input1, input2=input2,reg_lambda=reg_lambda,
                                               reg_method=reg_method,)
        self.model_block = BlockMainEffectsMLP(n_inp=n_inp,
                                               n_out=n_out,
                                               hidden_units=hidden_units,
                                               n_group = n_group,
                                               uni_hidden_unit=uni_hidden_unit,
                                               n_layers_uni=n_layers_uni,
                                               reg_lambda=reg_lambda,
                                               reg_method=reg_method,
                                               max_interaction_order=max_interaction_order,
                                               l0_lambda=l0_lambda,
                                               use_fixed_mask=use_fixed_mask)
        self.sum_layer = Sum_layer(reg_lambda=reg_lambda,reg_method=reg_method)
    def forward(self, input_linear, K, train_size, num_cat_variable ,num_num_variable, num_bin_variable, input_nonlinear, vectorized_cate_col_name_num_list, ):
        x1 = self.model_pwlnet(input_linear, K, train_size, num_cat_variable ,num_num_variable, num_bin_variable, vectorized_cate_col_name_num_list,)  #num_num_variable, vectorized_cate_col_name_num_list = None
        x2 = self.model_block(input_nonlinear)
        # sum_layer_inp = torch.cat((x1, x2), dim=1)
        # output = self.sum_layer(sum_layer_inp)
        return x1 + x2
        # return output
        # return torch.FloatTensor(np.array(0.5))*x1+torch.FloatTensor(np.array(0.5))*x2

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
            self._ws.append(Variable(torch.FloatTensor(w), requires_grad=False))
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


class GAMNetsMLP(nn.Module):
    def __init__(
        self,
        n_inp,
        n_out,
        hidden_units,
        interactions,
        reg_lambda=0,
        reg_method="l2",
        **kwargs,
    ):
        super(GAMNetsMLP, self).__init__()

        self.interactions = interactions

        interaction_layers = [
            createMLP(len(interaction), n_out, hidden_units)
            for interaction in interactions
        ]
        for i in range(len(interactions)): # 给不同的layer起个名字
            setattr(self, f"_inter_{i}", interaction_layers[i])
        self.interaction_layers = interaction_layers

        if reg_lambda > 0:

            def reg_term(self):
                reg_func = REG_FUNCTIONS[reg_method]
                reg_loss = 0
                for name, param in self.named_parameters():
                    reg_loss += reg_func(param)
                return reg_loss * reg_lambda

            self.reg_term = lambda: reg_term(self)

        else:
            self.reg_term = lambda: 0

    def forward(self, x):
        y = 0
        inters = self.interactions

        if self.interaction_layers is not None:

            interaction_nets = []
            for i, interaction_layer in enumerate(self.interaction_layers):
                inter = inters[i]
                inter2 = np.array([a - 1 for a in inter])

                interaction_nets.append(interaction_layer(x[:, inter2]))

            interaction_net = reduce(op.add, interaction_nets)

            y = y + interaction_net

        return y

    def get_weights(self):
        w_dicts = {}
        cnt = 1
        for layer in self.layers:
            if hasattr(layer, "weight"):
                w_dicts[f"h{cnt}"] = layer.weight.data.cpu().numpy().T
                cnt += 1

        return w_dicts


if __name__ == "__main__":
    x = Variable(torch.rand(1, 10))
    print(x)
    model = BlockMainEffectsMLP(10, 1, [140, 100, 60, 20], 10)
    print(model.Parameter.data)
    y = model(x)
    print(y)