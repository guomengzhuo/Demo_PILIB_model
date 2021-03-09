### this particular L0 implementation is based on the github repo at https://github.com/moskomule/l0.pytorch
### at the time of conducting experiments, the official code repo was not available

import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import utils

add_noise_to_mask = False


def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class _L0Norm(nn.Module):
    def __init__(
        self,
        origin,
        n_group=1,
        max_interaction_order=4,
        l0_lambda=1e-3,
        loc_mean=0,
        loc_sdev=0.01,
        beta=2 / 3,
        gamma=-0.1,
        zeta=1.1,
        fix_temp=True,
    ):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()

        self._origin = origin
        self._size = self._origin.weight.size()  # 输出torch.Size([layer size])
        self.mask_raw = None
        self.n_group = n_group
        if n_group >= 1:
            assert self._size[0] % n_group == 0 # 判断layer的output的数量能都被group数整除
            self.n_repeat = self._size[0] // n_group # 取整
            self._size = torch.Size((n_group, self._size[1]))
            self.expandor = Variable(
                torch.FloatTensor([[[1]] for i in range(self.n_repeat)])
            )

        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev)) # 这个函数理解为类型转换函数，
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)
        self.max_interaction_order = max_interaction_order
        self.l0_lambda = l0_lambda

    def _get_mask(self):
        # Test GMZ

        if self.training:
            # print('_get mask')
            self.uniform.uniform_()
            u = Variable(self.uniform)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = F.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio)
            relu = torch.nn.ReLU()
            interaction_sizes = penalty.sum(dim=1)
            # Test GMZ
            # print(interaction_sizes)
            temp = torch.ones_like(interaction_sizes)*2

            penalty = relu(
                interaction_sizes.max() - self.max_interaction_order
            ) + self.l0_lambda * (
                # (torch.sqrt((interaction_sizes - temp)*(interaction_sizes - temp))).sum()
                    (((interaction_sizes - temp) * (interaction_sizes - temp))).sum()
                / np.count_nonzero(interaction_sizes.data.cpu().numpy())
            )
            # penalty = self.l0_lambda * (
            #               # (torch.sqrt((interaction_sizes - temp)*(interaction_sizes - temp))).sum()
            #                   (((interaction_sizes - temp) * (interaction_sizes - temp))).sum()
            #                   / np.count_nonzero(interaction_sizes.data.cpu().numpy())
            #           )
            # print(penalty)
        else:
            np.random.seed(0)
            u = Variable(torch.FloatTensor(np.random.uniform(size=self._size)))
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = 0

        s = hard_sigmoid(s)

        self.mask_raw = s

        if self.n_group >= 1:
            s2 = s * self.expandor
            s2 = s2.transpose(0, 1).contiguous() # transpose 转置 contiguous 是为了后续用view处理不报错
            s2 = s2.view((self._size[0] * self.n_repeat, self._size[1])) #
        return s2, penalty


def get_permuted_mask(mask, expandor, _size, n_repeat):
    mask2 = np.copy(mask.cpu().numpy())

    to_permute = np.random.choice(mask2.shape[0], mask2.shape[0] // 2, replace=False)

    for i in to_permute:
        mask2[i, :] = np.random.permutation(mask2[i, :]) # 打乱顺序
    pmask_raw = Variable(torch.FloatTensor(mask2))

    pmask = pmask_raw * expandor
    pmask = pmask.transpose(0, 1).contiguous()
    pmask = pmask.view((_size[0] * n_repeat, _size[1]))

    return pmask.data, pmask_raw.data


class GroupL0Linear(_L0Norm):  # 声明GroupL0Linear的父类是_L0Norm
    def __init__(
        self,
        in_features,
        out_features,
        n_group,
        max_interaction_order=2,
        l0_lambda=1e-3,
        bias=True,
        use_fixed_mask=False,
        **kwargs
    ):
        super(GroupL0Linear, self).__init__(  # super()为了首先找到GroupL0Linear的父类 也就是_L0Norm
            # __init__: 在创建实例的时候，把一些我们认为必须绑定的属性强制填写进去;两个下划线开头的函数是声明该属性为私有
            # 不能在类的外部被使用或访问
            nn.Linear(in_features, out_features, bias=bias), # origin in L0Norm
            n_group=n_group,
            max_interaction_order=max_interaction_order,
            l0_lambda=l0_lambda,
            **kwargs
        )
        self.use_fixed_mask = use_fixed_mask
        self.register_buffer("trained_mask", torch.zeros(self._origin.weight.size()))

    def forward(self, input):
        if self.use_fixed_mask:
            self.penalty = 0
            self.mask = Variable(self.trained_mask)
        else:
            self.mask, penalty = self._get_mask()
            self.penalty = penalty

            if not hasattr(self, "fixed_mask"): # 判断self对象中是否存在 fixed_mask属性
                interaction_sizes = np.count_nonzero(
                    self.mask_raw.data.cpu().numpy(), axis=1  # 每一行！中存在的非零数量
                )
            else:
                interaction_sizes = np.count_nonzero(
                    self.fixed_mask_raw.cpu().numpy(), axis=1
                )

            if np.max(interaction_sizes) <= self.max_interaction_order:
                self.start_waiting_period = True

            if (
                utils.freeze_mask
                and np.max(interaction_sizes) <= self.max_interaction_order
            ):
                if not hasattr(self, "fixed_mask"):

                    if add_noise_to_mask:
                        self.fixed_mask, self.fixed_mask_raw = get_permuted_mask(
                            self.mask_raw.data, self.expandor, self._size, self.n_repeat
                        )

                    else:
                        self.fixed_mask = self.mask.data
                        self.fixed_mask_raw = self.mask_raw.data

                self.mask = Variable(self.fixed_mask)
                self.mask_raw = Variable(self.fixed_mask_raw)
                self.penalty = 0

            self.trained_mask.copy_(self.mask.data)
        return (
            F.linear(input, self._origin.weight * self.mask, self._origin.bias),
            self.penalty,
        )


class L0Linear(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(L0Linear, self).__init__(
            nn.Linear(in_features, out_features, bias=bias), **kwargs
        )

    def forward(self, input):
        self.mask, self.penalty = self._get_mask()
        return (
            F.linear(input, self._origin.weight * self.mask, self._origin.bias),
            self.penalty,
        )


class PenaltyLayer(nn.Module):
    def forward(self, x):
        self.penalty = x[1]
        # Test GMZ, the Reg_loss is the value of x[1]
        # print(x[1])
        return x[0]
