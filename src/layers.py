import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GroupBlockLayer(nn.Module):  # 生产Block layer
    def __init__(self, n_group, size_group_inp, size_group_out):
        super(GroupBlockLayer, self).__init__()

        torch.random.manual_seed(42)
        self.block_ws = []
        self.block_bs = []
        for i in range(n_group):
            block_w = nn.Parameter(torch.rand(size_group_out, size_group_inp))
            self.block_ws.append(block_w)
            setattr(self, f"_block_w_{i}", block_w)  # setattr() 函数对应函数 getattr()，用于设置属性值，该属性不一定是存在的。
            # setattr(object, attribute, value)
            # format string: _block_1, _block_2...

            block_b = nn.Parameter(torch.rand(size_group_out))
            self.block_bs.append(block_b)
            setattr(self, f"_block_b_{i}", block_b)

        self.n_group = n_group
        self.size_group_inp = size_group_inp
        self.size_group_out = size_group_out

    def forward(self, x):
        if len(x.size()) > 2: # 如果输入数据维度大于2 把其展平
            x = x.view(x.size()[0], -1)

        assert x.size()[1] == self.size_group_inp * self.n_group

        group_y = [
            F.linear(
                x[:, i * self.size_group_inp : (i + 1) * self.size_group_inp],
                self.block_ws[i],
                self.block_bs[i],
            )
            for i in range(self.n_group) # 对于每一个group 都采用相同的结构、初始化
        ]
        y = torch.stack(group_y, dim=1) #沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        y = y.view(x.size()[0], self.n_group * self.size_group_out)

        return y

    def __repr__(self): # 创建对类的object的属性描述 用于直接打印对象
        return (
            f"GroupLinear(group_in_features={self.size_group_inp}, "
            + f"group_out_features={self.size_group_out}, "
            + f"n_group={self.n_group})"
        )

    @property  #Python内置的@property装饰器就是负责把一个方法变成属性调用的：
    def weight(self):
        ws = torch.zeros(
            self.n_group * self.size_group_inp, self.n_group * self.size_group_out
        )
        sinp = self.size_group_inp
        sout = self.size_group_out
        n = self.n_group
        for i in range(n):
            ws[i * sinp : (i + 1) * sinp, i * sout : (i + 1) * sout] = self.block_ws[
                i
            ].data  # block_ws 已经被包装为torch.Parameter .data是提取数值部分
        ws = ws.transpose(0, 1)

        return Variable(ws)

    @property
    def block_weight(self):
        return self.block_ws

    @property
    def block_bias(self):
        return self.block_bs


class GroupMergeLayer(nn.Module):
    def __init__(self, n_group, size_out):
        super(GroupMergeLayer, self).__init__()
        self.n_group = n_group
        self.size_out = size_out

    def forward(self, x: Variable):
        if len(x.size()) > 2:
            x = x.view(x.size()[0], -1)

        assert x.size()[1] % self.n_group == 0
        x = x.view(x.size()[0], self.n_group, -1)
        x = x.sum(dim=1, keepdim=False)

        return x

    def __repr__(self):
        return (
            f"GroupMergeLayer(features={self.size_out}, " + f"n_group={self.n_group})"
        )


class GroupWeightedMergeLayer(nn.Module):
    def __init__(self, n_group, size_out):
        super(GroupWeightedMergeLayer, self).__init__()
        self.n_group = n_group
        self.size_out = size_out
        self.w = nn.Parameter(torch.rand(n_group, size_out))

    def forward(self, x: Variable):
        if len(x.size()) > 2:
            x = x.view(x.size()[0], -1)

        assert x.size()[1] % self.n_group == 0
        x = x.view(x.size()[0], self.n_group, -1)
        x = torch.mul(x, self.w)
        x = x.sum(dim=1, keepdim=False)

        return x

    def __repr__(self):
        return (
            f"GroupWeightedMergeLayer(features={self.size_out}, "
            + f"n_group={self.n_group})"
        )
