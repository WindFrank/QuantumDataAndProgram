# import sys
#
# from sklearn import linear_model        #表示，可以调用sklearn中的linear_model模块进行线性回归。
# import numpy as np
# import torch as t
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
#
import sys

from RandomCircuit.Utils import read_xls
from xlsTest import write_to_excel
#
#
# class LinerRegression(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(LinerRegression, self).__init__()
#         self.fc1 = nn.Linear(in_size, out_size)
#
#     def forward(self, x):
#         y_hat = self.fc1(x)
#         return y_hat
#
#
# def train():
#     batch_data_x = []
#     batch_data_y = []
#     loss_all = 0
#     for (train_info, index) in zip(train_data, range(len(train_data))):
#         row_batch_data = np.array([float(train_info[1]), float(train_info[2]), float(train_info[3]), float(train_info[4]), float(train_info[6])], dtype=t.float)
#         row_batch_data_y = np.array([float(train_info[5])], dtype=t.float)
#         batch_data_x.append(row_batch_data)
#         batch_data_y.append(row_batch_data_y)
#         if (index + 1) % batch_size == 0:
#             x = t.from_numpy(batch_data_x)
#             y = t.from_numpy(batch_data_y)
#             y_hat = model(x.float())
#             loss = criterion(y_hat, y)
#             loss_all += loss
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             sys.stdout.write(f'\rtrain_count:{index + 1} loss_all:{loss_all}')
#             sys.stdout.flush()
#             batch_data_x = []
#             batch_data_y = []
#     print()
#
#
# def test():
#     a = model.predict([[12]])
#     print("预测一张12英寸匹萨价格：{:.2f}".format(model.predict([[12]])[0][0]))
#
#
# if __name__ == '__main__':
#     circuits_info = read_xls('TrainData/e3_13_gate/GNN_training_data_13_gate_simulate_e3_ALL.xlsx')
#     circuits_info.pop(0)
#     train_data = []
#     test_data = []
#
#     for (circuit_info, i) in zip(circuits_info, range(len(circuits_info))):
#         if i % 5 == 0:
#             test_data.append(circuit_info)
#         else:
#             train_data.append(circuit_info)
#
#     xls_data = [['epoch', 'average_error']]
#     row_data = []
#     batch_size = 20
#     in_size = 5
#     out_size = 1
#     lr = 0.01
#     model = LinerRegression(in_size, out_size)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     for epoch in range(1000):
#         print(f'epoch:{epoch + 1}')
#         row_data.append(epoch + 1)
#         train()
#         '''print('origin:')
#         test('circuit_data_in_noise_model_3w13700.xlsx')'''
#         print('simulation_test:')
#         if (epoch + 1) % 5 == 0:
#             test()
#         row_data = []
#         if (epoch + 1) % 10 == 0:
#             write_to_excel(f'GNN_fidelity_new_gate_e3_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
#         # if (epoch + 1) % 50 == 0:
#         #     lr -= 0.000002
#         #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
#         print()
#
#
#
#     # a[0][0]

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.nn import init
import torch.optim as optim

'''生成数据集
'''
circuits_info = read_xls('TrainData/e3_13_gate/GNN_training_data_13_gate_simulate_e3_ALL.xlsx')
circuits_info.pop(0)
train_data = []
train_label = []
test_data = []
test_label = []

for (circuit_info, i) in zip(circuits_info, range(len(circuits_info))):
    if i % 5 == 0:
        row_batch_data_x = [float(circuit_info[1]), float(circuit_info[2]), float(circuit_info[3]), float(circuit_info[4])]
        test_data.append(row_batch_data_x)
        row_batch_data_y = [float(circuit_info[5])]
        test_label.append(row_batch_data_y)
    else:
        row_batch_data_x = [float(circuit_info[1]), float(circuit_info[2]), float(circuit_info[3]), float(circuit_info[4])]
        train_data.append(row_batch_data_x)
        row_batch_data_y = [float(circuit_info[5])]
        train_label.append(row_batch_data_y)

num_inputs = 4
features = torch.tensor(train_data, dtype=torch.float)
labels = torch.tensor(train_label, dtype=torch.float)

'''读取数据
'''
import torch.utils.data as Data

batch_size = 20
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
''' 定义模型
'''


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        #  n_feature表示输入个数（特征个数），1表示 输出层的个数 =神经元的个数
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
# print(net) # 使用print可以打印出网络的结构
# LinearNet( (linear): Linear(in_features=2, out_features=1, bias=True))
'''
# 比较方便的写法 Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
# 训练的参数 net目前只有个数，net.parameters()只是给所有可学习的参数随机化了一个值
for param in net.parameters():
    print(param)

# 注意：torch.nn仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用input.unsqueeze(0)来添加一维。
'''

'''初始化参数 ，初始化后net.parameters()的初始值则确定
'''

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

'''定义损失函数  # 均方误差损失
'''
loss = nn.MSELoss()

'''定义优化器
# 多个网络的优化器
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
'''
optimizer = optim.Adam(net.parameters(), lr=0.00012)
# print(optimizer)

'''训练模型
'''
num_epochs = 3
xls_data = [['epoch', 'average_error']]
for epoch in range(1000):
    loss_all = 0
    count = 0
    for X, y in data_iter:
        net.train()
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        loss_all += l
        optimizer.step()
        sys.stdout.write(f'\rtrain_count:{count * 20} loss_all:{loss_all}')
        sys.stdout.flush()
        count += 1
    print()
    # if (epoch + 1) % 5 == 0:
    all_error = 0
    average_error = 0
    for (test_feature, test_a_label, s_count) in zip(test_data, test_label, range(len(test_data))):
        net.eval()
        test_feature = torch.tensor(test_feature, dtype=torch.float)
        result = net(test_feature).item()
        error = abs(result - test_a_label[0])
        all_error += error
        average_error = all_error / (s_count + 1)
        sys.stdout.write(f'\rtest all_count:{s_count + 1} right:{test_a_label} predict:{result} error:{error} average_error:{all_error / (s_count + 1)}')
        sys.stdout.flush()
    print()
    xls_data.append([epoch + 1, average_error])
    write_to_excel(f'Linear_fidelity_new_gate_e3.xlsx', 'sheet1', 'info', xls_data)


