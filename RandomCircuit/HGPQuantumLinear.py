import sys

import random
import torch
from torch.nn import init
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv
from torch import nn


from RandomCircuit1 import trans_circuit_to_graph_delta
from layers import SAGPool
import argparse
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F


import numpy as np
import scipy.sparse as sp

from Utils import read_xls

from xlsTest import write_to_excel

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--num_features', type=int, default=38,
                    help='num_features')
parser.add_argument('--num_classes', type=int, default=1,
                    help='num_classes')

class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio

        self.conv1 = SAGEConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = SAGEConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = SAGEConv(self.nhid, self.nhid)
        self.pool3 = SAGPool(self.nhid, ratio=self.pooling_ratio)

        self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x


datas = []
test_datas = []
args = parser.parse_args()
circuits_info = read_xls('0.01_e1_delta_30000.xlsx')
circuits_info.pop(0)
split_count = 0
for data_circuit in circuits_info:
    data_graph = trans_circuit_to_graph_delta(data_circuit[0], data_circuit[10])
    if split_count % 5 == 0:
        test_datas.append(data_graph)
    else:
        datas.append(data_graph)
    split_count += 1
circuits_info = None
global_batch_size = 20
data_list = DataLoader(datas, batch_size=global_batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
crit = torch.nn.MSELoss()



def train():
    model.train()
    random.shuffle(datas)
    count = 0
    loss_all = 0
    for sub_data in data_list:
        recent_data = sub_data.to(device)
        optimizer.zero_grad()
        output = model(recent_data)
        label = recent_data.y.to(device)
        output = output.to(device)
        output = output.t()
        output = output.squeeze(0)
        output = output.requires_grad_()
        loss = crit(output, label)
        loss.backward()
        optimizer.step()
        count += global_batch_size
        loss_all += loss
        sys.stdout.write(f'\rtrain_count:{count} loss_all:{loss_all} loss_average:{loss_all / count}')
        sys.stdout.flush()
    print()


def test():
    model.eval()
    #random.shuffle(circuits_test_info)
    right_count = 0
    all_count = 0
    error_all = 0
    for test_data in test_datas:
        all_count += 1
        test_data = test_data.to(device)
        get_logit = model(test_data).item()
        '''logit = 1 if 0.5 <= get_logit[1] <= 1 else 0'''
        '''logit = 0
        zero_count = 0
        one_count = 0
        for number in get_logit:
            if number == 1:
                one_count += 1
            else:
                zero_count += 1

        if one_count >= zero_count:
            logit = 1'''
        get_real_logit = test_data.y.item()
        error = abs(get_logit - get_real_logit)
        error_all += error
        sys.stdout.write(f'\rtest all_count:{all_count} right_count:{right_count} right:{get_real_logit} predict{get_logit} error:{error} average_error:{error_all / all_count}')
        sys.stdout.flush()
    print()
    row_data.append(error_all / all_count)
    xls_data.append(row_data)
    return 0


xls_data = [['epoch', 'average_error']]
row_data = []
for epoch in range(1000):
    print(f'epoch:{epoch + 1}')
    row_data.append(epoch + 1)
    train()
    '''print('origin:')
    test('circuit_data_in_noise_model_3w13700.xlsx')'''
    print('simulation_test:')
    test()
    row_data = []
    if (epoch + 1) % 10 == 0:
        write_to_excel(f'HGP_e1_delta_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
    print()
