import random
import sys
import torch
from CirqSimulateEnvironment import circuit_to_string, find_number, read_xls, write_to_excel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from layers import SAGPool
import argparse
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

'''
trans_circuit_to_graph_from_zkd: 将中科大多行字符串电路转换为图数据结构
'''


def trans_circuit_to_graph_from_zkd(circuit_code, width, max_depth, max_gate_number, fidelity):  # 将电路转换为图数据结构
    circuit_string = circuit_to_string(circuit_code)
    circuit_string = list(circuit_string)
    depths = [0 for _ in range(width)]
    pre = [0 for _ in range(width)]
    i = 0
    all_count = 1
    x = []
    edge_index = []
    gate_one_hot = [0, 0, 0, 0, 0]
    qubit_one_hot = [0 for _ in range(width)]
    depth_one_hot = [0 for _ in range(max_depth + 1)]
    pre_node_one_hot = [0 for _ in range(max_gate_number)]
    depth_one_hot[0] = 1
    pre_node_one_hot[0] = 1
    node = gate_one_hot + qubit_one_hot + depth_one_hot + pre_node_one_hot
    x.append(node)
    while i < len(circuit_string):
        if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
            gate_one_hot = [0, 0, 0, 0, 1]
            qubit_one_hot_1 = [0 for _ in range(width)]
            qubit_one_hot_2 = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            pre_node_one_hot_1 = [0 for _ in range(max_gate_number)]
            pre_node_one_hot_2 = [0 for _ in range(max_gate_number)]
            [qubit1, i_later] = find_number(circuit_string, i + 3)
            i = i_later + 1
            [qubit2, i_later] = find_number(circuit_string, i)
            qubit_one_hot_1[qubit1] = 1
            qubit_one_hot_2[qubit2] = 1
            temp_depth = max(depths[qubit1] + 1, depths[qubit2] + 1)
            depths[qubit1] = temp_depth
            depths[qubit2] = temp_depth
            depth_one_hot[temp_depth] = 1
            pre_node_one_hot_1[pre[qubit1]] = 1
            pre_node_one_hot_2[pre[qubit2]] = 1
            edge_index.append([pre[qubit1], all_count])
            pre[qubit1] = all_count
            all_count += 1
            edge_index.append([pre[qubit2], all_count])
            edge_index.append([all_count - 1, all_count])
            edge_index.append([all_count, all_count - 1])
            pre[qubit2] = all_count
            node1 = gate_one_hot + qubit_one_hot_1 + depth_one_hot + pre_node_one_hot_1
            node2 = gate_one_hot + qubit_one_hot_2 + depth_one_hot + pre_node_one_hot_2
            x.append(node1)
            x.append(node2)
            i = i_later
        elif circuit_string[i] == 'M':
            break
        else:
            gate_one_hot = [0, 0, 0, 0, 0]
            qubit_one_hot = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            pre_node_one_hot = [0 for _ in range(max_gate_number)]
            mapping = {
                'X': 0,
                'Y': 1,
                'Z': 2,
                'H': 3
            }
            index = mapping[circuit_string[i]]
            gate_one_hot[index] = 1
            [qubit, i_later] = find_number(circuit_string, i + 2)
            depths[qubit] += 1
            this_depth = depths[qubit]
            depth_one_hot[this_depth] = 1
            qubit_one_hot[qubit] = 1  # find_number已经进行过-1处理，不必再修正
            pre_node = pre[qubit]
            pre_node_one_hot[pre_node] = 1
            edge_index.append([pre[qubit], all_count])
            pre[qubit] = all_count
            i = i_later
            node = gate_one_hot + qubit_one_hot + depth_one_hot + pre_node_one_hot
            x.append(node)
        all_count += 1
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(fidelity, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index.t().contiguous())


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=152, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--num_features', type=int, default=76,
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
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

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
circuits_info = read_xls('GNN_training_data_7200.xlsx')
circuits_info.pop(0)
split_count = 0
for data_circuit in circuits_info:
    data_graph = trans_circuit_to_graph_from_zkd(data_circuit[0], 8, 13, 49, data_circuit[5])
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
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
        sys.stdout.write(f'\rtrain_count:{count} loss_all:{loss_all}')
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
        sys.stdout.write(f'\rtest all_count:{all_count} right_count:{right_count} right:{get_real_logit} predict:{get_logit} error:{error} average_error:{error_all / all_count}')
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
        write_to_excel(f'GNN_fidelity_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
    print()