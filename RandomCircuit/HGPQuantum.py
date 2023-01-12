import sys

import random
import torch

from torch_geometric.loader import DataLoader

from RandomCircuit1 import trans_circuit_to_graph
import argparse
import torch.nn.functional as F

from HGP_SL.models import Model

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
parser.add_argument('--num_classes', type=int, default=2,
                    help='num_classes')


datas = []
test_datas = []
args = parser.parse_args()
circuits_info = read_xls('e2_30000.xlsx')
circuits_info.pop(0)
split_count = 0
for data_circuit in circuits_info:
    data_graph = trans_circuit_to_graph(data_circuit[0], data_circuit[6])
    if split_count % 5 == 0:
        test_datas.append(data_graph)
    else:
        datas.append(data_graph)
    split_count += 1
circuits_info = None
global_batch_size = 20
data_list = DataLoader(datas, batch_size=global_batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(args).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)
crit = torch.nn.BCELoss()



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
        output = output.requires_grad_()
        loss = F.nll_loss(output, label)
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
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for test_data in test_datas:
        all_count += 1
        test_data = test_data.to(device)
        temp1 = model(test_data)
        temp2 = temp1.max(1)
        temp3 = temp2[1]
        get_logit = temp3.tolist()[0]
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
        real_logit = 1 if get_real_logit >= 0.5 else 0
        if real_logit == get_logit:
            if real_logit == 1:
                tp += 1
            else:
                tn += 1
            right_count += 1
        elif real_logit == 1:  # 正例预测错误
            fn += 1
        else:  # 负例预测错误
            fp += 1
        sys.stdout.write(f'\rtest all_count:{all_count} right_count:{right_count} right_rate:{right_count / all_count}')
        sys.stdout.flush()
    print()
    p = r = f1 = 0
    if (tp + fp) != 0:
        p = tp / (tp + fp)  # 精确率
    if (tp + fn) != 0:
        r = tp / (tp + fn)  # 召回率
    if (p + r) != 0:
        f1 = 2 * p * r / (p + r)
    print(f'acc:{(tp + tn) / (tp + tn + fp + fn) } tp:{tp} tn:{tn} fp:{fp} fn:{fn} p:{p} r:{r} F1:{f1}')
    row_data.append(p)
    row_data.append(r)
    row_data.append(f1)
    xls_data.append(row_data)
    return 0


xls_data = [['epoch', 'acc', 'recall', 'f1']]
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
        write_to_excel(f'HGP_e2_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
    print()
