import sys

from CirqSimulateEnvironment import circuit_to_string, find_number, read_xls, write_to_excel
from torch_geometric.data import Data

import random
import torch

from torch_geometric.loader import DataLoader

from RandomCircuit1 import trans_circuit_to_graph
import argparse
import torch.nn.functional as F

from HGP_SL.models import Model2

from Utils import read_xls

from xlsTest import write_to_excel

'''
trans_circuit_to_graph_from_zkd: 将中科大多行字符串电路转换为图数据结构
max_gate_number: 最大门数量
max_depth: 最大深度
'''


def trans_circuit_to_graph_from_zkd(circuit_code, width, max_depth, max_gate_number, fidelity):  # 将电路转换为图数据结构
    circuit_string = circuit_to_string(circuit_code)
    circuit_string = list(circuit_string)
    depths = [0 for _ in range(width)]
    pre = [0 for _ in range(width)]
    pre_gates = [[0 for _ in range(5)] for _ in range(width)]
    pre_gate_zero_one_hot = [0 for _ in range(5 * width)]
    #pre_line = [[0 for _ in range(max_depth * 5)] for _ in range(width)]
    i = 0
    all_count = 1
    x_qubit = [[] for _ in range(9)]
    x = []
    edge_index = []
    gate_one_hot = [0, 0, 0, 0, 0]
    qubit_one_hot = [0 for _ in range(width)]
    depth_one_hot = [0 for _ in range(max_depth + 1)]
    # 单比特前向结点门序列，5代表5种门
    one_qubit_pre_one_hot = [[0 for _ in range(5 * max_depth)] for _ in range(width)]
    pre_node_one_hot = [0 for _ in range(max_gate_number)]
    sentence_struct_hot = [[0 for _ in range(9)] for _ in range(width)]
    sentence_struct = ['@', 'HH', 'YH', 'XY', 'HXH', 'XHX', 'XZX', 'XZY', 'HYZ']
    n_struct = ['@', 'X', 'Y', 'Z', 'H', '@@', '@X', '@Y', '@Z', '@H', 'X@', 'XX', 'XY', 'XZ', 'XH', 'Y@', 'YX', 'YY',
                'YZ', 'YH', 'Z@', 'ZX', 'ZY', 'ZZ', 'ZH', 'H@', 'HX', 'HY', 'HZ', 'HH']
    struct_hot = [0 for _ in range(len(n_struct))]
    sentence_calculate = ['' for _ in range(9)]
    depth_one_hot[0] = 1
    node = one_qubit_pre_one_hot[0] + gate_one_hot + qubit_one_hot + depth_one_hot + sentence_struct_hot[0] + struct_hot
    x.append(node)
    while i < len(circuit_string):
        if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
            gate_one_hot = [0, 0, 0, 0, 1]
            qubit_one_hot_1 = [0 for _ in range(width)]
            qubit_one_hot_2 = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            # pre_node_one_hot_1 = [0 for _ in range(max_gate_number)]
            # pre_node_one_hot_2 = [0 for _ in range(max_gate_number)]
            recent_struct_hot_1 = [0 for _ in range(len(n_struct))]
            recent_struct_hot_2 = [0 for _ in range(len(n_struct))]
            recent_struct_hot_1[0] = 1
            recent_struct_hot_2[0] = 1

            [qubit1, i_later] = find_number(circuit_string, i + 3)
            i = i_later + 1
            pre_gates[qubit1] = gate_one_hot
            [qubit2, i_later] = find_number(circuit_string, i)
            i = i_later
            pre_gates[qubit2] = gate_one_hot

            # 单比特前向结点门序列向量：
            first_one_qubit_hot = one_qubit_pre_one_hot[qubit1]
            second_one_qubit_hot = one_qubit_pre_one_hot[qubit2]
            for one_qubit_pre_index in range(max_depth):
                recent_gate = first_one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 5]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit1][one_qubit_pre_index + 4] = 1
                    break
            for one_qubit_pre_index in range(max_depth):
                recent_gate = second_one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 5]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit1][one_qubit_pre_index + 4] = 1
                    break

            # 比特hot编码
            qubit_one_hot_1[qubit1] = 1
            qubit_one_hot_2[qubit2] = 1

            # recent_pre_line_1 = pre_line[qubit1]
            # recent_pre_line_2 = pre_line[qubit2]
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 5):
            #     if line_flag == 5:
            #         pre_line[qubit1][zero_location + 4] = 1
            #     if line_index % 5 == 0:
            #         line_flag = 0
            #         zero_location = line_index
            #     if recent_pre_line_1[line_index] == 1:
            #         line_flag = 0
            #     else:
            #         line_flag += 1
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 5):
            #     if line_flag == 5:
            #         pre_line[qubit2][zero_location + 4] = 1
            #     if line_index % 5 == 0:
            #         line_flag = 0
            #         zero_location = line_index
            #     if recent_pre_line_2[line_index] == 1:
            #         line_flag = 0
            #     else:
            #         line_flag += 1

            temp_depth = max(depths[qubit1] + 1, depths[qubit2] + 1)
            depths[qubit1] = temp_depth
            depths[qubit2] = temp_depth
            depth_one_hot[temp_depth] = 1
            # 全位向结点信息
            pre_gate_one_hot_1 = []
            pre_gate_one_hot_2 = []
            for pre_gate in pre_gates:
                pre_gate_one_hot_1 += pre_gate
                pre_gate_one_hot_2 += pre_gate

            # 敏感结构hot编码
            sentence_calculate[qubit1] += '@'
            sentence_calculate[qubit2] += '@'
            sentence_struct_hot[qubit1][0] = 1
            sentence_struct_hot[qubit2][0] = 1
            recent_struct_hot_1[0] = 1
            recent_struct_hot_2[0] = 1
            qubit1_sentence_calculate = sentence_calculate[qubit1]
            qubit2_sentence_calculate = sentence_calculate[qubit2]
            len_qubit1_sentence_calculate = len(qubit1_sentence_calculate)
            len_qubit2_sentence_calculate = len(qubit2_sentence_calculate)
            if len_qubit1_sentence_calculate > 1:
                pre_struct_index_1 = n_struct.index(qubit1_sentence_calculate[-2] + qubit1_sentence_calculate[-1])
                recent_struct_hot_1[pre_struct_index_1] = 1
                # 前向结点敏感结构补充
                x[pre[qubit1]][pre_struct_index_1] = 1
            if len_qubit2_sentence_calculate > 1:
                pre_struct_index_2 = n_struct.index(qubit2_sentence_calculate[-2] + qubit2_sentence_calculate[-1])
                recent_struct_hot_2[pre_struct_index_2] = 1
                x[pre[qubit2]][pre_struct_index_2] = 1
            edge_index.append([pre[qubit1], all_count])
            pre[qubit1] = all_count
            all_count += 1
            edge_index.append([pre[qubit2], all_count])
            edge_index.append([all_count - 1, all_count])
            edge_index.append([all_count, all_count - 1])
            pre[qubit2] = all_count
            node1 = one_qubit_pre_one_hot[qubit1] + gate_one_hot + qubit_one_hot_1 + depth_one_hot + sentence_struct_hot[qubit1] + recent_struct_hot_1
            node2 = one_qubit_pre_one_hot[qubit2] + gate_one_hot + qubit_one_hot_2 + depth_one_hot + sentence_struct_hot[qubit2] + recent_struct_hot_2
            x.append(node1)
            x.append(node2)
        elif circuit_string[i] == 'M':
            break
        else:
            gate_one_hot = [0, 0, 0, 0, 0]
            qubit_one_hot = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            pre_node_one_hot = [0 for _ in range(max_gate_number)]
            recent_struct_hot = [0 for _ in range(len(n_struct))]
            mapping = {
                'X': 0,
                'Y': 1,
                'Z': 2,
                'H': 3
            }
            index = mapping[circuit_string[i]]
            gate_one_hot[index] = 1
            [qubit, i_later] = find_number(circuit_string, i + 2)
            pre_gates[qubit] = gate_one_hot
            sentence_calculate[qubit] += circuit_string[i]
            for sentence, index in zip(sentence_struct, range(9)):
                if sentence in sentence_calculate[qubit]:
                    sentence_struct_hot[qubit][index] = 1

            # 单比特前向结点门序列嵌入
            one_qubit_hot = one_qubit_pre_one_hot[qubit]
            for one_qubit_pre_index in range(max_depth):
                recent_gate = one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 5]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit][one_qubit_pre_index + index] = 1
                    break

            # recent_pre_line = pre_line[qubit]
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 5):
            #     if line_flag == 5:
            #         pre_line[qubit][zero_location + index] = 1
            #     if line_index % 5 == 0:
            #         line_flag = 0
            #         zero_location = line_index
            #     if recent_pre_line[line_index] == 1:
            #         line_flag = 0
            #     else:
            #         line_flag += 1
            depths[qubit] += 1
            this_depth = depths[qubit]
            depth_one_hot[this_depth] = 1
            qubit_one_hot[qubit] = 1  # find_number已经进行过-1处理，不必再修正
            pre_gate_one_hot = []
            for pre_gate in pre_gates:
                pre_gate_one_hot += pre_gate
            qubit_sentence_calculate = sentence_calculate[qubit]
            len_qubit_sentence_calculate = len(qubit_sentence_calculate)
            if len_qubit_sentence_calculate > 1:
                pre_struct_index = n_struct.index(qubit_sentence_calculate[-2] + qubit_sentence_calculate[-1])
                recent_struct_hot[pre_struct_index] = 1
                # 前向结点敏感结构补充
                x[pre[qubit]][pre_struct_index] = 1

            edge_index.append([pre[qubit], all_count])
            pre[qubit] = all_count
            i = i_later
            node = one_qubit_pre_one_hot[qubit] + gate_one_hot + qubit_one_hot + depth_one_hot + sentence_struct_hot[qubit] + recent_struct_hot
            x.append(node)
        all_count += 1
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(fidelity, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    return Data(x=x, y=y, edge_index=edge_index.t().contiguous())


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
    # random.shuffle(circuits_test_info)
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
        sys.stdout.write(
            f'\rtest all_count:{all_count} right_count:{right_count} right:{get_real_logit} predict:{get_logit} error:{error} average_error:{error_all / all_count}')
        sys.stdout.flush()
        if all_count % 100 == 0 and all_count != 0:
            print()
    print()
    row_data.append(error_all / all_count)
    xls_data.append(row_data)
    if error_all / all_count <= 0.06:
        torch.save(model, f'Model/HGP_model{epoch + 1}')
        print('模型存储成功！')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=286, help='hiddens size')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--num_features', type=int, default=143,
                        help='num_features')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='num_classes')
    datas = []
    test_datas = []
    args = parser.parse_args()
    circuits_info = read_xls('TrainData/e3_5_gate/GNN_training_data_2w_five_gate_e3.xlsx_ALL.xlsx')
    circuits_info.pop(0)
    split_count = 0
    for data_circuit in circuits_info:
        data_graph = trans_circuit_to_graph_from_zkd(data_circuit[0], 8, 15, 52, data_circuit[5])
        if split_count % 5 == 0:
            test_datas.append(data_graph)
        else:
            datas.append(data_graph)
        split_count += 1
    circuits_info = None
    global_batch_size = 20
    data_list = DataLoader(datas, batch_size=global_batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model2(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00012, weight_decay=5e-4)
    crit = torch.nn.MSELoss()
    # lr = 0.00012

    xls_data = [['epoch', 'average_error']]
    row_data = []
    for epoch in range(1000):
        print(f'epoch:{epoch + 1}')
        row_data.append(epoch + 1)
        train()
        '''print('origin:')
        test('circuit_data_in_noise_model_3w13700.xlsx')'''
        print('simulation_test:')
        if (epoch + 1) % 5 == 0:
            test()
        row_data = []
        if (epoch + 1) % 10 == 0:
            write_to_excel(f'GNN_fidelity_5_gate_e3_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
        # if (epoch + 1) % 50 == 0:
        #     lr -= 0.000002
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        print()
