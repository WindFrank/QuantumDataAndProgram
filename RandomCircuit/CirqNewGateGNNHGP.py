import sys

from RandomCircuit.CirqSimulateEnvironment import circuit_to_string, find_number, read_xls, write_to_excel
from torch_geometric.data import Data

import random
import torch

from torch_geometric.loader import DataLoader

import argparse

from HGP_SL.models import Model2

from RandomCircuit.Utils import read_xls

from xlsTest import write_to_excel

'''
寻完整门的方法，附带索引改变
'''


def find_gate(circuit_string, i):
    gate = ''
    while i < len(circuit_string) and circuit_string[i] != 'Q':
        gate += circuit_string[i]
        i += 1
    return gate, i


'''
trans_circuit_to_graph_from_zkd: 将中科大多行字符串电路转换为图数据结构
max_gate_number: 最大门数量
max_depth: 最大深度
'''


def trans_circuit_to_graph_from_zkd(circuit_code, width, max_depth, max_gate_number, fidelity=0.5):  # 将电路转换为图数据结构
    circuit_string = circuit_to_string(circuit_code)
    circuit_string = list(circuit_string)
    depths = [0 for _ in range(width)]
    pre = [0 for _ in range(width)]
    pre_gates = [[0 for _ in range(13)] for _ in range(width)]
    pre_gate_zero_one_hot = [0 for _ in range(13 * width)]
    pre_line = [[0 for _ in range(max_depth * 13)] for _ in range(width)]
    i = 0
    all_count = 1
    x = []
    edge_index = []
    gate_one_hot = [0 for _ in range(13)]
    qubit_one_hot = [0 for _ in range(width)]
    depth_one_hot = [0 for _ in range(max_depth + 1)]
    # 单比特前向结点门序列，13代表13种门
    one_qubit_pre_one_hot = [[0 for _ in range(13 * max_depth)] for _ in range(width)]
    # pre_node_one_hot = [0 for _ in range(max_gate_number)]
    sentence_struct = ['@', 'Y2M', 'Y2P', 'X2P', 'H', 'X2MTD', 'TX2M', 'X2MSD', 'XS', 'X2MT', 'SYX2M', 'SXX2M',
                       'ZX2MSD', 'TZX']
    sentence_struct_hot = [[0 for _ in range(len(sentence_struct))] for _ in range(width)]
    n_struct = ['@', 'X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'X2P', 'X2M', 'Y2M', 'Y2P', '@@', '@X', '@Y', '@Z', '@H',
                '@S', '@SD', '@T', '@TD', '@X2P', '@X2M', '@Y2M', '@Y2P', 'X@', 'XX', 'XY', 'XZ', 'XH', 'XS', 'XSD',
                'XT', 'XTD', 'XX2P', 'XX2M', 'XY2M', 'XY2P', 'Y@', 'YX', 'YY', 'YZ', 'YH', 'YS', 'YSD', 'YT', 'YTD',
                'YX2P', 'YX2M', 'YY2M', 'YY2P', 'Z@', 'ZX', 'ZY', 'ZZ', 'ZH', 'ZS', 'ZSD', 'ZT', 'ZTD', 'ZX2P', 'ZX2M',
                'ZY2M', 'ZY2P', 'H@', 'HX', 'HY', 'HZ', 'HH', 'HS', 'HSD', 'HT', 'HTD', 'HX2P', 'HX2M', 'HY2M', 'HY2P',
                'S@', 'SX', 'SY', 'SZ', 'SH', 'SS', 'SSD', 'ST', 'STD', 'SX2P', 'SX2M', 'SY2M', 'SY2P', 'SD@', 'SDX',
                'SDY', 'SDZ', 'SDH', 'SDS', 'SDSD', 'SDT', 'SDTD', 'SDX2P', 'SDX2M', 'SDY2M', 'SDY2P', 'T@', 'TX', 'TY',
                'TZ', 'TH', 'TS', 'TSD', 'TT', 'TTD', 'TX2P', 'TX2M', 'TY2M', 'TY2P', 'TD@', 'TDX', 'TDY', 'TDZ', 'TDH',
                'TDS', 'TDSD', 'TDT', 'TDTD', 'TDX2P', 'TDX2M', 'TDY2M', 'TDY2P', 'X2P@', 'X2PX', 'X2PY', 'X2PZ',
                'X2PH', 'X2PS', 'X2PSD', 'X2PT', 'X2PTD', 'X2PX2P', 'X2PX2M', 'X2PY2M', 'X2PY2P', 'X2M@', 'X2MX',
                'X2MY', 'X2MZ', 'X2MH', 'X2MS', 'X2MSD', 'X2MT', 'X2MTD', 'X2MX2P', 'X2MX2M', 'X2MY2M', 'X2MY2P',
                'Y2M@', 'Y2MX', 'Y2MY', 'Y2MZ', 'Y2MH', 'Y2MS', 'Y2MSD', 'Y2MT', 'Y2MTD', 'Y2MX2P', 'Y2MX2M',
                'Y2MY2M', 'Y2MY2P', 'Y2P@', 'Y2PX', 'Y2PY', 'Y2PZ', 'Y2PH', 'Y2PS', 'Y2PSD', 'Y2PT', 'Y2PTD',
                'Y2PX2P', 'Y2PX2M', 'Y2PY2M', 'Y2PY2P']
    struct_hot = [0 for _ in range(len(n_struct))]
    sentence_calculate = ['' for _ in range(9)]
    depth_one_hot[0] = 1
    node = one_qubit_pre_one_hot[0] + gate_one_hot + qubit_one_hot + depth_one_hot + sentence_struct_hot[0] + struct_hot
    x.append(node)
    while i < len(circuit_string):
        if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
            gate_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            qubit_one_hot_1 = [0 for _ in range(width)]
            qubit_one_hot_2 = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            #pre_node_one_hot_1 = [0 for _ in range(max_gate_number)]
            #pre_node_one_hot_2 = [0 for _ in range(max_gate_number)]
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
                recent_gate = first_one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 13]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit1][one_qubit_pre_index + 12] = 1
                    break
            for one_qubit_pre_index in range(max_depth):
                recent_gate = second_one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 13]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit1][one_qubit_pre_index + 12] = 1
                    break

            # 比特hot编码
            qubit_one_hot_1[qubit1] = 1
            qubit_one_hot_2[qubit2] = 1

            # recent_pre_line_1 = pre_line[qubit1]
            # recent_pre_line_2 = pre_line[qubit2]
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 13):
            #     if line_flag == 13:
            #         pre_line[qubit1][zero_location + 4] = 1
            #     if line_index % 13 == 0:
            #         line_flag = 0
            #         zero_location = line_index
            #     if recent_pre_line_1[line_index] == 1:
            #         line_flag = 0
            #     else:
            #         line_flag += 1
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 13):
            #     if line_flag == 13:
            #         pre_line[qubit2][zero_location + 4] = 1
            #     if line_index % 13 == 0:
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
            # for pre_index in pre:
            #     pre_node_one_hot_1[pre_index] = 1
            #     pre_node_one_hot_2[pre_index] = 1
            pre_gate_one_hot_1 = []
            pre_gate_one_hot_2 = []
            for pre_gate in pre_gates:
                pre_gate_one_hot_1 += pre_gate
                pre_gate_one_hot_2 += pre_gate

            # 敏感结构hot编码
            sentence_calculate[qubit1] += ' @'
            sentence_calculate[qubit2] += ' @'
            sentence_struct_hot[qubit1][0] = 1
            sentence_struct_hot[qubit2][0] = 1
            recent_struct_hot_1[0] = 1
            recent_struct_hot_2[0] = 1
            qubit1_sentence_calculate = sentence_calculate[qubit1].split(' ')
            if qubit1_sentence_calculate[0] == '':
                qubit1_sentence_calculate.remove('')
            qubit2_sentence_calculate = sentence_calculate[qubit2].split(' ')
            if qubit2_sentence_calculate[0] == '':
                qubit2_sentence_calculate.remove('')
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
            node1 = one_qubit_pre_one_hot[
                        qubit1] + gate_one_hot + qubit_one_hot_1 + depth_one_hot + sentence_struct_hot[qubit1] + recent_struct_hot_1
            node2 = one_qubit_pre_one_hot[
                        qubit2] + gate_one_hot + qubit_one_hot_2 + depth_one_hot + sentence_struct_hot[qubit2] + recent_struct_hot_2
            x.append(node1)
            x.append(node2)
        elif circuit_string[i] == 'M':
            break
        else:
            gate_one_hot = [0 for _ in range(13)]
            qubit_one_hot = [0 for _ in range(width)]
            depth_one_hot = [0 for _ in range(max_depth + 1)]
            pre_node_one_hot = [0 for _ in range(max_gate_number)]
            recent_struct_hot = [0 for _ in range(len(n_struct))]
            mapping = {
                'X': 0,
                'Y': 1,
                'Z': 2,
                'H': 3,
                'S': 4,
                'SD': 5,
                'T': 6,
                'TD': 7,
                'X2P': 8,
                'X2M': 9,
                'Y2M': 10,
                'Y2P': 11
            }
            temp_i = i
            [gate, i] = find_gate(circuit_string, i)
            index = mapping[gate]
            gate_one_hot[index] = 1
            [qubit, i_later] = find_number(circuit_string, i + 1)
            pre_gates[qubit] = gate_one_hot
            sentence_calculate[qubit] += ' ' + gate
            for sentence, index in zip(sentence_struct, range(len(sentence_struct))):
                if sentence in sentence_calculate[qubit]:
                    sentence_struct_hot[qubit][index] = 1

            # 单比特前向结点门序列嵌入
            one_qubit_hot = one_qubit_pre_one_hot[qubit]
            for one_qubit_pre_index in range(max_depth):
                recent_gate = one_qubit_hot[one_qubit_pre_index:one_qubit_pre_index + 13]
                if not any(recent_gate):
                    one_qubit_pre_one_hot[qubit][one_qubit_pre_index + index] = 1
                    break

            # recent_pre_line = pre_line[qubit]
            # line_flag = 0
            # zero_location = 0
            # for line_index in range(max_depth * 13):
            #     if line_flag == 13:
            #         pre_line[qubit][zero_location + index] = 1
            #     if line_index % 13 == 0:
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
            qubit_sentence_calculate = sentence_calculate[qubit].split(' ')
            len_qubit_sentence_calculate = len(qubit_sentence_calculate)
            if len_qubit_sentence_calculate > 1:
                pre_struct_index = n_struct.index(qubit_sentence_calculate[-2] + qubit_sentence_calculate[-1])
                recent_struct_hot[pre_struct_index] = 1
                # 前向结点敏感结构补充
                x[pre[qubit]][pre_struct_index] = 1

            edge_index.append([pre[qubit], all_count])
            pre[qubit] = all_count
            i = i_later
            node = one_qubit_pre_one_hot[qubit] + gate_one_hot + qubit_one_hot + depth_one_hot  + sentence_struct_hot[qubit] + recent_struct_hot
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
    if error_all / all_count <= 0.105:
        torch.save(model, f'Model/HGP_model_new_gate_e1_{epoch + 1}')
        print('模型存储成功！')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--batch_size', type=int, default=856, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--nhid', type=int, default=1064, help='hiddens size')
    parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
    parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--num_features', type=int, default=428,
                        help='num_features')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='num_classes')
    datas = []
    test_datas = []
    args = parser.parse_args()
    circuits_info = read_xls('TrainData/e3_13_gate/GNN_training_data_13_gate_simulate_e3_ALL.xlsx')
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
            write_to_excel(f'GNN_fidelity_new_gate_e3_{epoch + 1}.xlsx', 'sheet1', 'info', xls_data)
        # if (epoch + 1) % 50 == 0:
        #     lr -= 0.000002
        #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        print()
