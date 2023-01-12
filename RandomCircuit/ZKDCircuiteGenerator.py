import sys

import cirq

import random

from xlsTest import write_to_excel

'''
随机电路生成——适用于中科大的版本
该电路不再使用电路码
'''


def random_circuit_generate_no_gate_number(pre_width, pre_depth):
    width = pre_width
    depth = pre_depth
    temp_gate_number = 0
    temp_multi_gate_number = 0
    depths = [0 for i in range(width)]
    judgement = 0
    gate_type = []
    circuit = '''
    '''
    while judgement == 0:
        if max(depths) >= depth:
            judge = random.random()
            if judge > 0.5:
                break
        gate_type = [1 for _ in range(12)]
        gate_type.append(2)
            #gate_type.append(3)
        random_type = gate_type[random.randint(0, len(gate_type) - 1)]
        gates = []
        if random_type == 1:
            qubit = random.randint(1, width)
            gates = ['X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'X2P', 'X2M', 'Y2M', 'Y2P']
            # gates = ['X', 'Y', 'Z', 'H']
            gate_choose = random.randint(0, len(gates) - 1)
            final_gate = gates[gate_choose]
            circuit += f'''{final_gate} Q{qubit}
            '''
            depths[qubit - 1] += 1
        elif random_type == 2 and width >= 2:
            qubit1 = random.randint(1, width)
            qubit2 = qubit1 + 1
            if qubit2 > width:
                qubit2 = qubit1 - 1
            together_depth = max(depths[qubit1 - 1], depths[qubit2 - 1])
            if together_depth == depth:
                continue
            final_gate = 'CZ'
            depths[qubit1 - 1] = together_depth + 1
            depths[qubit2 - 1] = together_depth + 1
            circuit += f'''{final_gate} Q{qubit1} Q{qubit2}
            '''
            temp_multi_gate_number += 1
        temp_gate_number += 1
    # 为所有的位添加测量门
    circuit += '''M '''
    for i in range(width):
        circuit += f''' Q{i + 1}'''
    depth = max(depths)
    for i in depths:
        if i == 0:
            return random_circuit_generate_no_gate_number(pre_width, pre_depth)
    return circuit, width, depth, temp_gate_number, temp_multi_gate_number


'''
批量获取随机电路
'''


def random_circuit_get(width, depth, count_max):
    count = 0
    sub_data = []
    while count < count_max:
        circuit, width2, depth2, gate_number2, multi_gate_number2 = random_circuit_generate_no_gate_number(width, depth)
        row_data = [circuit, width2, depth2, gate_number2, multi_gate_number2]
        sub_data.append(row_data)
        count += 1
        print()

        sys.stdout.write(
            f'\rCount:{count} width:{width2} depth:{depth2} gate_number:{gate_number2}'
            f' multi_gate_number:{multi_gate_number2}')
        sys.stdout.flush()

    print()
    return sub_data


if __name__ == '__main__':
    data = [['circuit_code', 'width', 'depth', 'gate_number', 'multi_gate_number']]
    for i in range(6):
        for j in range(6):
            data = data + random_circuit_get(3 + i, 3 + j, 1000)
    write_to_excel(f'zkd_circuit_each_all_struct_36000_use_for_Scale_Show.xlsx', 'sheet1', 'info', data)
