import random

import cirq
import torch

from RandomCircuit.CirqEnvironmentWithNewGate import read_circuit_string_to_cirq, get_bas
from RandomCircuit.CirqNewGateGNNHGP import trans_circuit_to_graph_from_zkd
from RandomCircuit.CirqSimulateEnvironment import circuit_to_string, find_number
from RandomCircuit.ZKDCircuiteGenerator import random_circuit_get
from RandomCircuit.ZKDGNNHGP import find_gate


def m_remove(circuit_string):
    index = circuit_string.rindex('M')
    return circuit_string[0:index]


def m_add(width):
    result = ''
    for i in range(width):
        result += f'\nM Q{i + 1}'
    return result


def operate_random_insert(initial_circuit, select_operate_gate, m_add_code, width):
    cut_line = []
    circuit_string = circuit_to_string(initial_circuit)
    qubits = []
    for i in range(width):
        qubits.append(cirq.GridQubit(0, i))
    i = 0
    circuit = ''
    while i < len(circuit_string):
        if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
            [qubit1, i_later] = find_number(circuit_string, i + 3)
            i = i_later + 1
            [qubit2, i_later] = find_number(circuit_string, i)
            cut_line.append(f'\nCZ Q{qubit1 + 1} Q{qubit2 + 1}')
            i = i_later
        elif circuit_string[i] == 'M':
            raise Exception('别往里传带测量门的电路呀')
        else:
            [operate, i] = find_gate(circuit_string, i)
            [qubit, i_later] = find_number(circuit_string, i + 1)
            cut_line.append(f'\n{operate} Q{qubit + 1}')
            i = i_later
    insert_index = random.randint(0, len(cut_line))
    cut_line.insert(insert_index, select_operate_gate)
    for element in cut_line:
        circuit += element
    return circuit + m_add_code


def sentence_circuit_find(initial_width, initial_depth, repetitions, max_width, max_depth, max_gate_number,
                          component_number, environment, gates):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = torch.load('D:\\workspace\\QuantumBase\\RandomCircuit\\Model\\e1_13_gate\\HGP_model_new_gate_e1_15').to(device)
    # model.eval()
    depth = initial_depth
    width = initial_width
    circuit = random_circuit_get(width, depth, 1)[0][0]
    circuit_no_measure = m_remove(circuit)

    # 下面的方法，必须保证fuzzing生成的随机电路，不能超出最大门数量和最大深度。fidelity的输入值任意
    b_coefficient = get_bas(circuit, initial_width, environment, 1000)
    # torch_data = trans_circuit_to_graph_from_zkd(circuit, 8, 15, 52).to(device)
    # b_coefficient = model(torch_data).item()
    # if b_coefficient < 0.3:
    #     return sentence_circuit_find(initial_width, initial_depth, repetitions, max_width, max_depth, max_gate_number,
    #                                  component_number, environment, gates)
    final_operates = []
    # 开始fuzzing
    while b_coefficient >= 0.3:
        i = 0
        operates_gate = []
        circuits_fuzz = []
        temps_b_coefficient = []
        temps_width = []
        operate_gate = ''
        j = 0
        #qubit_choose = random.randint(1, width + 1)
        qubit_choose = random.randint(1, width)
        qubit2_choose = -1
        # if qubit_choose == width + 1:
        #     temp_width += 1
        while j < component_number:
            gate_choose = random.randint(0, len(gates) - 1)
            #gate_choose = j
            if gate_choose == 4:
                qubit_choose = random.randint(1, width - 1)
                qubit2_choose = qubit_choose + 1
                operate_gate += f'\nCZ Q{qubit_choose} Q{qubit2_choose}'
            else:
                qubit_choose = random.randint(1, width)
                operate_gate += f'\n{gates[gate_choose]} Q{qubit_choose}'
                # if qubit2_choose != -1:
                #     qubit_random = qubit_choose if random.randint(0, 1) == 0 else qubit2_choose
                #     operate_gate += f'\n{gates[gate_choose]} Q{qubit_random}'
                # else:
                #     operate_gate += f'\n{gates[gate_choose]} Q{qubit_choose}'
            if operate_gate in operates_gate:
                continue
            circuit_fuzz = operate_random_insert(circuit_no_measure, operate_gate, m_add(width), width)
            # circuit_fuzz_torch = trans_circuit_to_graph_from_zkd(circuit_fuzz, 8, 15, 52, 0).to(device)
            try:
                temp_b_coefficient = get_bas(circuit_fuzz, width, environment, 1000)
                # temp_b_coefficient = model(circuit_fuzz_torch).item()
            except Exception:
                return sentence_circuit_find(initial_width, initial_depth, repetitions, max_width, max_depth,
                                             max_gate_number, component_number, environment, gates)
            temps_b_coefficient.append(temp_b_coefficient)
            operates_gate.append(operate_gate)
            circuits_fuzz.append(circuit_fuzz)
            j += 1
        min_index = temps_b_coefficient.index(min(temps_b_coefficient))
        circuit = circuits_fuzz[min_index]
        if temps_b_coefficient[min_index] > b_coefficient:
            # print(f'continue {temps_b_coefficient[min_index]}')
            circuits_fuzz = []
            continue
        # b_coefficient = temps_b_coefficient[min_index]
        final_operates.append(operates_gate[min_index])
        # circuit_no_measure = m_remove(circuit)
        # print(f'circuit: {circuit_to_string(circuit)} b_coefficient:{b_coefficient}')
        cirq_circuit = read_circuit_string_to_cirq(circuit, width)
        print(cirq_circuit)
        return circuit, final_operates


'''
返回一系列敏感结构
'''


def sentence_struct_find(circuit_number, component_number, environment, gates, filename):
    result = {}
    j = 0
    error = 0
    while j < circuit_number:
        sentence_circuit, final_operates = sentence_circuit_find(3, 3, 1000, 8, 15, 52, component_number, environment, gates)
        # if sentence_circuit is None:
        #     if error == 5:
        #         print("出现不可查找的未知错误")
        #         return result
        #     else:
        #         error += 1
        #         continue
        circuit = circuit_to_string(sentence_circuit)
        circuit_string = list(circuit)
        i = 0
        qubit_calculate = [[] for _ in range(8)]
        while i < len(circuit_string):
            if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
                [qubit1, i_later] = find_number(circuit_string, i + 3)
                i = i_later + 1
                [qubit2, i_later] = find_number(circuit_string, i)
                qubit_calculate[qubit1].append('@')
                qubit_calculate[qubit2].append('@')
                i = i_later
            elif circuit_string[i] == 'M':
                break
            else:
                [operate, i] = find_gate(circuit_string, i)
                [qubit, i_later] = find_number(circuit_string, i + 1)
                qubit_calculate[qubit].append(operate)
                i = i_later
        for array in qubit_calculate:
            for sub_component_number in [1, 2, 3]:
                array_index = 0
                while array_index + sub_component_number - 1 < len(array):
                    sub_result = ''
                    small_loop = array_index
                    while small_loop < array_index + sub_component_number:
                        sub_result += array[small_loop]
                        small_loop += 1
                    all_keys = result.keys()
                    if sub_result not in all_keys:
                        result[sub_result] = 1
                    else:
                        result[sub_result] += 1
                    array_index += 1
        j += 1
        return_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return_result.append(str(j))
        print(return_result)
        data_txt = open(filename + ".txt", 'w+')
        print(return_result, file=data_txt)
        print(f"Count:{j}")
        data_txt.close()
    return result


def sentence_choose_find(circuit_number, component_number, environment, gates):
    result = {}
    j = 0
    error = 0
    while j < circuit_number:
        sentence_circuit, final_operates = sentence_circuit_find(3, 3, 1000, 8, 15, 52, component_number, environment, gates)
        if sentence_circuit is None:
            if error == 5:
                print("出现不可查找的未知错误")
                return result
            else:
                error += 1
                continue
        for operate in final_operates:
            circuit = circuit_to_string(operate)
            circuit_string = list(circuit)
            i = 0
            sub_operate = ''
            while i < len(circuit_string):
                if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
                    [qubit1, i_later] = find_number(circuit_string, i + 3)
                    i = i_later + 1
                    [qubit2, i_later] = find_number(circuit_string, i)
                    sub_operate += '@'
                    i = i_later
                elif circuit_string[i] == 'M':
                    break
                else:
                    operate = circuit_string[i]
                    [qubit, i_later] = find_number(circuit_string, i + 2)
                    sub_operate += operate
                    i = i_later
            all_keys = result.keys()
            if sub_operate not in all_keys:
                result[sub_operate] = 1
            else:
                result[sub_operate] += 1
    return result


if __name__ == '__main__':
    print(sentence_struct_find(1000, 5, [0.033, 0.033, 0.034], ['X', 'Y', 'Z', 'H', 'CZ', 'S', 'SD', 'T', 'TD', 'X2P', 'X2M', 'Y2M', 'Y2P'], 'Sentence_13_gate_SANER_e1_revise'))
    # print(sentence_struct_find(1000, 2, [0.033, 0.033, 0.034], ['X', 'Y', 'Z', 'H', 'CZ'], 'Sentence_5_gate_e1'))
