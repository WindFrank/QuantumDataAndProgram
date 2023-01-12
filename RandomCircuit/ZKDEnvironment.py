import math

import cirq
import numpy as np
import numpy
#from Challenge_Account import *

from RandomCircuit.CirqEnvironmentWithNewGate import read_circuit_string_to_cirq
from RandomCircuit.Utils import read_xls
from xlsTest import write_to_excel


'''
circuit_to_string: Return the print mode of multiple line circuit string.
'''


def circuit_to_string(circuit_string, star_wash=True):
    circuit_string = circuit_string.replace('\n', '')
    circuit_string = circuit_string.replace(' ', '')
    if star_wash:
        circuit_string = circuit_string.replace('*', '')
    return circuit_string


'''
电路模拟与真实运行数据，及其保真度数据生成
'''


def dataset_generate_by_certain(input_filename, output_filename):
    # 随机生成电路，输入satisfy_circuit.xlsx表格
    circuits_info = read_xls(input_filename)
    circuits_info.pop(0)
    sub_dataset_generate_multi_sample(circuits_info, output_filename)


'''
获得量子比特索引，传入获得索引向量的长度
'''


def get_qubit_vector_index(vector_number):
    result = []
    num_qubit = int(math.log2(vector_number))
    for i in range(0, vector_number):
        bin_temp_number_list = list(bin(i))
        bin_temp_number_list.reverse()
        index_sub_result = ''
        for j in range(num_qubit):
            if bin_temp_number_list[j] != 'b':
                index_sub_result = bin_temp_number_list[j] + index_sub_result
            else:
                break
        while len(index_sub_result) < num_qubit:
            index_sub_result = '0' + index_sub_result
        result.append(index_sub_result)
    return result


"""
由结果向量向每位量子态转化
"""


def get_qubit_formal(result_vector):
    num_qubit = int(math.log2(len(result_vector)))
    result = np.zeros([num_qubit, 2])
    index_list = get_qubit_vector_index(len(result_vector))
    for qubit_index, order in zip(index_list, range(len(result_vector))):
        qubit_index = list(qubit_index)
        for single_bit, row in zip(qubit_index, range(num_qubit)):
            if single_bit == '0':
                if isinstance(result_vector[order], complex):
                    result[row, 0] += pow(abs(result_vector[order]), 2)
                else:
                    result[row, 0] += result_vector[order]
            else:
                if isinstance(result_vector[order], complex):
                    result[row, 1] += pow(abs(result_vector[order]), 2)
                else:
                    result[row, 1] += result_vector[order]
    return np.sqrt(result)


def get_fidelity_by_single_qubit(right_qubit_state, not_know_qubit_state):
    right_qubit_state = np.array([right_qubit_state])
    not_know_qubit_state = np.array([not_know_qubit_state])
    # density = np.dot(right_qubit_state.T, right_qubit_state)
    # n_density = np.dot(not_know_qubit_state.T, not_know_qubit_state)
    # multi1 = np.dot(np.sqrt(density), n_density)
    # multi2 = np.dot(multi1, np.sqrt(density))
    # sqrt1 = np.sqrt(multi2)

    multi1 = np.dot(right_qubit_state, not_know_qubit_state.T)
    multi2 = np.dot(not_know_qubit_state, right_qubit_state.T)
    sqrt1 = math.sqrt(multi1 * multi2)

    single_fidelity = sqrt1 # * pow(2, -1 * num_qubit)
    # trace = np.trace(np.dot(density, n_density))
    # plus = 2 * np.sqrt(np.linalg.det(density) * np.linalg.det(n_density))
    # single_fidelity = (trace + plus)  # * pow(2, -1 * num_qubit)
    return single_fidelity


'''
get_probability_vector: 转复数向量
'''


def get_probability_vector(result_vector):
    temp = result_vector[0]
    if (not isinstance(temp, complex)) and (not isinstance(temp, numpy.complex64)):
        raise Exception("This method can only handle complex vector.")
    result_probability = []
    for i in range(len(result_vector)):
        result_probability.append(pow(abs(result_vector[i]), 2))
    return result_probability


"""
计算保真度
"""


def get_fidelity(result_right, not_know_result):
    result_vector = get_probability_vector(result_right)
    fidelity = 0
    for right_item, not_know_item in zip(result_vector, not_know_result):
        fidelity += math.sqrt(right_item * not_know_item)
    fidelity = pow(fidelity, 2)
    return fidelity


"""
适用于比较中科大量子云的cirq状态向量获取
"""


def get_state_simulate_result_compare_for_ZKDCloud(c_cirq, vector_number):
    simulator = cirq.Simulator()
    result = simulator.simulate(c_cirq)
    result = result.final_state_vector
    clockwise = get_qubit_vector_index(vector_number)
    counterclockwise = []
    for index in clockwise:
        counterclockwise.append(''.join(reversed(index)))
    recent_json = dict(zip(counterclockwise, result))
    true_result = []
    for key_index in sorted(counterclockwise):
        true_result.append(recent_json[key_index])
    return true_result


"""
baseline电路测试获得环境数据
"""


def baseline_noise_test():
    baseline_circuit = '''
    X Q1
    Y Q2
    Z Q3
    H Q4
    M Q1 Q2 Q3 Q4
    '''
    m_index = baseline_circuit.rindex('M')
    # 这里调用的是cirqSimulateEnvironment的读电路函数，如果要13种读电路，用withnewgate的。
    circuit_no_measure = read_circuit_string_to_cirq(baseline_circuit[0:m_index], 4)
    result_right = get_state_simulate_result_compare_for_ZKDCloud(circuit_no_measure, 16)  # 输入2的width次方
    not_know_result = submit_circuit(baseline_circuit, 'baseline')
    fidelity = get_fidelity(result_right, not_know_result)
    return fidelity


"""
数据采样底层实现
"""


def sub_dataset_generate_multi_sample(circuits_info, output_filename):
    data = [
        ['circuit_code', 'width', 'depth', 'gate_number', 'multi_gate_number', 'fidelity', 'e_noise']]
    # 读取每一电路数据，获取电路，每个电路运行默认的12000shots，然后与正确向量使用保真度公式比对。
    control_gate = 374
    for circuit_info, all_count in zip(circuits_info, range(len(circuits_info))):
        if all_count < control_gate:
            continue
        circuit_code = circuit_info[0]
        e_noise = baseline_noise_test()
        m_index = circuit_code.rindex('M')
        # 这里调用的是cirqSimulateEnvironment的读电路函数，如果要13种读电路，用withnewgate的。
        circuit_no_measure = read_circuit_string_to_cirq(circuit_code[0:m_index], circuit_info[1])  # 把M测量门去掉再运行。
        result_right = get_state_simulate_result_compare_for_ZKDCloud(circuit_no_measure, pow(2, int(circuit_info[1])))  # 输入2的width次方
        not_know_result = submit_circuit(circuit_code, str(all_count))
        fidelity = get_fidelity(result_right, not_know_result)
        print(f'C_Count:{all_count + 1} circuit:{circuit_to_string(circuit_code)} width:{circuit_info[1]} depth:{circuit_info[2]}'
              f' gate_number:{circuit_info[3]} multi_gate_number:{circuit_info[4]} '
              f'fidelity:{fidelity} e_noise:{e_noise}')
        row_data = circuit_info
        row_data.append(fidelity)
        row_data.append(e_noise)
        data.append(row_data)
        write_to_excel(f'{output_filename}_{all_count}.xlsx', 'sheet1', 'info', data)
    write_to_excel(f'{output_filename}_ALL.xlsx', 'sheet1', 'info', data)


"""
读电路方法，需要电路码
"""


def read_circuit_code_by_ZKD(circuit_code):
    c = '''
    '''
    codes = list(map(int, circuit_code))
    qubits = []
    i = 0
    while i < len(codes):
        if i == 0:
            for qubit_index in range(codes[i]):
                qubits.append('Q' + str(qubit_index + 1))
        else:
            gate_multi_number = codes[i]  # 读门位数类型
            if gate_multi_number == 1:
                i = i + 1
                gate_index = codes[i]  # 读门类型

                i = i + 1
                qubit = qubits[codes[i]]  # 读量子位
                gates = ['X', 'Y', 'Z', 'H', 'S', 'SD', 'T', 'TD', 'X2P', 'X2M', 'Y2M', 'Y2P']
                c += f'''{gates[gate_index]} {qubit}
                '''
            elif gate_multi_number == 2:
                i = i + 1
                gate_index = codes[i]  # 读门类型
                i = i + 1
                qubit1 = qubits[codes[i]]  # 读量子位
                i = i + 1
                qubit2 = qubits[codes[i]]  # 读量子位
                gates = ['CZ']
                c += f'''{gates[gate_index]} {qubit1} {qubit2}
                '''
            else:
                raise Exception('The circuit code is error.')
        i = i + 1
    print(c)
    return c + '\n'


# 读电路方法，不需要电路码
def read_zkd_circuit(c_string):
    c_string = circuit_to_string(c_string)
    c = cirq.Circuit()
    # 门类型句柄
    gate_type_mapping = {
        'X': cirq.X,
        'Y': cirq.Y,
        'Z': cirq.Z,
        'H': cirq.H,
        'I': cirq.I,
        'M': cirq.measure
    }
    i = 0
    code_length = len(c_string)
    # 取电路字符串的最后一个值作为电路比特数目
    qubit_number = int(c_string[code_length - 1])
    qubits = [cirq.GridQubit(0, i) for i in range(qubit_number)]
    while i < len(c_string):
        if c_string[i] == 'C' and c_string[i + 1] == 'Z':
            gate_type = 'CZ'
            i += 3  # 索引后移到数字上
            # 返回数字和索引再后移的位置
            [number, i] = find_number(c_string, i)
            recent_qubit_1 = number - 1  # qubit位置从0开始，-1处理。这里获得第一个数字
            i += 1  # 索引向后移一位越过Q到数字
            [number, i] = find_number(c_string, i)
            recent_qubit_2 = number - 1
            c.append(cirq.CZ(qubits[recent_qubit_1], qubits[recent_qubit_2]))
        else:  # 否则为简单门，如果表明不插入的值为False，那么就不插入随机旋转门
            # 待添加门
            gate_set = []
            # 处理单比特门
            gate_type = c_string[i]
            i += 2  # 索引后移到数字上
            # 返回数字和索引再后移的位置
            [number, i] = find_number(c_string, i)
            recent_qubit = number - 1  # qubit位置从0开始，-1处理。
            gate_simple = gate_type_mapping[gate_type]
            gate_set.append(gate_simple)
            for gate in gate_set:
                c.append(gate(qubits[recent_qubit]))
    return c


# 获得字符串中连续数字的方法，防止十位数以上的输入出现问题
def find_number(circuit_string, index):
    i = index
    temp = 0
    while i < len(circuit_string) and '0' <= circuit_string[i] <= '9':
        temp = temp * 10
        temp += int(circuit_string[i])
        i = i + 1
    return [temp, i]


'''
真实环境运行提交方法
'''


def submit_circuit(circuit, i):
    username = 'a0610328'
    password = 'tfGtfb5x'
    account = Account(username, password, False)
    if account.login:
        exp_id, n_measure = account.submit_job(circuit, exp_name='Random_Generate_one_0' + i)
        if isinstance(exp_id, int):
            result = account.query_experiment_result(exp_id, n_measure, max_wait_time=1)
            # 如果返回值为字典形式，实验结果读取成功，您可以基于result做下一步计算
            if isinstance(result, dict):
                order = sorted(result)
                result_value = []
                for k in order:
                    result_value.append(result[k])
                return result_value
            #如果返回值实验结果没有运行出来，您可以继续查询或等待
            elif '实验结果没有运行出来' in result:
                result = account.query_experiment_result(exp_id, n_measure, max_wait_time=30)
                if isinstance(result, dict):
                    order = sorted(result)
                    result_value = []
                    for k in order:
                        result_value.append(result[k])
                    return result_value
                elif '实验结果没有运行出来' in result:
                    print("机器估计已经进入校准")
                    result = account.query_experiment_result(exp_id, n_measure, max_wait_time=3600)
                    order = sorted(result)
                    result_value = []
                    for k in order:
                        result_value.append(result[k])
                    return result_value
            else:
                print(result)
        else:
            print(exp_id)


if __name__ == '__main__':
    dataset_generate_by_certain('OtherXlsx/zkd_circuit_each_all_struct_1029_use_for_ZKDCloud.xlsx', 'ZKD_real_environment_training_data_13_gate')
