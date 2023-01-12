import math

import numpy as np
import numpy

import cirq

from RandomNoise import RandomNoise
from Utils import read_xls
from xlsTest import write_to_excel


# 获取平均变分距离
def get_bas(circuit, width, repetitions, quantum_result):
    simulation_result = get_cirq_run_result(circuit, width, repetitions)  # 这里的模拟结果就是二进制排序结果
    fidelity = get_variational_distance(quantum_result, simulation_result)
    return fidelity


# 获取变分距离底层函数
def get_variational_distance(vector1, vector2):
    distance = 0
    for (item1, item2) in zip(vector1, vector2):
        distance += 0.5 * abs(item1 - item2)
    return distance


def get_cirq_run_result(c_cirq, width, repetitions=12000):
    simulator = cirq.Simulator()
    result = simulator.run(c_cirq, repetitions=repetitions)
    all_result = []
    for i in range(width):
        data = result.data[f'q(0, {i})'].tolist()
        all_result.append(data)
    all_result = np.array(all_result).T
    indexes = get_qubit_vector_index(pow(2, width))
    result = {}
    for index in indexes:
        result[index] = 0
    for i in all_result:
        temp = ''
        for j in i:
            temp += str(j)
        result[temp] += 1
    value = list(result.values())
    sum_value = sum(value)
    for i in range(len(value)):
        value[i] = value[i] / sum_value
    return value


'''
get_state_simulate_result: Get the simulate result from statevector_simulator
It's return is a complex vector.
Remember: Don't add measurement gates.
'''


def get_state_simulate_result(c_cirq):
    simulator = cirq.Simulator()
    result = simulator.simulate(c_cirq)
    result = result.final_state_vector
    return result


'''
circuit_to_string: Return the print mode of multiple line circuit string.
'''


def circuit_to_string(circuit_string):
    circuit_string = circuit_string.replace('\n', '')
    circuit_string = circuit_string.replace(' ', '')
    return circuit_string


'''
find_number: Get the whole number in circuit string.
This method has let result minus 1.
'''


def find_number(circuit_string, index):
    i = index
    temp = 0
    while i < len(circuit_string) and '0' <= circuit_string[i] <= '9':
        temp = temp * 10
        temp += int(circuit_string[i])
        i = i + 1
    return [temp - 1, i]


'''
read_circuit_string_to_qiskit: Get the circuit object corresponding string.
You need to transport a param "width" which denotes the circuit's qubit number.
'''


def read_circuit_string_to_cirq(circuit_string, width):
    circuit_string = circuit_to_string(circuit_string)
    circuit_string = list(circuit_string)
    circuit = cirq.Circuit()
    qubits = []
    for i in range(width):
        qubits.append(cirq.GridQubit(0, i))
    i = 0
    while i < len(circuit_string):
        if circuit_string[i] == 'C' and circuit_string[i + 1] == 'Z':
            [qubit1, i_later] = find_number(circuit_string, i + 3)
            i = i_later + 1
            [qubit2, i_later] = find_number(circuit_string, i)
            circuit.append(cirq.CZ(qubits[qubit1], qubits[qubit2]))
            i = i_later
        elif circuit_string[i] == 'M':
            for qubit in qubits:
                circuit.append(cirq.measure(qubit))
            return circuit
        else:
            mapping = {
                'X': cirq.X,
                'Y': cirq.Y,
                'Z': cirq.Z,
                'H': cirq.H
            }
            operate = mapping[circuit_string[i]]
            [qubit, i_later] = find_number(circuit_string, i + 2)
            circuit.append(operate(qubits[qubit]))
            i = i_later
    return circuit


'''
电路模拟与真实运行数据，及其保真度数据生成
'''


def dataset_generate_by_certain(input_filename, output_filename, environment, repetitions=12000):
    # 随机生成电路，输入satisfy_circuit.xlsx表格
    circuits_info = read_xls(input_filename)
    circuits_info.pop(0)
    sub_dataset_generate_multi_sample(circuits_info, output_filename, environment, repetitions)


'''
获得量子比特索引
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
            temp = result_vector[order]
            if single_bit == '0':
                if isinstance(temp, numpy.complex64):
                    result[row, 0] += pow(abs(result_vector[order]), 2)
                else:
                    result[row, 0] += result_vector[order]
            else:
                if isinstance(temp, numpy.complex64):
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

    single_fidelity = sqrt1  # * pow(2, -1 * num_qubit)
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
baseline电路测试获得环境数据
"""


def baseline_noise_test(environment, repetitions=12000):
    baseline_circuit = '''
    X Q1
    Y Q2
    Z Q3
    H Q4
    M Q1 Q2 Q3 Q4
    '''
    m_index = baseline_circuit.index('M')
    circuit_no_measure = read_circuit_string_to_cirq(baseline_circuit[0:m_index], 4)
    result_right = get_probability_vector(get_state_simulate_result(circuit_no_measure))
    random_noise = RandomNoise()
    random_noise.set_concrete_p(environment[0], environment[1], environment[2])
    circuit_measure = read_circuit_string_to_cirq(baseline_circuit, 4)
    circuit_measure = circuit_measure.with_noise(random_noise)
    not_know_result = get_cirq_run_result(circuit_measure, 4, repetitions=repetitions)
    v_distance = get_variational_distance(result_right, not_know_result)
    return v_distance


"""
获取巴氏系数总方法：
"""


def get_bas_fidelity(circuit_code, width, repetitions, environment):
    m_index = circuit_code.index('M')
    circuit_no_measure = read_circuit_string_to_cirq(circuit_code[0:m_index], width)
    result_right = get_state_simulate_result(circuit_no_measure)
    circuit_measure = read_circuit_string_to_cirq(circuit_code, width)
    random_noise = RandomNoise()
    random_noise.set_concrete_p(environment[0], environment[1], environment[2])
    circuit_measure = circuit_measure.with_noise(random_noise)
    not_know_result = get_cirq_run_result(circuit_measure, width, repetitions)
    fidelity = get_fidelity(result_right, not_know_result)
    return fidelity


"""
数据采样底层实现
"""


def sub_dataset_generate_multi_sample(circuits_info, output_filename, environment, repetitions):
    data = [
        ['circuit_code', 'width', 'depth', 'gate_number', 'multi_gate_number', 'v_distance', 'e_noise', 'noise_x',
         'noise_y', 'noise_z']]
    control_gate = 0
    # 读取每一电路数据，获取电路，每个电路运行默认的12000shots，然后与正确向量使用保真度公式比对。
    for circuit_info, all_count in zip(circuits_info, range(len(circuits_info))):
        if all_count < control_gate:
            continue
        circuit_code = circuit_info[0]
        width = circuit_info[1]
        e_noise = baseline_noise_test(environment, repetitions)
        circuit = read_circuit_string_to_cirq(circuit_code, width)
        m_index = circuit_code.index('M')
        circuit_no_measure = read_circuit_string_to_cirq(circuit_code[0:m_index], width)
        result_right = get_probability_vector(get_state_simulate_result(circuit_no_measure))
        random_noise = RandomNoise()
        random_noise.set_concrete_p(environment[0], environment[1], environment[2])
        circuit = circuit.with_noise(random_noise)
        v_distance = get_bas(circuit, width, repetitions, result_right)
        if all_count == 19:
            print()
        print(f'C_Count:{all_count + 1} circuit:{circuit_to_string(circuit_code)} width:{width} depth:{circuit_info[2]}'
              f' gate_number:{circuit_info[3]} multi_gate_number:{circuit_info[4]} '
              f'v_distance:{v_distance} e_noise:{e_noise}')
        row_data = circuit_info
        row_data.append(v_distance)

        row_data.append(e_noise)
        row_data.append(environment[0])
        row_data.append(environment[1])
        row_data.append(environment[2])
        data.append(row_data)
        if all_count % 100 == 0 and all_count != 0:
            write_to_excel(f'{output_filename}_{all_count}.xlsx', 'sheet1', 'info', data)
    write_to_excel(f'{output_filename}_ALL.xlsx', 'sheet1', 'info', data)


if __name__ == '__main__':
    # print(isinstance(.0j, complex))
    dataset_generate_by_certain('D:\\workspace\\QuantumBase\\RandomCircuit\\TrainData\\origin_2w_five_gate_circuit.xlsx', 'SANER_GNN_training_data_2w_five_gate_e2.xlsx', [0.05, 0.03, 0.02], 1000)
