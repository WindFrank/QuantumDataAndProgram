import random
import cirq
import argparse
import time

from util import create_Uf_matrix, Cirq_Custom
from bernstein_vazirani_classical import test_f_4, test_f_5, test_f_6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",dest="n", type=int, required=True)
    parser.add_argument("-f",dest="f", type=create_lambda_with_globals, required=True)
    parser.add_argument("-t",dest="t", type=int, default=10)
    args = parser.parse_args()

    start = time.time()

    # Choose qubits to use.
    input_qubits = [cirq.GridQubit(i, 0) for i in range(args.n)]
    output_qubit = cirq.GridQubit(args.n, 0)

    # Create oracle
    Uf = Cirq_Custom(args.n+1, create_Uf_matrix(args.f, args.n), name="Uf")

    # Create BV circuit
    circuit = make_bernstein_vazirani_circuit(input_qubits, output_qubit, Uf)
    print('Circuit:')
    print(circuit)

    # Sample from the circuit a couple times.
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=args.t)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    print('Sampled results:\n{}'.format(frequencies))

    end = time.time()
    print("Execution time: ", end - start)

def create_lambda_with_globals(s):
    return eval(s, globals())

def make_bernstein_vazirani_circuit(input_qubits, output_qubit, oracle):
    """Creates BV circuit"""
    c = cirq.Circuit()
    # Initialize qubits.
    c.append([
        cirq.X(output_qubit),
        cirq.H(output_qubit),
        cirq.H.on_each(*input_qubits),
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    # Query oracle.
    c.append(oracle(*(input_qubits + [output_qubit])))
    # Measure in X basis.
    c.append([
        cirq.H.on_each(*input_qubits),
        cirq.measure(*input_qubits, key='result')
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

if __name__ == '__main__':
    main()