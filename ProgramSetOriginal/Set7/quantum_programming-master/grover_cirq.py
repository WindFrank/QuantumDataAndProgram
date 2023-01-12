import random
import cirq
import argparse
import time
import math

from util import create_Z0_matrix, create_Zf_matrix, Cirq_Custom
from grover_classical import test_only_one, test_0, test_1, test_01, test_000, test_110, test_1111, test_0010, test_10010, test_10111

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",dest="n", type=int, required=True)
    parser.add_argument("-f",dest="f", type=create_lambda_with_globals, required=True)
    parser.add_argument("-t",dest="t", type=int, default=10)
    args = parser.parse_args()

    start = time.time()

    # Choose qubits to use.
    qubits = [cirq.GridQubit(i, 0) for i in range(args.n)]

    # Create reflection operations
    Z0 = Cirq_Custom(args.n, create_Z0_matrix(args.n), name="Z0")
    Zf = Cirq_Custom(args.n, create_Zf_matrix(args.f, args.n), name="Zf")

    # Create DJ circuit
    circuit = make_grover_circuit(qubits, Z0, Zf)
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

def make_grover_circuit(qubits, Z0, Zf):
    """Creates grovers circuit"""
    c = cirq.Circuit()
    num_iter = math.floor((math.pi / 4) * math.sqrt(2**len(qubits)))
    # Initialize qubits.
    c.append([
        cirq.H.on_each(*qubits),
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    # Repeat G
    for _ in range(num_iter):
        c.append(Zf(*(qubits)))
        c.append([
            cirq.H.on_each(*qubits)
        ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        c.append(Z0(*(qubits)))
        c.append([
            cirq.H.on_each(*qubits)
        ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    c.append([
        cirq.measure(*qubits, key='result')
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

if __name__ == '__main__':
    main()