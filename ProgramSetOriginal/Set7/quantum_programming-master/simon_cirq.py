import random
import cirq
import argparse
import time
import math

from util import create_Bf_matrix, Cirq_Custom, rank, solve_reduced_row_echelon_form, new_sample
from simon_classical import test_wiki , test_2, test_linear, test_2b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",dest="n", type=int, required=True)
    parser.add_argument("-f",dest="f", type=create_lambda_with_globals, required=True)
    args = parser.parse_args()

    start = time.time()

    # Special corner case for n = 1
    if args.n == 1:
        if args.f(0) == args.f(1):
            s = 1
        else:
            s = 0
        end = time.time()
        print("s: ", s)
        print("Execution time: ", end - start)
        return

    # Choose qubits to use.
    measure_qubits = [cirq.GridQubit(i, 0) for i in range(args.n)]
    aux_qubits = [cirq.GridQubit(i, 0) for i in range(args.n, 2 * args.n)]

    # Create Bf
    Bf = Cirq_Custom(2 * args.n, create_Bf_matrix(args.f, args.n), name="Bf")

    # Create DJ circuit
    circuit = make_simon_circuit(measure_qubits, aux_qubits, Bf)
    print('Circuit:')
    print(circuit)

    # Sample from the circuit a couple times until we have the proper
    # number of results to calculate "s".
    simulator = cirq.Simulator()
    M = []
    while rank(M) < args.n - 1:
        result = simulator.run(circuit, repetitions=1)
        frequencies = result.histogram(key='result', fold_func=bitstring)
        r = list(frequencies.keys())[0]
        y = [int(i) for i in r]
        y.append(0)
        M = new_sample(M, y)
    s = solve_reduced_row_echelon_form(M)
    # Test s = all zeros, because that is the only other option other than our result.
    if args.f("".join(['0' for i in range(args.n)])) != args.f("".join([str(i) for i in s])):
        print ("s: ", [0 for i in range(args.n)])
    else:
        print ("s: ", s)

    end = time.time()
    print("Execution time: ", end - start)

def create_lambda_with_globals(s):
    return eval(s, globals())

def make_simon_circuit(measure_qubits, aux_qubits, Bf):
    """Creates grovers circuit"""
    c = cirq.Circuit()
    c.append([
        cirq.H.on_each(*measure_qubits)
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    c.append(Bf(*(measure_qubits + aux_qubits)))
    c.append([
        cirq.H.on_each(*measure_qubits),
        cirq.measure(*measure_qubits, key='result')
    ],strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

if __name__ == '__main__':
    main()