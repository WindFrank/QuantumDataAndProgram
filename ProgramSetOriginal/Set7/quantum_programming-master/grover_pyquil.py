# Quantum Grover's algorithm. Implemented as described in lecture.

import numpy as np
from pyquil import get_qc, Program
from pyquil.api import QuantumComputer
from pyquil.gates import *

from grover_classical import test_only_one, test_01, test_000, test_10010, test_1111
from util import create_Z0_matrix, create_Zf_matrix, process_results, send_to_server

import time
import sys
import argparse
import math

def create_lambda_with_globals(s):
    return eval(s, globals())

parser = argparse.ArgumentParser()

parser.add_argument("-n",dest="n", type=int, required=True)
parser.add_argument("-f",dest="f", type=create_lambda_with_globals, required=True)
parser.add_argument('--aspen', action = 'store_true', default = False)
parser.add_argument('--send_to_server', action = 'store_true', default = False)
parser.add_argument('--email', default = '')
args = parser.parse_args()

n = args.n
f = args.f
t = 10

start = time.time()

p = Program()
p.defgate("Z_0", create_Z0_matrix(n))
p.defgate("Z_f", create_Zf_matrix(f, n))

num_iter = math.floor((math.pi / 4) * math.sqrt(2**n))

if args.aspen:
    if n > 12:
        raise ValueError("Aspen only has 12 qubits.")
    qubits = [0, 1, 2, 6, 7, 10, 11, 13, 14, 15, 16, 17]
    qubits = qubits[:n]
else:
    qubits = range(n)

# Initial Hadamard
for q in qubits:
    p.inst(H(q))
# Repeat G
for i in range(num_iter):
    p.inst(("Z_f",) + tuple(qubits[::-1]))
    for q in qubits:
        p.inst(H(q))
    p.inst(("Z_0",) + tuple(qubits[::-1]))
    for q in qubits:
        p.inst(H(q))

if args.aspen:
    qc = get_qc('Aspen-4-12Q-A', as_qvm=True)
    qc.compiler.client.timeout = 100
    if args.send_to_server and args.email != '':
        p_out = "# grover F=" + args.f.__name__ + " N=" + str(args.n) + "\n" + p.out()
        print("program: ", p_out)
        print("email: ", args.email)
        send_to_server(p_out, args.email)
    else:
        result = qc.run_and_measure(p, trials = t)
        end = time.time()
        print(process_results(result, qubits))
        print("Execution time: ", end - start)
else:
    # You can make any 'nq-qvm' this way for any reasonable 'n'
    qc = get_qc(str(n)+'q-qvm')
    qc.compiler.client.timeout = 100
    result = qc.run_and_measure(p, trials = t)
    end = time.time()
    print(process_results(result, qubits))
    print("Execution time: ", end - start)