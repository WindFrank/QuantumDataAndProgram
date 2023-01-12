# Quantum Bernstein-Vazirani. Implemented as described in lecture.

import numpy as np
from pyquil import get_qc, Program
from pyquil.api import QuantumComputer
from pyquil.gates import *

from bernstein_vazirani_classical import test_f_4, test_f_5, test_f_6
from util import create_Uf_matrix, process_results, send_to_server

import time
import sys
import argparse

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
f = args.f  # test_f_4, test_f_5
t = 10

start = time.time()

p = Program()
p.defgate("U_f", create_Uf_matrix(f, n))

if args.aspen:
    if n+1 > 12:
        raise ValueError("Aspen only has 12 qubits.")
    qubits = [0, 1, 2, 6, 7, 10, 11, 13, 14, 15, 16, 17]
    qubits = qubits[:n+1]
else:
    qubits = range(n+1)

p.inst(X(qubits[0]))
p.inst(H(qubits[0]))
for q in qubits[1:]:
    p.inst(H(q))
p.inst(("U_f",) + tuple(qubits[::-1]))  # Applying U_f
for q in qubits[1:]:
    p.inst(H(q))

if args.aspen:    
    qc = get_qc('Aspen-4-6Q-A', as_qvm=True)
    qc.compiler.client.timeout = 100
    if args.send_to_server and args.email != '':
        p_out = "# BV F=" + args.f.__name__ + " N=" + str(args.n) + "\n" + p.out()
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
    qc = get_qc(str(n+1)+'q-qvm')
    qc.compiler.client.timeout = 100
    result = qc.run_and_measure(p, trials = t)
    end = time.time()
    print(process_results(result, qubits))
    print("Execution time: ", end - start)