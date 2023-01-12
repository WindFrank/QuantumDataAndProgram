import itertools
import numpy as np
from pyquil import get_qc, Program
from pyquil.gates import *

from simon_classical import test_wiki , test_2, test_linear
from util import *

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
f = args.f # test_wiki

start = time.time()

if n == 1:
    if f(0) == f(1):
        s = 1
    else:
        s = 0
    
    end = time.time()
    print("s: ", s)
    print("Execution time: ", end - start)

else:
    M = []

    while rank(M) < n - 1:
        p = Program()

        p.defgate("B_f", create_Bf_matrix(f, n))

        if args.aspen:
            if n * 2 > 12:
                raise ValueError("Aspen only has 12 qubits.")
            qubits = [0, 1, 2, 6, 7, 10, 11, 13, 14, 15, 16, 17]
            qubits = qubits[:n*2]
        else:
            qubits = range(n*2)

        for q in qubits[:n]:
            p.inst(I(q))
        for q in qubits[n:]:
            p.inst(H(q))

        p.inst(("B_f",) + tuple(qubits[::-1]))

        for q in qubits[n:]:
            p.inst(H(q))

        if args.aspen:    
            qc = get_qc('Aspen-4-12Q-A', as_qvm=True)
            qc.compiler.client.timeout = 100
            if args.send_to_server and args.email != '':
                p_out = "# simon F=" + args.f.__name__ + " N=" + str(args.n) + "\n" + p.out()
                print("program: ", p_out)
                print("email: ", args.email)
                send_to_server(p_out, args.email)
                exit()  # Just send job over, no processing
            else:
                result = qc.run_and_measure(p, trials = 1)
        else:
            qc = get_qc(str(n*2)+'q-qvm')
            qc.compiler.client.timeout = 100
            result = qc.run_and_measure(p, trials = 1)

        y = list(process_results(result, qubits)[0][:n])
        y = [int(i) for i in y]
        y.append(0)

        M = new_sample(M, y)

        del p

    s = solve_reduced_row_echelon_form(M)

    end  = time.time()

    if f("".join(['0' for i in range(n)])) != f("".join([str(i) for i in s])):
        print ("s: ", [0 for i in range(n)])
    else:
        print ("s: ", s)
    print("Execution time: ", end - start)