# Utilities, mostly for U_f creation
# Reference repos:
# https://github.com/msohaibalam/quantum_algorithms_from_scratch/blob/master/Bernstein-Vazirani%20Algorithm.ipynb
# https://github.com/msohaibalam/quantum_algorithms_from_scratch/blob/master/Deutsch's%20Algorithm.ipynb

import numpy as np
import cirq
import itertools
import requests
from bernstein_vazirani_classical import test_f_1, test_f_2, test_f_3
from deutsch_jozsa_classical import balanced_f, constant_f, pseudo_balanced_f

# Convert results from measurment into full state per trial
def process_results(result, qubits):
    trials = []
    for ti in range(len(result[0])):
        trial = ''
        for ni in qubits:
            trial = str(result[ni][ti]) + trial
        trials.append(trial)
    return trials

# Converts string-representation of qubit into a vector.
def ket(qs):
    val_map = {'0': np.array([[1], [0]]), 
               '1': np.array([[0], [1]])}
    k = val_map[qs[0]]  # Ket
    for i in range(1, len(qs)):
        k = np.kron(k, val_map[qs[i]])
    return k

# Calculates the outer product of the state input.
def outer_product(qs):
    k = ket(qs)  # Ket
    b = np.transpose(k)  # Bra
    proj = np.kron(k, b)
    return proj

# Returns all calls of f(x) for all possible inputs.
# Stores result in a dict.
def f_x(f, n):
    qs = []
    for q in itertools.product(['0', '1'], repeat=n):
        qs.append(''.join(q))
    fx = {}
    for q in qs:
        fx[q] = f(q)
    return fx

# Creates the Uf matrix. Takes in a lambda function and the input size, n.
# Does this by essentially placing many 2x2 matricies along the diagonal, in 
# order to either keep (add 0 to y) or negate (add 1 to y) the least significant
# bit. The identity gate adds 0, while the X gate adds 1.
def create_Uf_matrix(f, n):
    N = 2**(n+1)
    mat = np.zeros(shape=(N, N))
    fx = f_x(f, n)
    for k, v in fx.items():
        mat += np.kron(outer_product(k), np.eye(2) + v * (np.array([[-1, 1], [1, -1]])))
    return mat

def create_Bf_matrix(f, n):
    N = 2 ** (n*2)
    mat = np.zeros(shape=(N, N))
    fx = f_x(f, n)
    for k, v in fx.items():
        target_operator = np.eye(2) + int(v[0]) * (-np.eye(2) + np.array([[0, 1], [1, 0]]))
        for i in range(1, n):
            constant_operator = np.eye(2) + int(v[i]) * (-np.eye(2) + np.array([[0, 1], [1, 0]]))
            target_operator = np.kron(target_operator, constant_operator)
        mat += np.kron(outer_product(k), target_operator)
    return mat

# Creates the Z0 reflection matrix in Grover's Algorithm. Takes in the input size, n.
# Does this by just creating a big identity matrix and setting the element corresponding
# to all zeros to -1, which is the top left.
def create_Z0_matrix(n):
    N = 2**(n)
    mat = np.identity(N)
    mat[0][0] = -1
    return mat

# Creates the Zf reflection matrix in Grover's Algorithm. 
# Takes in a lambda function and the input size, n.
# Does this by just creating a big identity matrix and setting 
# each element where f(x) = 1 to -1.
def create_Zf_matrix(f, n):
    N = 2**(n)
    mat = np.zeros(shape=(N, N))
    fx = f_x(f, n)
    for k, v in fx.items():
        if v == 1:
            mat -= outer_product(k)
        else:
            mat += outer_product(k)
    return mat

# Cirq custom gate class
# Takes in the number of qubits it operates on, n, and
# the matrix representing the gate operation. Also optionally
# takes in a name.
class Cirq_Custom(cirq.Gate):
    def __init__(self, n, f_mat, name="oracle"):
        cirq.Gate.__init__(self)
        self.n = n
        self.f_mat = f_mat
        self.name = name

    def num_qubits(self):
        return self.n

    def _unitary_(self):
        return self.f_mat

    def __str__(self):
        return self.name

# Sends the job to the 239 server
def send_to_server(program, email, shots=10):
    r = requests.post('http://63.110.29.106:80/send' , 
                    data = {'quil' : program , 'email' : email , 'shots' : shots})
    print(r.text)

# ---------------------------------------------------------------------------------------------------
# Row Echelon linear equation solver functions
# Code taken from the following repository: 
# https://github.com/msohaibalam/quantum_algorithms_from_scratch/blob/master/gaussian_elimination.py

def rank(A):
    '''
    Computes the rank of matrix A
    :param list A: list of lists, representing matrix assumed to be in reduced
        row-echelon form
    :return int: rank of the input matrix
    '''
    # return 0 for empty list input
    if A is None:
        return 0
    # otherwise, count the number of non-zero rows
    return len([l for l in A if set(l) != set([0])])


def add_vec_mod2(v, w):
    """
    Mod 2 addition separately on each components of v and w
    :params list/str v, w: bit-strings (or list representations) to be added mod 2
    :return str: component-wise mod 2 sum of the input strings/lists
    """
    if len(v) != len(w):
        raise AssertionError("Input lengths unequal")
    # convert strings into lists
    if type(v) == str:
        v_ = [int(s) for s in v]
    if type(w) == str:
        w_ = [int(s) for s in w]
    else:
        v_, w_ = v, w
    vadd_mod2 = np.vectorize(lambda x, y: (x + y) % 2)
    # return a bit-string
    return ''.join([str(i) for i in vadd_mod2(v_, w_)])


def msb(l):
    """
    Given a list of bits, find the most significant bit position, counting from right
    E.g. For [0, 1, 0, 1], the output would be 2
    :param list l: list containing bits
    :return int: most significant bit position in the list
    """
    LtoR_index = min([i for i in range(len(l)) if l[i] == 1] or [-1])
    return len(l) - 1 - LtoR_index


def new_sample(W, z):
    """
    Given new sample z, loop until either z has been added to W, or z=0 produced
    :param list W: list of lists, representing matrix in reduced row-echelon form,
        or empty list
    :param list z: new sample
    :return list W: modified list of lists, with the new sample either added or
        discarded
    """
    # sample z, producing/maintaing reduced row echelon form
    for row in range(len(W)):
        # replace z <- z +mod2 W[row] if msbs equal
        if msb(W[row]) == msb(z):
            z = [int(i) for i in add_vec_mod2(z, W[row])]
        # insert non-zero z preserving reduced row echelon form
        if (msb(z) > msb(W[row])) and (set(z) != set([0])):
            W = W[:row] + [z] + W[row:]
            break
        if row == len(W)-1:
            if (msb(z) < msb(W[row])) and (set(z) != set([0])):
                W.append(z)
            break

    # append the very first sample
    if (len(W) == 0) and (set(z) != set([0])):
        W.append(z)

    return W


def complete_basis(A):
    '''
    Complete basis for matrix of rank (n-1) with a new vector not orthogonal to the
    period vector
    :param list A: lists of lists, representing matrix in reduced row-echelon form
        of rank (n-1)
    :return list A: modified list of lists, with all 1s along the diagonal
    '''
    # Starting from top row w_0, look for the lowest w_i which has its leading 1
    # in column k, but w_(i+1) directly below it has a 0 in its (i+1)st position
    num_rows = len(A)
    num_cols = len(A[0])
    for i in range(num_rows - 1):
        if (A[i][i] == 1) and (A[i + 1][i + 1] == 0):
            A = A[:i + 1] + [[1 if ((ii == i + 1) or (ii == num_cols - 1))
                             else 0 for ii in range(num_cols)]] + A[i + 1:]

    # if a new row has not been added, then either
    # (a) W has all 1s on its diagonal, or
    # (b) W has all 0s on its diagonal
    new_num_rows = len(A)
    new_num_cols = len(A[0])
    if new_num_rows == num_rows:
        # if (a), then complete basis at the end
        if A[0][0] == 1:
            new_row = [1 if ((ii == new_num_cols - 2) or (ii == new_num_cols - 1)) else 0
                       for ii in range(new_num_cols)]
            A.append(new_row)
        # if (b), then complete basis at the beginning
        elif A[0][0] == 0:
            new_row = [1 if ((ii == 0) or (ii == new_num_cols - 1)) else 0
                       for ii in range(new_num_cols)]
            A = [new_row] + A

    return A


def back_substitue_mod2(A):
    '''
    Perform mod 2 back-substition, assuming reduced row-echelon form
    :param list A: list of lists, representing matrix in reduced row-echelon form
    :return list: solution resulting from mod 2 back-substition
    '''
    # starting with bottom-most row, solve for all variables via back-substitution
    n = len(A)
    x = [0 for _ in range(n)]
    # loop along the diagonal
    for i in range(n - 1, -1, -1):
        # solve for variable x[i]
        x[i] = A[i][n]
        # back-substitute this solution to previous equations
        for j in range(i - 1, -1, -1):
            A[j][n] = (A[j][n] + (A[j][i] * x[i])) % 2

    return x


def solve_reduced_row_echelon_form(A):
    '''
    Solve the system of equations resulting from the reduced row-echelon form
    of the input matrix, by first completing the basis, then solving by
    mod-2 back-substitution
    :param list A: list of lists, representing matrix in reduced row-echelon form
    :return list: solution resulting from the system of equations, i.e. the period
        vector
    '''
    # complete basis with vector not orthogonal to the period vector
    A = complete_basis(A)
    # ensure that matrix is now diagonal
    if (len(A) != len(A[0]) - 1) or (set([A[i][i] for i in range(len(A))]) != set([1])):
        print ("len(A): ", len(A))
        print ("len(A[0]): ", len(A[0]))
        print ("A: ", A)
        raise ValueError("Matrix not in expected form, even after completing basis")
    # solve by back-substitution
    x = back_substitue_mod2(A)

    return x

# ---------------------------------------------------------------------------------------------------
# Test Code

def test_uf_matrix():
    print(create_Uf_matrix(balanced_f, 3))
    print(create_Uf_matrix(constant_f, 3))
    print(create_Uf_matrix(pseudo_balanced_f, 3))
    print(create_Uf_matrix(test_f_1, 5))
    # print(create_Uf_matrix(test_f_2, 20))
    print(create_Uf_matrix(test_f_3, 2))

if __name__ == "__main__":
    test_uf_matrix()