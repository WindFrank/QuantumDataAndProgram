#  gate set of the Google xmon architecture
# import cirq.google

import cirq
import random

import numpy as np

def rand2d(rows, cols):
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

def random_instance(length):
    # transverse field terms
    h = rand2d(length, length)
    # links within a row
    jr = rand2d(length - 1, length)
    # links within a column
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

def rot_x_layer(length, half_turns):
    """Yields X rotations by half_turns on a square grid of given length."""
    rot = cirq.XPowGate(exponent=half_turns)
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))

def rot_z_layer(h, half_turns):
    """Yields Z rotations by half_turns conditioned on the field h."""
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.GridQubit(i, j))

def rot_11_layer(jr, jc, half_turns):
    """Yields rotations about |11> conditioned on the jr and jc fields."""
    gate = cirq.CZPowGate(exponent=half_turns)
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            if jr_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i + 1, j))
            yield gate(cirq.GridQubit(i, j),
                       cirq.GridQubit(i + 1, j))
            if jr_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i + 1, j))

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            if jc_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i, j + 1))
            yield gate(cirq.GridQubit(i, j),
                       cirq.GridQubit(i, j + 1))
            if jc_ij == -1:
                yield cirq.X(cirq.GridQubit(i, j))
                yield cirq.X(cirq.GridQubit(i, j + 1))

def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_x_layer(length, x_half_turns)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(jr, jc, j_half_turns)


# define the length of the grid.
length = 3
# define qubits on the grid.
qubits = [cirq.GridQubit(i, j) for i in range(length) for j in range(length)]

# J and h in the Hamiltonian definition
# For J use two lists, one for the row links - jr and one for the column links - jc
h, jr, jc = random_instance(3)
simulator = cirq.Simulator()
circuit = cirq.Circuit()
circuit.append(one_step(h, jr, jc, 0.1, 0.2, 0.3))
circuit.append(cirq.measure(*qubits, key='x'))
print(circuit)

# repeat for 100 times and histogram of the counts of the measurement results
results = simulator.run(circuit, repetitions=100)
print("results - ")
print(results.histogram(key='x'))

#-------------------------------------------------------------------------------

# calculate the value of the objective function
def energy_func(length, h, jr, jc):
    def energy(measurements):
        # Reshape measurement into array that matches grid shape.
        meas_list_of_lists = [measurements[i * length:(i + 1) * length]
                              for i in range(length)]
        # Convert true/false to +1/-1.
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)

        tot_energy = np.sum(pm_meas * h)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]
        return tot_energy
    return energy


def obj_func(result):
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, h, jr, jc))
    print("energy_hist - ")
    print(energy_hist)
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions

print('Value of the objective function {}'.format(obj_func(results)))


# output like

#                    ????????????????????????????????????       ????????????????????????????????????       ????????????????????????????????????
# (0, 0): ?????????X^0.1????????????X??????????????????????????????????????????????????????@???????????????????????????????????????X????????????X???????????????????????????????????????@?????????????????????X?????????????????????????????????????????????????????????????????????????????????????????????M('x')?????????
#                                        ???                                ???                                       ???
# (0, 1): ?????????X^0.1???????????????????????????@????????????????????????X???????????????????????????????????????????????????????????????????????????????????????????????????????????????@^0.3?????????X?????????????????????@?????????????????????????????????????????????????????????????????????M????????????????????????
#                          ???             ???                                                ???                       ???
# (0, 2): ?????????X^0.1????????????Z^0.2???????????????????????????X???????????????????????????@????????????????????????X??????????????????????????????????????????????????????????????????????????????????????????????????????@^0.3?????????????????????????????????????????????????????????M????????????????????????
#                          ???             ???    ???                                                                   ???
# (1, 0): ?????????X^0.1????????????Z^0.2???????????????????????????X????????????@^0.3???????????????????????????X????????????@???????????????????????????????????????@?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????M????????????????????????
#                          ???                  ???             ???             ???                                       ???
# (1, 1): ?????????X^0.1???????????????????????????@^0.3????????????X????????????@???????????????????????????????????????X??????????????????????????????????????????????????????@^0.3?????????X?????????????????????@?????????????????????X?????????????????????????????????????????????M????????????????????????
#                                        ???    ???             ???                             ???                       ???
# (1, 2): ?????????X^0.1????????????X?????????????????????????????????????????????????????????????????????@^0.3????????????X???????????????????????????@????????????????????????X?????????????????????????????????????????????@^0.3?????????X?????????????????????????????????????????????M????????????????????????
#                                        ???                  ???    ???                                                ???
# (2, 0): ?????????X^0.1????????????Z^0.2???????????????????????????????????????????????????????????????????????????????????????????????????@^0.3???????????????????????????X?????????????????????@?????????????????????X?????????????????????????????????????????????????????????????????????M????????????????????????
#                                        ???                       ???                ???                               ???
# (2, 1): ?????????X^0.1????????????X??????????????????????????????????????????????????????@^0.3???????????????????????????X????????????X???????????????????????????????????????????????????????????????@^0.3?????????X?????????????????????X?????????@?????????????????????X?????????M????????????????????????
#                                                                ???                                    ???           ???
# (2, 2): ?????????X^0.1????????????Z^0.2??????????????????????????????????????????????????????????????????????????????????????????????????????????????????@^0.3????????????X?????????????????????????????????????????????????????????????????????????????????@^0.3?????????X?????????M????????????????????????
#                    ????????????????????????????????????       ????????????????????????????????????       ????????????????????????????????????
#
# results -
# Counter({0: 78, 2: 4, 1: 3, 64: 3, 8: 2, 128: 2, 16: 1, 260: 1, 4: 1, 256: 1, 20: 1, 68: 1, 272: 1, 6: 1})
#
# energy_hist -
# Counter({-3: 81, -5: 7, 5: 5, 1: 3, -1: 2, 3: 1, -7: 1})
# Value of the objective function -2.56
#

