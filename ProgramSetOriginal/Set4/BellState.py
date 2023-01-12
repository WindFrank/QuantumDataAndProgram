"""Script for preparing the Bell State |\PHI^{+}> in Cirq."""

import cirq

# Get qubits and circuit
qreg = [cirq.LineQubit(x) for x in range(2)]
circ = cirq.Circuit()

# Add the Bell State Preparation Circuit
circ.append([cirq.H(qreg[0]),
            cirq.CNOT(qreg[0], qreg[1])])

# Display the Circuit
print("Circuit: ")
print(circ)

# Add Measurement 
circ.append(cirq.measure(*qreg, key="z"))

# Simulate the circuit
sim = cirq.Simulator()
res = sim.run(circ, repetitions=100)

# Display the outcomes
print("\nMeasurements: ")
print(res.histogram(key="z"))

# Binary Representation of bitstrings:
#     0: stands for 00
#     3: stands for 11
# As expected, there's an approximate 50/50 chance of 00 && 11.
