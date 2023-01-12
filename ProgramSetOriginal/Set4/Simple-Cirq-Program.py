#*** SIMPLE PROGRAM in Cirq.***

# Import the Cirq Package.
import cirq

# Pick a Qubit.
qubit = cirq.GridQubit(0,0)
# Create a Circuit.
circuit = cirq.Circuit([
    cirq.X(qubit),  # NOT Gate
    cirq.measure(qubit, key='m')  # Measurment Gate.
])
# Display the Circuit
print("Circuit: ", circuit)
# Get a simulator to execute the circuit.
simulator = cirq.Simulator()
# Simulate the circuit several times.
result = simulator.run(circuit, repetitions=10)
# Print the Results.
print("Results: ", result)
