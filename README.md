# QuantumProgramAnalysisAndCircuitGeneration

The project collects 18 quantum programs which are selected from 76 programs on Github by the steps as follow: (The file basic_arithmetic.py has two sections of code with different functions )

 (1) The quantum circuit structures are correctly supported by the quantum cloud platform.
 
 (2) If the quantum circuit uses a structure that is not supported by the quantum cloud platform, we attempt to replace the complex quantum gate with the equivalent of the supported quantum gate. 
 
 (3) If the structure cannot be replaced or the scale after replacement changes too much, the quantum circuit will be only used in the quantum simulator or abandoned. 
 
 (4) If a quantum circuit still does not work properly for some reason (running time is too long, requiring too much computing resource, etc.), it will be excluded. 

2of5.py: 2of5 function takes 5 inputs and 1 output. The output is 1 if and only if exactly two of its inputs are 1.

6sym.py: 6sym function takes 6 inputs and 1 output. The output is 1 if and only if exactly 2 or 3 or 6 of its inputs are 1.

Simple-Cirq-Program.py: A bit-flip program which changes 0 to 1 or changes 1 to 0.

adder.py: Cirq implementation of the adder function.

basic_arithmetic.py: Cirq implementation of binary adder and multiplication functions. We split it into two programs to study.

bb84.py: Example program to demonstrate BB84 QKD Protocol. BB84 is a quantum key distribution (QKD) protocol developed by Charles Bennett and Gilles Brassard in 1984. It was the first quantum cryptographic protocol, using the laws of quantum mechanics (specifically, no-cloning) to provide provably secure key generation.

bernstein_vazirani.py: Demonstrates the Bernstein-Vazirani algorithm. The (non-recursive) Bernstein-Vazirani algorithm takes a black-box oracle implementing a function f(a) = a·factors + bias (mod 2), where 'bias' is 0 or 1, 'a' and 'factors' are vectors with all elements equal to 0 or 1, and the algorithm solves for 'factors' in a single query to the oracle.

deutsch.py: Demonstrates Deutsch's algorithm. Deutsch's algorithm is one of the simplest demonstrations of quantum parallelism and interference. It takes a black-box oracle implementing a Boolean function f(x), and determines whether f(0) and f(1) have the same parity using just one query.  This version of Deutsch's algorithm is a simplified and improved version from Nielsen and Chuang's textbook.

grover.py: Demonstrates Grover algorithm. The Grover algorithm takes a black-box oracle implementing a function {f(x) = 1 if x==x', f(x) = 0 if x!= x'} and finds x' within a randomly ordered sequence of N items using O(sqrt(N))  operations and O(N log(N)) gates, with the probability p >= 2/3.

hello_quantum_world.py: Use a series of quantum circuits to build a binary number array and then turn it into message "hello quantum wolrd".

hello_qubit.py:  A quantum circuit with only a square root of X gate and measurement gate which will produce 0 or 1 with equal probability.

hidden_shift_algorithm.py: Example program that demonstrates a Hidden Shift algorithm. The Hidden Shift Problem is one of the known problems whose quantum algorithm solution shows exponential speedup over classical computing. Part of the advantage lies on the ability to perform Fourier transforms efficiently. This can be used to extract correlations between certain functions.

little_belle.py little_elsa.py little_jasmine.py: The programs for quantum research: https://arxiv.org/abs/2004.08539.

quantum_teleportation.py: Quantum Teleportation. Quantum Teleportation is a process by which a quantum state can be transmitted by sending only two classical bits of information. This is accomplished by pre-sharing an entangled state between the sender (Alice) and the receiver (Bob). This entangled state allows the receiver (Bob) of the two classical bits of information to possess a qubit with the same state as the one held by the sender (Alice)

superdense_coding.py: Superdense Coding. Superdense Coding is a method to transmit two classical bits of information by sending only one qubit of information. This is accomplished by pre-sharing an entangled state between the sender and the receiver. This entangled state allows the receiver of the one qubit of information to decode the two classical bits that were originally encoded by the sender.

Because of the dearth of real data, we design a random circuit program to generate circuits with various structures. 
With a large volumn of data, we can use HGP-SL GNN model to training and predict the circuit behaviors on model.

