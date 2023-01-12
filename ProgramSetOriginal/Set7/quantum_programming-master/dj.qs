namespace dj_algorithm
{
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;

    operation Oracle_Zero (x : Qubit[], y : Qubit) : Unit {
        // Since f(x) = 0 for all values of x, |y ⊕ f(x)⟩ = |y⟩.
        // This means that the operation doesn't need to do any transformation to the inputs.
    }

    operation Oracle_One (x : Qubit[], y : Qubit) : Unit {
        X(y);
    }

    operation Oracle_OddNumberOfOnes (x : Qubit[], y : Qubit) : Unit {       
        for (q in x) {
            CNOT(q, y);
        }
    }

    operation DJ_Algorithm (N : Int, oracle : ((Qubit[], Qubit) => Unit)) : Bool {
        // Create a boolean variable for storing the return value.
        // You'll need to update it later, so it has to be declared as mutable.
        mutable isConstantFunction = true;

        // Allocate an array of N qubits for the input register x and one qubit for the answer register y.
        using ((x, y) = (Qubit[N], Qubit())) {
            // Newly allocated qubits start in the |0⟩ state.
            // The first step is to prepare the qubits in the required state before calling the oracle.
            // Each qubit of the input register has to be in the |+⟩ state.
            ApplyToEachA(H, x);

            // The answer register has to be in the |-⟩ state.
            X(y);
            H(y);

            // Apply the oracle to the input register and the answer register.
            oracle(x, y);

            // Apply a Hadamard gate to each qubit of the input register again.
            ApplyToEach(H, x);

            // Measure each qubit of the input register in the computational basis using the M operation.
            // If any of the measurement results is One, the function implemented by the oracle is balanced.
            mutable r = new Int[N];
            for (i in 0 .. N - 1) {
                if (M(x[i]) != Zero) {
                    set r w/= i <- 1;
                }
            }

            // Before releasing the qubits make sure they are all in the |0⟩ state.
            ResetAll(x);
            Reset(y);

            for (i in 0 .. N-1) {
                set isConstantFunction = isConstantFunction and r[i] == 0;
            }
        }
        
        // Return the answer.

        return isConstantFunction;
    }

    operation Run_DeutschJozsa_Algorithm () : String {
        // This is an example for constant function
        // let ret1 = DJ_Algorithm(4, Oracle_One);
        // if(ret1) {
        //     Message("Oracle_One represents a constant function");
        // } else {
        //     Message("Oracle_One represents a balanced function");
        // }
        
        // This is an example for balanced function
        let ret2 = DJ_Algorithm(4, Oracle_OddNumberOfOnes);
        if(ret2) {
            Message("Oracle_OddNumberOfOnes represents a constant function");
        } else {
            Message("Oracle_OddNumberOfOnes represents a balanced function");
        }
        
        // If all tests pass, report success!
        return "Success!";
    }
}
 