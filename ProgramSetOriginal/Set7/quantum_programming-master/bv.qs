namespace bv_algorithm
{
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Arrays;

    operation Oracle_Zero (x : Qubit[], y : Qubit, r : Int[]) : Unit {
        // Since f(x) = 0 for all values of x, |y ⊕ f(x)⟩ = |y⟩.
        // This means that the operation doesn't need to do any transformation to the inputs.
    }

    operation Oracle_ProductFunction (x : Qubit[], y : Qubit, r : Int[]) : Unit {
        for (i in IndexRange(x)) {
            if (r[i] == 1) {
                CNOT(x[i], y);
            }
        }
    }

    operation BV_Algorithm (N : Int, oracle : ((Qubit[], Qubit, Int[]) => Unit), a : Int[]) : Int[] {
        using ((x, y) = (Qubit[N], Qubit())) {
            // Newly allocated qubits start in the |0⟩ state.
            // The first step is to prepare the qubits in the required state before calling the oracle.
            // Each qubit of the input register has to be in the |+⟩ state.
            ApplyToEachA(H, x);

            // The answer register has to be in the |-⟩ state.
            X(y);
            H(y);

            // Apply the oracle to the input register and the answer register.
            oracle(x, y, a);

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
            return r;
        }
    }

    function AllEqualityFactI(actual : Int[], expected : Int[]) : Bool {
        // Check that array lengths are equal
        if(Length(actual) != Length(expected)) {
            return false;
        }
        
        // Check that the corresponding elements of the arrays are equal
        let N = Length(actual);
        for (i in 0 .. N - 1) {
            if(actual[i] != expected[i]) {
                return false;
            }
        }
        return true;
    }

    operation Run_BernsteinVazirani_Algorithm (a : Int[]) : String {
        let n = Length(a);
        if (not AllEqualityFactI(BV_Algorithm(n, Oracle_ProductFunction, a), a)) {
            return "Incorrect result for f(x) = [1 0 1]x";
        }
        
        // If all tests pass, report success!
        return "Success!";
    }
}
 