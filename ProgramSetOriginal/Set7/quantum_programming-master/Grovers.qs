// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

namespace GroversAlgorithm {
    
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Arrays;


    
    
    //////////////////////////////////////////////////////////////////
    // Welcome!
    //////////////////////////////////////////////////////////////////
    
    // The "Grover's Search" quantum kata is a series of exercises designed
    // to get you familiar with Grover's search algorithm.
    // It covers the following topics:
    //  - writing oracles for Grover's search,
    //  - performing steps of the algorithm, and
    //    - putting it all together: Grover's search algorithm.
    
    // Each task is wrapped in one operation preceded by the description of the task.
    // Each task (except tasks in which you have to write a test) has a unit test associated with it,
    // which initially fails. Your goal is to fill in the blank (marked with // ... comment)
    // with some Q# code to make the failing test pass.
    
    // Within each section, tasks are given in approximate order of increasing difficulty;
    // harder ones are marked with asterisks.
    
    
    //////////////////////////////////////////////////////////////////
    // Part I. Oracles for Grover's Search
    //////////////////////////////////////////////////////////////////
    
    // Task 1.1. The |11...1⟩ oracle
    operation Oracle_AllOnes (queryRegister : Qubit[], target : Qubit) : Unit
    is Adj {        
        Controlled X(queryRegister, target);
    }
    
    
    // Task 1.2. The |1010...⟩ oracle
    operation Oracle_AlternatingBits (queryRegister : Qubit[], target : Qubit) : Unit
    is Adj {

        // flip the bits in odd (0-based positions),
        // so that the condition for flipping the state of the target qubit is "query register is in 1...1 state"
        FlipOddPositionBits(queryRegister);
        Controlled X(queryRegister, target);
        Adjoint FlipOddPositionBits(queryRegister);
    }
    
    
    operation FlipOddPositionBits (register : Qubit[]) : Unit
    is Adj {
        
        // iterate over elements in odd positions (indexes are 0-based)
        for (i in 1 .. 2 .. Length(register) - 1) {
            X(register[i]);
        }
    }
    
    
    // Task 1.3. Arbitrary bit pattern oracle
    operation Oracle_ArbitraryPattern (queryRegister : Qubit[], target : Qubit, pattern : Bool[]) : Unit
    is Adj {        
        (ControlledOnBitString(pattern, X))(queryRegister, target);
    }
    
    
    // Task 1.4*. Oracle converter
    operation OracleConverterImpl (markingOracle : ((Qubit[], Qubit) => Unit is Adj), register : Qubit[]) : Unit
    is Adj {
        
        using (target = Qubit()) {
            // Put the target into the |-⟩ state
            X(target);
            H(target);
                
            // Apply the marking oracle; since the target is in the |-⟩ state,
            // flipping the target if the register satisfies the oracle condition will apply a -1 factor to the state
            markingOracle(register, target);
                
            // Put the target back into |0⟩ so we can return it
            H(target);
            X(target);
        }
    }
    
    
    function OracleConverter (markingOracle : ((Qubit[], Qubit) => Unit is Adj)) : (Qubit[] => Unit is Adj) {
        return OracleConverterImpl(markingOracle, _);
    }
    
    
    //////////////////////////////////////////////////////////////////
    // Part II. The Grover iteration
    //////////////////////////////////////////////////////////////////
    
    // Task 2.1. The Hadamard transform
    operation HadamardTransform (register : Qubit[]) : Unit
    is Adj {
        
        ApplyToEachA(H, register);

        // ApplyToEach is a library routine that is equivalent to the following code:
        // let nQubits = Length(register);
        // for (idxQubit in 0..nQubits - 1) {
        //     H(register[idxQubit]);
        // }
    }
    
    
    // Task 2.2. Conditional phase flip
    operation ConditionalPhaseFlip (register : Qubit[]) : Unit {
        
        body (...) {
            // Define a marking oracle which detects an all zero state
            let allZerosOracle = Oracle_ArbitraryPattern(_, _, new Bool[Length(register)]);
            
            // Convert it into a phase-flip oracle and apply it
            let flipOracle = OracleConverter(allZerosOracle);
            flipOracle(register);
        }
        
        adjoint self;
    }
    
    // Task 2.3. The Grover iteration
    operation GroverIteration (register : Qubit[], oracle : (Qubit[] => Unit is Adj)) : Unit
    is Adj {
        
        oracle(register);
        HadamardTransform(register);
        ConditionalPhaseFlip(register);
        HadamardTransform(register);
    }
    
    
    //////////////////////////////////////////////////////////////////
    // Part III. Putting it all together: Grover's search algorithm
    //////////////////////////////////////////////////////////////////
    
    // Task 3.1. Grover's search
    operation GroversSearch (register : Qubit[], oracle : ((Qubit[], Qubit) => Unit is Adj), iterations : Int) : Unit
    {
        
        let phaseOracle = OracleConverter(oracle);
        HadamardTransform(register);
            
        for (i in 1 .. iterations) {
            GroverIteration(register, phaseOracle);
        }
       
    }
    
    operation Run_Grovers_Algorithm () : Int[] {
        //select qubit size n 
        let n = 5;
        
        //generate the hidden value
        let iterations = PowD(PowD(2.0, IntAsDouble(n)), 0.5);
        let pattern = IntAsBoolArray(RandomIntPow2(n), n);
        let markingOracle = Oracle_ArbitraryPattern(_, _, pattern);

        //run the algorithm
        let ret1 = GroversSearch(_, markingOracle, Ceiling(iterations));

        //read the solution 
        using ((x, y) = (Qubit[n], Qubit())) {

            ret1(x);

            mutable j = new Int[n];
            for (i in 0 .. n - 1) {
                if (M(x[i]) == One) {
                    set j w/= i <- 1;
                }
            }


            let answer = BoolArrayAsInt(pattern);
            Message(IntAsString(answer));

            let check = markingOracle(x, y);
            if(M(y) == Zero) {
                Message("Success!");
            }
            ResetAll(x);
            Reset(y);
            
            return j;

        }

    }
    
}
