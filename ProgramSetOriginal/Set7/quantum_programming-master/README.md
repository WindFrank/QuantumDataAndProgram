# quantum_programming

## Bernstein-Vazirani and Deutsch-Jorsa

BV_pyquil.py runs Bernstein-Vazirani, and DJ_pyquil.py runs Deutsch-Jorsa. Both take parameters -f and -n, for the lambda input function and the number of qubits in x respectively.

Here are a few example runs:

```
python DJ_pyquil.py -n 5 -f constant_f
python BV_pyquil.py -n 4 -f test_f_4
```

The output is an array of bitstrings. The number of bitstrings is equal to the number of trials (this is hard-coded to 10 in the files, but you can change that by modifying the ‘t’ variable in the code). Each bitstring shows the result of each trial which is the measured result of the qubits. We include the helper ‘y’ qubit in the measurement for the sake of thoroughness, but in these algorithms typically this qubit is ignored. Also note that if you want to use your own lambda function, you’ll have to make sure it’s imported into the respective file.

