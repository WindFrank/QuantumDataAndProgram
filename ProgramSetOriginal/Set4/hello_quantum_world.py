import cirq
import binascii

def convert_string_to_bit_pairs(wordstring):
    print ("received " +  wordstring)

    binary_string = ' '.join(format(x, '08b') for x in bytearray(wordstring, 'utf-8'))
    return get_array_of_bit_pairs(binary_string)

def get_array_of_bit_pairs(binary_string):
    print ("received " +  binary_string)
    n=9
    binary_array = [(binary_string[i:i+n]) for i in range(0, len(binary_string), n)]
    bitpair_array = []
    for bit_string in binary_array:
        n=2
        tmp_bitpair_array = [(bit_string[i:i+n]) for i in range(0, len(bit_string), n)]
        for bit_pair in tmp_bitpair_array :
                bitpair_array.append(bit_pair)

    return bitpair_array

# helper method for readable output
def bitstring(bits):
    return ''.join('1' if e else '0' for e in bits)

bitpair_array = convert_string_to_bit_pairs("hello quantum world")

# Create two quantum and classical registers
qreg = [cirq.LineQubit(x) for x in range (4)]

 #create a cirquit object
circ = cirq.Circuit()

# recvd bit pair string
bitpairstring = ""
recvd_string = ""

loopcount = 0
for bitpair in bitpair_array :

   print (loopcount)
   if (bitpair == ' '):
       continue

   loopcount = loopcount + 1

   #Dictionary of operations for each message
   message = {"00": [],
              "10": [cirq.X(qreg[0])],
              "01": [cirq.Z(qreg[0])],
              "11": [cirq.X(qreg[0]), cirq.Z(qreg[0])]}

   # sender creates bell pair
   circ.append(cirq.H(qreg[0]))
   circ.append(cirq.CNOT(qreg[0], qreg[1]))

   #pick a message to send
   m = bitpair
   print ("sender's sent message =", m)

   #sender encodes the message with the apropriate quantum operation
   circ.append(message[m])

   # receiver measures the bell state
   circ.append(cirq.CNOT(qreg[0], qreg[1]))
   circ.append(cirq.H(qreg[0]))

   circ.append([cirq.measure(qreg[0]), cirq.measure(qreg[1])])

   #print out the cirquit
   print ("\ncirquit:")
   print (circ)

   #print(circ.to_qasm())

   # run the quantum cirquit on a Simulator bqckend
   sim = cirq.Simulator()
   res = sim.run(circ, repetitions=1)

   # print receivers recv'd message - the outcome of the cirquit
   measured_values =  bitstring(res.measurements.values())
   print ("recv'ers recv'd message =", measured_values)
   print ("\n")

   bitpairstring = bitpairstring + measured_values
   if (len(bitpairstring) == 8):
       print (bitpairstring)
       recvd_string = recvd_string + chr(int(bitpairstring[:8], 2))
       bitpairstring = ""

   circ.clear_operations_touching(qreg, range(0, 7))

print ("Received string is: " + recvd_string)
