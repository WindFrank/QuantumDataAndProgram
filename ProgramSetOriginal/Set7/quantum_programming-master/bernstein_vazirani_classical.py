# Classical Implementation of Bernstein-Vazirani.
# We'll just use strings to represent bit-strings of 0 and 1.

import random

# Return (a, b), in the form of string bits.
# f takes in a string of 0s and 1s and returns an int of 0 or 1.
#
# If x = all 0's, then we know the only contribution to the output
# is "b". So we can obtain the value of b from one call.
# We can obtain the value of each bit of "a" by setting x = 1 in only
# one bit, and 0 in the rest. The result will simply be that bit value in
# "a", plus b. So we can make a call to f for each bit to construct what "a"
# is one bit at a time.
def bernstein_vazirani(f, n):
	b_format = "{0:0" + str(n) + "b}"
	bits = []
	i = 0
	b = f(b_format.format(0))
	while i < n:
		bi = b_format.format(2**i)
		bits = [str((f(bi) + b) % 2)] + bits
		i += 1
	return ''.join(bits), str(b)

# TESTING

def general_f(x, a, b):
	a_dec = int(a, 2)
	x_dec = int(x, 2)
	mult = a_dec & x_dec
	f = bin(mult).count("1") % 2
	return (f + int(b)) % 2

# n = 5
def test_f_1(x):
	a = "01010"
	b = "0"
	return general_f(x, a, b)

# n = 20
def test_f_2(x):
	a = "01010000001111110101"
	b = "1"
	return general_f(x, a, b)

# n = 2
def test_f_3(x):
	a = "00"
	b = "0"
	return general_f(x, a, b)

# n = 4
def test_f_4(x):
	a = "0101"
	b = "0"
	return general_f(x, a, b)

# n = 5
def test_f_5(x):
	a = "10001"
	b = "0"
	return general_f(x, a, b)

# n = 6
def test_f_6(x):
	a = "101001"
	b = "0"
	return general_f(x, a, b)

def test_bv():
	assert bernstein_vazirani(test_f_1, 5) == ("01010", "0")
	assert bernstein_vazirani(test_f_2, 20) == ("01010000001111110101", "1")
	assert bernstein_vazirani(test_f_3, 2) == ("00", "0")

if __name__ == '__main__':
	test_bv()