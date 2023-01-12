# Classical Implementation of Simon's Problem.
# We'll just use strings to represent bit-strings of 0 and 1.

import random

# Returns s according to Simon's problem's definition.
# f takes in a string of 0s and 1s and returns a string of 0s and 1s
#
# Essentially we know that if we can find an x and y that have indentical
# outputs, then we can deduce s. To make it easy we'll just set x = all 0's,
# because if we just iterate over all possible y's we will have tried all possible
# s's, and this also means that y = s so it makes the computation easy.
def simon(f, n):
	b_format = "{0:0" + str(n) + "b}"
	x = b_format.format(0)
	f_x = f(x)
	i = 1
	while i < 2**(n):
		y = b_format.format(i)
		if f(y) == f_x:
			return y
		i += 1
	return x  # Can only occur for valid f inputs if s = all 0's

# TESTING

# Function from the simon's problem wikipedia page
# n=3, s = '110'
def test_wiki(x):
	if x == '000':
		return '101'
	elif x == '001':
		return '010'
	elif x == '010':
		return '000'
	elif x == '011':
		return '110'
	elif x == '100':
		return '000'
	elif x == '101':
		return '110'
	elif x == '110':
		return '101'
	elif x == '111':
		return '010'

# n=2, s = '10'
def test_2(x):
	if x == '00':
		return '01'
	elif x == '01':
		return '10'
	elif x == '10':
		return '01'
	elif x == '11':
		return '10'

# n=2, s = '11'
def test_2b(x):
	if x == '00':
		return '01'
	elif x == '01':
		return '10'
	elif x == '10':
		return '10'
	elif x == '11':
		return '01'

	
# Should return s = all zeros
def test_linear(x):
	return x

def test_sp():
	assert simon(test_wiki, 3) == '110'
	assert simon(test_linear, 10) == '0000000000'

if __name__ == '__main__':
	test_sp()