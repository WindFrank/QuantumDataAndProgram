# Classical Implementation of Deutsch-Jozsa.
# We'll just use strings to represent bit-strings of 0 and 1.

import random

# Assume f is either balanced or constant.
# f takes in a string of 0s and 1s and returns an int of 0 or 1.
#
# This function tests just over half the possible inputs to f.
# If they all return the same value, then under the assumption that
# f must be constant or balanced, the function must be constant.
# Otherwise if it returns differing results for any of the inputs
# we know that it's balanced.
def deutsch_jozsa(f, n):
	b_format = "{0:0" + str(n) + "b}"
	out = f(b_format.format(0))
	i = 1
	while i < 2**(n-1)+1:
		if f(b_format.format(i)) != out:
			return 0
		i += 1
	return 1

# TESTING

def balanced_f(s):
	c = s[-1]
	if c == '0':
		return 0
	else:
		return 1

def constant_f(s):
	return 0

def pseudo_balanced_f(s):
	return random.randint(0, 1)

def test_dj():
	assert deutsch_jozsa(balanced_f, 10) == 0, "balanced_f is not balanced."
	assert deutsch_jozsa(constant_f, 10) == 1, "constant_f is not constant."
	assert deutsch_jozsa(pseudo_balanced_f, 10) == 0, "pseudo_balanced_f is not labelled balanced."
	# Larger Inputs
	assert deutsch_jozsa(balanced_f, 20) == 0, "balanced_f is not balanced."
	assert deutsch_jozsa(constant_f, 20) == 1, "constant_f is not constant."
	assert deutsch_jozsa(pseudo_balanced_f, 20) == 0, "pseudo_balanced_f is not labelled balanced."

if __name__ == '__main__':
	test_dj()