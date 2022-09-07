"""  Create plot for thesis app. C, Theoretical basis for
perturb–and–MAP
"""

import math

import numpy as np
import matplotlib.pyplot as plt

eps = 1e-13

p = 0.99
q = 0.95



def kl_per_bit(p, q):
	return p * math.log(p / q) + (1- p) * math.log((1 - p) / (1 -q ) )

def error_bound_per_bit(d):
	r = d ** 2 - d
	return math.log2(2 * r) / r

kl = kl_per_bit(p, q)
print(f'{kl=:.1e}')
d = 30
error_bound = error_bound_per_bit(d)
print(f'error bound per bit={error_bound:.1e}')

xs = np.linspace(2, 100)
ys = np.array([error_bound_per_bit(x) for x in xs])

plt.figure()
plt.plot(xs, ys, label='error bound')
plt.hlines(kl, xmin=2, xmax=100, colors='r', label='certainty change')
plt.xlabel(r'$d$')
plt.ylabel('per bit')
plt.legend()
plt.show()

