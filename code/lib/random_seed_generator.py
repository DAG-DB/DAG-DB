# script to generate random seeds for use in config

import numpy as np


HIGH = 10 ** 7

t = 0
rng = np.random.default_rng()
n = int(input(f'How many sets of {t} seeds?'))
if t > 0:
    seeds = rng.integers(HIGH, size=(n, t))
else:
    seeds = rng.integers(HIGH, size=(n))
print("\t'RANDOM_SEEDS': [")
for i, seed in enumerate(seeds):
    if i > 0:
        print('')
    if t > 0:
        print(f'\t\t\t{tuple(seed)},', end='')
    else:
        print(f'\t\t\t{seed},', end='')
print('\b\n\t\t],')
