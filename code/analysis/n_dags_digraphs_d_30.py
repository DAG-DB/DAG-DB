""" For calculation of number of digraphs for d=30 and comparison to number
of DAGs
"""

import math

d = 30

# From https://oeis.org/A003024/b003024.txt number of DAGs on 30 labelled
# vertices:
n_dags = \
	271485443716752943844808616164267238071595554149100046470743664333951689240913458768567182648011040358976634252990715237305791310221459864881091225593413894143.
print(n_dags)

# Number of DAGs with no self-loops:
n_digraphs = math.pow(2, (d * (d-1)))
print(n_digraphs)
