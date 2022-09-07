""" Show the SHD between two random DAGs
"""

from data_management.synthetic_dataset import SyntheticDataset
from utils.metrics import count_accuracy


n = 1000
d = 30
er = 2
B1 = SyntheticDataset(n, d, 'ER', 2 * er, 'gaussian_ev', 1., seed=101)
B1_bin = B1.B.astype(bool).astype(int)
B2 = SyntheticDataset(n, d, 'ER', 2 * er, 'gaussian_ev', 1., seed=20453)
B2_bin = B2.B.astype(bool).astype(int)
size1 = B1_bin.sum()
size2 = B2_bin.sum()
metrics = count_accuracy(B1_bin, B2_bin)
print(metrics['nshd_c'])
