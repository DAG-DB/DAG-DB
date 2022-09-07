

import causaldag as cd
import networkx as nx
import numpy as np


def is_dag(B):
    """Check whether B corresponds to a DAG.

    Args:
        B (numpy.ndarray): [d, d] binary or weighted matrix.
    """
    return nx.is_directed_acyclic_graph(nx.DiGraph(B))


def get_shd_prec_rec_c(G_true, G_est):
	"""
	Return class DAG related metrics

	:param G_true: numpy (d, d) {-1, 0, 1} matrix: true adjacency matrix,
	with -1 in one of position [i, j] or [j, i] representing
	undirected edge i---j if is already CPDAG
	:param G_est: numpy (d, d) {0, 1} matrix, the estimated adjacency matrix
	:return: SHD_c, prec_c, rec_c as defined in thesis
	"""
	cpdag_true = cd.DAG.from_amat(G_true).cpdag()
	if (G_est == -1).any():
		undir_edges = (G_est == -1)
		dir_edges = (G_est == 1)
		cpdag_est = cd.PDAG.from_amat(dir_edges + undir_edges + undir_edges.T)
	else:
		assert is_dag(G_est)
		cpdag_est = cd.DAG.from_amat(G_est).cpdag()
	shd_c = cpdag_true.shd(cpdag_est)
	edges_est = cpdag_est.edges | cpdag_est.arcs  # in casualdag
	# an edge is undirected, an arc is directed
	edges_true = cpdag_true.edges | cpdag_true.arcs  # in casualdag
	# an edge is undirected, an arc is directed
	prec_c = len(edges_est.intersection(edges_true)) / max(len(edges_est), 1)
	rec_c = len(edges_est.intersection(edges_true)) / max(len(edges_true), 1)
	return shd_c, prec_c, rec_c


def count_accuracy(B_bin_true, B_bin_est, check_input=False):
    """
	Thanks to https://github.com/ignavierng/golem/blob/main/src/utils/utils.py
	from which this mainly comes, with some additions here to deal with
	precision and CPDAG metrics

    Compute various accuracy metrics for B_bin_est.

    true positive = predicted association exists in condition in correct direction.
    reverse = predicted association exists in condition in opposite direction.
    false positive = predicted association does not exist in condition.

    Args:
        B_bin_true (np.ndarray): [d, d] binary adjacency matrix of ground truth. Consists of {0, 1}.
                ch: or None to skip things that involve this
        B_bin_est (np.ndarray): [d, d] estimated binary matrix. Consists of {0, 1, -1},
            where -1 indicates undirected edge in CPDAG.

    Returns:
        fdr: (reverse + false positive) / prediction positive.
        tpr: (true positive) / condition positive.
        fpr: (reverse + false positive) / condition negative.
        shd: undirected extra + undirected missing + reverse.
        pred_size: prediction positive.

    Code adapted slightly from GOLEM, itself based on:
        https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    if check_input:
        if (B_bin_est == -1).any():  # CPDAG
            if not ((B_bin_est == 0) | (B_bin_est == 1) | (B_bin_est == -1)).all():
                raise ValueError("B_bin_est should take value in {0, 1, -1}.")
            if ((B_bin_est == -1) & (B_bin_est.T == -1)).any():
                raise ValueError("Undirected edge should only appear once.")
        else:  # dag
            if not ((B_bin_est == 0) | (B_bin_est == 1)).all():
                raise ValueError("B_bin_est should take value in {0, 1}.")
            if not is_dag(B_bin_est):
                raise ValueError("B_bin_est should be a DAG.")
    if B_bin_true is not None:
        d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_bin_est == -1)
    pred = np.flatnonzero(B_bin_est == 1)
    if B_bin_true is not None:
        cond = np.flatnonzero(B_bin_true)
        cond_reversed = np.flatnonzero(B_bin_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True)
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    if B_bin_true is not None:
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance
        pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
        cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
        extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        shd = len(extra_lower) + len(missing_lower) + len(reverse)
        nshd = shd / B_bin_true.shape[0]
        shd_c, prec_c, rec_c = get_shd_prec_rec_c(B_bin_true, B_bin_est)
        nshd_c = shd_c / B_bin_true.shape[0]
        return {'fdr': fdr, 'tpr': tpr, 'prec_c': prec_c, 'fpr': fpr, 'shd': shd,
                'nshd': nshd, 'shd_c': shd_c, 'rec_c': rec_c,
                'nshd_c': nshd_c, 'pred_size': pred_size}
    else:
        return {'pred_size': pred_size}


if __name__ == '__main__':
	G = nx.DiGraph()
	G.add_edge(0, 2)
	G.add_edge(1, 2)
	G.add_edge(1, 4)
	G.add_edge(2, 3)
	G.add_edge(2, 5)
	G.add_edge(3, 4)
	G.add_edge(3, 6)
	G = nx.to_numpy_array(G)

	H = nx.DiGraph()
	H.add_edge(0, 2)
	H.add_edge(1, 2)
	H.add_edge(1, 4)
	H.add_edge(2, 3)
	H.add_edge(2, 5)
	H.add_edge(5, 6)
	H = nx.to_numpy_array(H)
	shd_c, prec_c = get_shd_prec_rec_c(G, H)
	pass
