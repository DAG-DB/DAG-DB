""" The algorithm for generate a DAG by uniform sampling, set out in the
thesis' Appendix A.  Plus automated test that confirms it produces DAGs
"""


import igraph as ig
import networkx as nx
import numpy as np


RNG = np.random.default_rng(seed=1882)
TEST_RNG = np.random.default_rng(seed=1961)


def is_perm_matrix(mat):
    """
    Check if a matrix is a permutation matrix

    :param mat: numpy matrix with two dimensions
    :return: whether is permutation matrix
    """
    # Check mat is {0, 1} matrix
    if not np.all(mat.astype(bool).astype(int) == mat):
        return False
    # Check is column and row adds to one
    col_sums = mat.sum(axis=0)
    row_sums = mat.sum(axis=1)
    for dimension_sum in [col_sums, row_sums]:
        if not np.all(dimension_sum == 1):
            return False
    return True


def generate_dag_uniformly(d, digraph_method):
    """
    Sample DAGs uniformly
    :param d: int > 0, number of nodes
    :param digraph_method: function for generating random digraph adj. matrix.
     Must have no arguments (e.g. handle them before as at foot of file)
    :return: networkx DAG, and, for convenience,
     numpy (d, d) {0, 1} matrix, the DAG adjacency matrix
    """

    # Nodes are already arbitrarily numbered by indices of digraph adj. matrix

    g = digraph_method()

    adj = nx.to_numpy_array(g)
    adj = np.tril(adj, k=-1)  # get lower triangular part

    # Get (and sort) topological generations (could be done directly in numpy
    # as an alternative approach, given that the matrix is lower-triangular)
    graph = nx.from_numpy_matrix(adj, create_using=nx.DiGraph)
    top_gens = [
        sorted(generation) for generation in nx.topological_generations(graph)]

    # Uniformly sample a permutation on d nodes
    sigma = RNG.permutation(d)

    # Create the modified permutation matrix perm_mod
    col = 0
    perm_mod = np.zeros((d, d), dtype=np.int64)
    for top_gen in top_gens:
        for old_node in top_gen:
            new_node = sigma[old_node]
            perm_mod[new_node, col] = 1
            col += 1

    assert is_perm_matrix(perm_mod)

    dag_adj = perm_mod @ adj @ perm_mod.T

    dag = nx.from_numpy_matrix(dag_adj, create_using=nx.DiGraph)

    assert nx.is_directed_acyclic_graph(dag)

    return dag, dag_adj


def digraph_method_factory(d, graph_type, k):
    """
    Factory to create digraph random sampling methods for sample_dag_uniformly

    :param d: int > 0, no. of nodes
    :param graph_type: 'ER' or 'SF' or 'SF-nx'
    :param k: int > 0 parameter for ER or SF graph
    :return: digraph_method function, with no arguments
    """
    match graph_type:
        case 'ER':  # Erdos-Renyi (as used in thesis)
            def digraph_method():
                return nx.generators.erdos_renyi_graph(
                    n=d, p=2 * k / (d - 1))
            return digraph_method
        case 'SF':  # Barabasi-Albert (as used in thesis) but hard to get to
            # work in a factory function for  testing
            def digraph_method():
                return ig.Graph.Barabasi(
                    n=d, m=k, directed=True).to_networkx()
            return digraph_method
        case 'SF-nx':  # Barabasi-Albert (not as used in thesis)
             # Initialises with star graph on k + 1 nodes
            def digraph_method():
                return nx.generators.barabasi_albert_graph(d, k)
            return digraph_method
        case _:
            raise NotImplementedError



def test_algorithm():
    """
    Test the algorithm over some fairly arbitrary ranges

    If no assert errors, has worked
    """
    d = TEST_RNG.integers(5, 100 + 1)
    graph_type = TEST_RNG.choice(['ER', 'SF-nx'])
    k = TEST_RNG.integers(2, min(d - 2, 10)  + 1)
    print(d, graph_type, k)
    digraph_method = digraph_method_factory(d, graph_type, k)
    generate_dag_uniformly(d, digraph_method)




if __name__ == "__main__":
    for i in range(1_000):
        print(f'{i}:\t', end='')
        test_algorithm()
