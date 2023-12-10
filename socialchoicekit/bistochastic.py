import numpy as np
from typing import List, Tuple, Dict

from socialchoicekit.utils import check_square_matrix
from socialchoicekit.flow import maximum_cardinality_matching_bipartite

def birkhoff_von_neumann(X: np.ndarray) -> List[Tuple[float, np.ndarray]]:
  """
  The Birkhoff-von-Neumann algorithm for decomposing a bistochastic matrix into a convex combination of permutation matrices.

  Parameters
  ----------
  X : np.ndarray
    A bistochastic matrix.

  Returns
  -------
  List[Tuple[int, np.ndarray]]
    A list of tuples of the form (coefficient, permutation matrix).
  """
  check_square_matrix(X)
  n = X.shape[0]

  result = []
  while True:
    # Compare with some threshold to avoid floating point errors
    if np.all(np.abs(X) < 1e-9):
      break
    G_X = positivity_graph(X)
    # Positivty graphs always have an 2n vertices.
    perfect_matching = maximum_cardinality_matching_bipartite(G_X, list(range(n)), list(range(n, n * 2)))
    P = np.zeros(X.shape)
    z = np.inf
    for (i, j) in perfect_matching:
      P[i, j - n] = 1
      z = min(z, X[i, j - n])
    X -= z * P
    result.append((z, P))
  return result

def positivity_graph(X: np.ndarray) -> Dict[int, List[int]]:
  """
  The positivity graph of a bistochastic matrix.

  Must run utils.check_square_matrix on A before calling this function.

  Parameters
  ----------
  X : np.ndarray
    A bistochastic matrix.

  Returns
  -------
  Dict[int, List[int]]
    A dictionary of the form {i: [j, k, ...]} where i is the index of a vertex and [j, k, ...] are the indices of the vertices that i is connected to.
    Here, vertices 1 to n represent the rows of A and vertices n + 1 to 2n represent the columns of A. (where n is the number of rows/columns of the square matrix A)
    The returned graph is an undirectional bipartite graph. There are edges from rows to columns and columns to rows.
  """
  n = X.shape[0]
  G_X = dict()
  for i in range(n):
    for j in range(n):
      if X[i, j] > 0:
        G_X[i] = G_X.get(i, []) + [j + n]
        G_X[j + n] = G_X.get(j + n, []) + [i]
  return G_X
