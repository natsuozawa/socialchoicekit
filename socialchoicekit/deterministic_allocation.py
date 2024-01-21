import numpy as np
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix

from socialchoicekit.utils import check_square_matrix, check_valuation_profile

class MaximumWeightMatching:
  """
  The maximum weight matching algorithm, which solves a special case of the minimum cost flow problem, finds an optimal matching between agents and items given the full cardinal utilities of the agents.

  Uses the scipy implementation of LAPJVsp algorithm.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool = False
  ) -> None:
    self.index_fixer = 0 if zero_indexed else 1

  def scf(
    self,
    valuation_profile: np.ndarray
  ) -> np.ndarray:
    """
    The (provisional) social choice function, which takes in a valuation profile and returns an allocation.

    Parameters
    ----------
    valuation_profile: np.ndarray
      This is the (complete) cardinal profile. A (N, N) array, where N is the number of agents and also the number of items. The element at (i, j) indicates the utility value (agent's cardinal preference) for item j. If agent i finds item j unacceptable, the element would be np.nan

    Returns
    -------
    allocation: np.ndarray
      This is the allocation. A (N,) array, where N is the number of items. Agent i is assigned to element i.
    """
    check_valuation_profile(valuation_profile, is_complete=False)
    check_square_matrix(valuation_profile)

    biadjacency_matrix = csr_matrix(np.where(np.isnan(valuation_profile), 0, valuation_profile))
    _, col_ind = min_weight_full_bipartite_matching(biadjacency_matrix, maximize=True)
    return col_ind + self.index_fixer
