import numpy as np
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse import csr_matrix

from socialchoicekit.utils import check_square_matrix
from socialchoicekit.profile_utils import ValuationProfile, Profile

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
    valuation_profile: ValuationProfile,
  ) -> np.ndarray:
    """
    The (provisional) social choice function, which takes in a valuation profile and returns an allocation.

    Parameters
    ----------
    valuation_profile: ValuationProfile
      This is the (complete) cardinal profile. A (N, N) array, where N is the number of agents and also the number of items. The element at (i, j) indicates the utility value (agent's cardinal preference) for item j. If agent i finds item j unacceptable, the element would be np.nan

    Returns
    -------
    allocation: np.ndarray
      This is the allocation. A (N,) array, where N is the number of items. Agent i is assigned to element i.
    """
    check_square_matrix(valuation_profile)

    biadjacency_matrix = csr_matrix(np.where(np.isnan(valuation_profile), 0, valuation_profile))
    _, col_ind = min_weight_full_bipartite_matching(biadjacency_matrix, maximize=True)
    return col_ind + self.index_fixer

def root_n_serial_dictatorship(
  profile: Profile
) -> np.ndarray:
  """
  Root n serial dictatorship is a subroutine used in the Match-TwoQueries routine [ABFV2022a]_ for elicitation allocation. This does not compute an approriate allocation. Instead, it generates a 'sufficiently representative assignment'.

  The algorithm assigns at most root n agents to each item based on a serial dictatorship algorithm. The algorithm is deterministic - hence, the order of the agents matters.

  Note that the return values are 0-indexed.

  Parameters
  ----------
  profile: Profile
    A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the agent's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

  Returns
  -------
  np.ndarray
    A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
  """
  n = profile.shape[0]
  m = profile.shape[1]

  ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)
  # Element j is the number of times item j was allocated so far.
  allocation_count = np.zeros(m)
  # Element i is the item that agent i is allocated to.
  allocation = np.full(n, np.nan)

  for agent in range(n):
    for alternative in ranked_profile[agent]:
      if allocation_count[alternative] < np.sqrt(n):
        allocation_count[alternative] += 1
        allocation[agent] = alternative
        break
    if allocation[agent] == np.nan:
      # This is possible if profile has a lot of NaNs.
      raise ValueError("No allocation found")
  return allocation.astype(int)
