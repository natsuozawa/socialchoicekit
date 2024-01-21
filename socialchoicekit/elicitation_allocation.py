import numpy as np

from socialchoicekit.deterministic_allocation import MaximumWeightMatching
from socialchoicekit.utils import check_profile, check_valuation_profile

class BaseElicitationAllocation:
  """
  The abstract base elicitation allocation rule. This class should not be instantiated directly.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool = False
  ) -> None:
    self.index_fixer = 0 if zero_indexed else 1
    self.mwm = MaximumWeightMatching(zero_indexed=zero_indexed)

class LambdaTSF(BaseElicitationAllocation):
  """
  Lambda-Threshold Step Function (Amanatidis et al. 2022) is a generalization of K-Acceptable Range Voting (Amanatidis et a. 2021) for allocation. (K-ARV is available in elicitation_voting) The algorithm partitions alternatives into lambda + 1 sets for evey agent to create a simulated value function using binary search.

  Parameters
  ----------
  lambda_ : int
    The number of positions to query.

  zero_indexed : bool
    If True, the output of the social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    lambda_: int = 1,
    zero_indexed: bool = False,
  ):
    self.lambda_ = lambda_
    super().__init__(zero_indexed=zero_indexed)

  def scf(
    self,
    profile: np.ndarray,
    valuation_profile: np.ndarray,
  ) -> np.ndarray:
    """
    The (provisional) social choice function for this voting rule. Returns one item allocated for each agent.

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    valuation_profile: np.ndarray
      This is the (partial) cardinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    check_profile(profile, is_complete=False)
    check_valuation_profile(valuation_profile, is_complete=False)

    n = profile.shape[0]
    m = profile.shape[1]

    # Element at (i, j) is agent i's j+1th most preferred alternative (0-indexed alternative number)
    ranked_alternatives = np.argsort(profile, axis=1)
    vp = np.array(valuation_profile)
    # Element at i is agent i's favorite alternative
    v_favorite = vp[np.arange(n), ranked_alternatives[:, 0]]

    # We have this as an inner function because it currently needs to access the vp and ranked_alternatives arrays.
    # TODO: When we have the mechanism to do interactive querying, we will move this.
    # We've modified this binary search from the paper for our implementation.
    def binary_search(i: int, left: int, right: int, alpha: int, v: float):
      if right - left <= 1:
        return left
      # This will never be more than m - 1, even if we start with beta = m.
      mid = (right + left) // 2
      u = vp[i, ranked_alternatives[i, mid]]
      if u >= v / alpha:
        return binary_search(i, mid, right, alpha, v)
      else:
        return binary_search(i, left, mid, alpha, v)

    # Element at (i, j) is the simulated welfare of alternative j for agent i
    epsilon = 1e-5
    v_tilde = profile * 0 + epsilon
    v_tilde[np.arange(n), ranked_alternatives[:, 0]] = v_favorite
    # Element at i is the least preferred alternative (0-indexed alternative number) in agent i's lambda-acceptable set
    # Add a very small threshold to distinguish between unacceptable alterantives and alternatives that did not fit in any acceptable set.
    Q_prev = np.zeros(n)
    for l in range(1, self.lambda_ + 1):
      alpha_l = m ** (l / (self.lambda_ + 1))
      p_star = np.array([binary_search(i, 0, m, alpha_l, v_favorite[i]) for i in range(n)])
      j_indices = np.concatenate([ranked_alternatives[i, np.arange(Q_prev[i] + 1, p_star[i] + 1, dtype=int)] for i in range(n)])
      i_indices = np.concatenate([np.ones(int(p_star[i] - Q_prev[i]), dtype=int) * i for i in range(n)])
      v_tilde[(i_indices, j_indices)] = v_favorite[i_indices] / alpha_l
      Q_prev = p_star

    return self.mwm.scf(v_tilde)
