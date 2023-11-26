import numpy as np

from socialchoicekit.utils import check_tie_breaker, check_profile, check_valuation_profile, break_tie

class BaseElicitationVoting:
  """
  The abstract base elicitation voting rule. This class should not be instantiated directly.

  While there is a tie-breaking mechanism for this class, it is only used to tie-break between alternatives that have the same score. It is not used to decide which alternative would be queried (if they have the same cardinal utility).

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.

  """
  def __init__(
      self,
      tie_breaker: str = "random",
      zero_indexed: bool = False
  ) -> None:
    self.tie_breaker = tie_breaker
    self.index_fixer = 0 if zero_indexed else 1
    check_tie_breaker(self.tie_breaker)

  def scf(self, score: np.ndarray) -> np.ndarray or int:
    """
    Common logic for the social choice function.

    Parameters
    ----------
    score: np.ndarray
      A M-array, where M is the number of alternatives. The ith element indicates the social welfare value for alternative i.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    winners = np.argwhere(score == np.amax(score)).flatten() + self.index_fixer
    break_tie(winners, self.tie_breaker)

  def interactive_score(self) -> np.ndarray:
    # TODO: implement
    pass

  def interactive_scf(self) -> np.ndarray or int:
    # TODO: implement
    pass

class LambdaPRV(BaseElicitationVoting):
  """
  Lambda-Prefix Range Voting (Amanatidis et al. 2021) is the most basic elicitation voting rule that queries every agent at the first lambda >= 1 positions.

  Parameters
  ----------
  lambda_: int
    The number of positions to query.

  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
      self,
      lambda_: int = 1,
      tie_breaker: str = "random",
      zero_indexed: bool = False
    ):
    super().__init__(tie_breaker, zero_indexed)
    self.lambda_ = lambda_

  def score(self, valuation_profile: np.ndarray):
    """
    Parameters
    ----------
    valuation_profile: np.ndarray
      This is the (partial) cardinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    self._check_valuation_profile(valuation_profile)
    vp = np.array(valuation_profile)
    # Column indices for the values that are not in the top lambda
    j_indices = np.argpartition(np.where(np.isnan(vp), 0, vp), -self.lambda_, axis=1)[:, :-self.lambda_].flatten()
    # Row indices for the values that are not in the top lambda
    i_indices = (np.arange(vp.shape[0]).reshape(-1, 1) * np.ones(vp.shape[1] - self.lambda_, dtype=int)).flatten()
    vp[(i_indices, j_indices)] = np.nan
    return np.nansum(vp, axis=0)

  def scf(self, valuation_profile: np.ndarray):
    """
    Parameters
    ----------
    valuation_profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(valuation_profile)
    return super().scf(score)

  def _check_valuation_profile(self, valuation_profile):
    check_valuation_profile(valuation_profile)
    super()._check_valuation_profile(valuation_profile)
    num_not_nan = np.sum(np.where(np.isnan(valuation_profile), 0, 1), axis=0)
    if np.amin(num_not_nan) < self.lambda_:
      raise ValueError("Profile doesn't contain enough cardinal information to compute the score.")

class KARV(BaseElicitationVoting):
  """
  k-Acceptable Range Voting (Amanatidis et al. 2021) is a generalization of Lambda-Prefix Range Voting that queries every agent at the first k positions.

  Parameters
  ----------
  k: int
    The number of positions to query.

  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """

  def __init__(
      self,
      k: int = 1,
      tie_breaker: str = "random",
      zero_indexed: bool = False
  ):
    super().__init__(tie_breaker, zero_indexed)
    self.k = k

  def score(self, profile: np.ndarray, valuation_profile: np.ndarray):
    """
    Parameters
    ----------
    profile: np.ndarray or None
     This is the ordinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    valuation_profile: np.ndarray
      This is the (partial) cardinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    check_profile(profile)
    check_valuation_profile(valuation_profile, is_complete=True)

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
    def binary_search(i: int, alpha: int, beta: int, lambda_: int, v: float):
      if beta - alpha <= 1:
        return alpha
      # This will never be more than m - 1, even if we start with beta = m.
      mid = (alpha + beta) // 2
      u = vp[i, ranked_alternatives[i, mid]]
      if u >= v / lambda_:
        return binary_search(i, mid, beta, lambda_, v)
      else:
        return binary_search(i, alpha, mid, lambda_, v)

    # Element at (i, j) is the simulated welfare of alternative j for agent i
    v_tilde = np.zeros((n, m))
    v_tilde[np.arange(n), ranked_alternatives[:, 0]] = v_favorite
    # Element at i is the least preferred alternative (0-indexed alternative number) in agent i's lambda-acceptable set
    S_prev = np.zeros(n)
    for l in range(1, self.k + 1):
      lambda_l = m ** (l / (self.k + 1))
      p_star = np.array([binary_search(i, 0, m, lambda_l, v_favorite[i]) for i in range(n)])
      j_indices = np.concatenate([ranked_alternatives[i, np.arange(S_prev[i] + 1, p_star[i] + 1, dtype=int)] for i in range(n)])
      i_indices = np.concatenate([np.ones(int(p_star[i] - S_prev[i]), dtype=int) * i for i in range(n)])
      v_tilde[(i_indices, j_indices)] = v_favorite[i_indices] / lambda_l
      S_prev = p_star

    return np.sum(v_tilde, axis=0)

  def scf(self, profile: np.ndarray, valuation_profile: np.ndarray):
    """
    Parameters
    ----------
    profile: np.ndarray or None
     This is the ordinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    valuation_profile: np.ndarray
      This is the (partial) cardinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile, valuation_profile)
    return super().scf(score)
