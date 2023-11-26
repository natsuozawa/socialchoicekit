import numpy as np

class BaseElicitationVoting:
  """
  The abstract base elicitation voting rule. This class should not be instantiated directly.

  While there is a tie-breaking mechanism for this class, it is only used to tie-break between alternatives that have the same score. It is not used to decide which alternative would be queried (if they have the same cardinal utility).

  Parameters
  ----------
  tie_breaker : {"random", "accept"}
    - "random": pick from a uniform distribution among the winners
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
    self._check_tie_breaker()

  def score(self, valuation_profile: np.ndarray) -> np.ndarray or int:
    """
    Common logic for the computing the score.

    Parameters
    ----------
    valuation_profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    pass

  def scf(self, valuation_profile: np.ndarray) -> np.ndarray or int:
    """
    Common logic for the social choice function.

    Parameters
    ----------
    valuation_profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    pass

  def interactive_score(self) -> np.ndarray:
    # TODO: implement
    pass

  def interactive_scf(self) -> np.ndarray or int:
    # TODO: implement
    pass

  def _check_valuation_profile(self, valuation_profile) -> None:
    # TODO: extract this commong logic (also appears in _check_profile to a separate function)
    if isinstance(valuation_profile, np.ndarray):
      if np.ndim(valuation_profile) == 2:
        pass
      raise ValueError("Profile must be a two-dimensional array")
    # TODO: turn this into a common utils method and accept other formats
    raise ValueError("Profile is not in a recognized data format")

  def _check_tie_breaker(self) -> None:
    # TOODO: move this out into a common utils method
    if self.tie_breaker in ["random", "accept"]:
      return
    raise ValueError("Tie breaker is not recognized")


class LambdaPRV(BaseElicitationVoting):
  """
  Lambda-Prefix Range Voting (Amanatidis et al. 2021) is the most basic elicitation voting rule that queries every agent at the first lambda >= 1 positions.

  Parameters
  ----------
  tie_breaker : {"random", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.

  lambda_: int
    The number of positions to query.
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
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    vp = np.array(valuation_profile)
    i_indices = np.argpartition(np.where(np.isnan(vp), 0, vp), -self.lambda_, axis=1)[:, :-self.lambda_].flatten()
    j_indices = (np.arange(vp.shape[0]).reshape(-1, 1) * np.ones(vp.shape[1] - self.lambda_, dtype=int)).flatten()
    vp[(j_indices, i_indices)] = np.nan
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
    return self.score(valuation_profile)

  def _check_valuation_profile(self, valuation_profile):
    super()._check_valuation_profile(valuation_profile)
    num_not_nan = np.sum(np.where(np.isnan(valuation_profile), 0, 1), axis=0)
    if np.amin(num_not_nan) < self.lambda_:
      raise ValueError("Profile doesn't contain enough cardinal information to compute the score.")

class KARV(BaseElicitationVoting):
  """
  k-Acceptable Range Voting (Amanatidis et al. 2021) is a generalization of Lambda-Prefix Range Voting that queries every agent at the first k positions.
  """

  def __init__(self, k: int = 1):
    """
    Parameters
    ----------
    k: int
      The number of positions to query.
    """
    super().__init__()
    self.k = k

  def score(self, valuation_profile: np.ndarray):
    """
    Parameters
    ----------
    valuation_profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    return np.nansum(valuation_profile, axis=0)

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
    return self.score(valuation_profile)
