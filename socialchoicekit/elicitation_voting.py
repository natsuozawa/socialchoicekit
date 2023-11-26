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
    if self.tie_breaker == "random":
      return np.random.choice(winners)
    return winners

  def interactive_score(self) -> np.ndarray:
    # TODO: implement
    pass

  def interactive_scf(self) -> np.ndarray or int:
    # TODO: implement
    pass

  def _check_profile(self, profile) -> None:
    if isinstance(profile, np.ndarray):
      if np.ndim(profile) == 2:
        if np.amin(profile) == 1 and np.amax(profile) == profile.shape[1]:
          return
        raise ValueError("Profile must contain exactly integers from 1 to M")
      raise ValueError("Profile must be a two-dimensional array")
    # TODO: turn this into a common utils method and accept other formats
    raise ValueError("Profile is not in a recognized data format")

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
  lambda_: int
    The number of positions to query.

  tie_breaker : {"random", "accept"}
    - "random": pick from a uniform distribution among the winners
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
    self._check_valuation_profile(valuation_profile)
    score = self.score(valuation_profile)
    return super().scf(score)

  def _check_valuation_profile(self, valuation_profile):
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

  tie_breaker : {"random", "accept"}
    - "random": pick from a uniform distribution among the winners
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
    self._check_profile(profile)
    self._check_valuation_profile(valuation_profile)
    return np.nansum(valuation_profile, axis=0)

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
    self._check_profile(profile)
    self._check_valuation_profile(valuation_profile)
    score = self.score(profile, valuation_profile)
    return super().scf(score)
