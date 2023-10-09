import numpy as np

class BaseTournament:
  """
  The abstract tournament rule. This class should not be instantiated directly.

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
    tie_breaker: str="random",
    zero_indexed: bool=False
  ) -> None:
    self.tie_breaker = tie_breaker
    self.index_fixer = 0 if zero_indexed else 1
    self._check_tie_breaker()

  def swf(self, score: np.ndarray) -> np.ndarray:
    """
    Common logic for computing the social welfare function.

    Parameters
    ----------

    score : np.ndarray
      A two dimensional (N, M) numpy array where N is the number of alternatives and M is the number of voters. The element at (i, j) indicates the score arising from voter i's ordering of alternative j. Obtain this array by calling a score function in a subclass.

    Returns
    -------

    np.array
      A two dimensional (2, M) numpy array where the first row indicates the alternative number and the second row indicates the score.
    """
    rank = np.argsort(-score)
    return np.array([rank + self.index_fixer, score[rank]])

  def scf(self, score: np.ndarray) -> np.ndarray or int:
    """
    Common logic for computing the social choice function.

    Parameters
    ----------

    score : np.ndarray
      A two dimensional (N, M) numpy array where N is the number of alternatives and M is the number of voters. The element at (i, j) indicates the score arising from voter i's ordering of alternative j. Obtain this array by calling a score function in a subclass.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    winners = np.argwhere(score == np.amax(score)).flatten() + self.index_fixer
    if self.tie_breaker == "random":
      return np.random.choice(winners)
    return winners

  def _check_profile(self, profile) -> None:
    if isinstance(profile, np.ndarray):
      if np.ndim(profile) == 2:
        if np.amin(profile) == 1 and np.amax(profile) == profile.shape[1]:
          return
        raise ValueError("Profile must contain exactly integers from 1 to M")
      raise ValueError("Profile must be a two-dimensional array")
    # TODO: turn this into a common utils method and accept other formats
    raise ValueError("Profile is not in a recognized data format")

  def _check_tie_breaker(self) -> None:
    if self.tie_breaker in ["random", "accept"]:
      return
    raise ValueError("Tie breaker is not recognized")

class Copeland(BaseTournament):
  """
  Copeland rewards an alternative x for each pairwise victory x > y over an opponent and punishes her for each defeat, but disregards the margins of victory or defeat.

  Parameters
  ----------

  tie_breaker : {"random", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: np.ndarray) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(M^2N)

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    self._check_profile(profile)
    net_preferences = np.zeros((profile.shape[1], profile.shape[1]))
    for i in range(profile.shape[1]):
      preferences_by_voter = profile - profile[:, i].reshape(profile.shape[0], 1)
      preferences_by_voter = np.where(preferences_by_voter > 0, 1, preferences_by_voter)
      preferences_by_voter = np.where(preferences_by_voter < 0, -1, preferences_by_voter)
      preferences_by_voter = np.sum(preferences_by_voter, axis=0)
      net_preferences[i, :] = preferences_by_voter

    score = np.where(net_preferences > 0, 1, net_preferences)
    score = np.where(score < 0, -1, score)
    score = np.sum(score, axis=1).T

    return score

  def swf(self, profile: np.ndarray) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(M^N)

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    self._check_profile(profile)
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: np.ndarray) -> np.ndarray or int:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    -------
    np.ndarray or int
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    self._check_profile(profile)
    score = self.score(profile)
    return super().scf(score)
