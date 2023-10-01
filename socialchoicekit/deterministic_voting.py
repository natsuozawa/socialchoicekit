import numpy as np

class Plurality:
  """
  A plurality ballot names a single, most-preferred alternative, and the plurality voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with a plurality (greatest number) of votes. Alternately, we can identify a ranking with a plurality ballot for the top-ranked alternative (while we ignore the rest of the ranking).
  """
  def __init__(
    self,
    tie_breaker: str or None="random",
    zero_indexed: bool=False
  ) -> None:
    self.tie_breaker = tie_breaker
    self.index_fixer = 0 if zero_indexed else 1
    self._check_tie_breaker()

  def score(self, profile: np.ndarray) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Complexity
    O(MN)

    Parameters
    :type profile: np.ndarray

    A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns

    :return np.ndarray
    A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    self._check_profile(profile)
    scores_by_voter = np.where(profile == 1, 1, 0)
    return np.sum(scores_by_voter, axis=0)

  def swf(self, profile: np.ndarray) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Complexity
    O(MN + MlogM)

    Parameters
    :type profile: np.ndarray

    A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns

    :return np.ndarray
    A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    self._check_profile(profile)
    scores = self.score(profile)
    rank = np.argsort(-scores)
    return np.array([rank + self.index_fixer, scores[rank]])

  def scf(self, profile: np.ndarray) -> np.ndarray or int:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Complexity
    O(MN)

    Parameters
    :type profile: np.ndarray

    A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    :return np.ndarray or int
    """
    self._check_profile(profile)
    scores = self.score(profile)
    winners = np.argwhere(scores == np.amax(scores)).flatten()
    if self.tie_breaker == "random":
      return np.random.choice(winners)
    return winners

  def _check_profile(self, profile) -> None:
    if isinstance(profile, np.ndarray):
      if np.ndim(profile) == 2:
        return
      raise ValueError("Profile must be a two-dimensional array")
    # TODO: turn this into a common utils method and accept other formats
    raise ValueError("Profile is not in a recognized data format")

  def _check_tie_breaker(self) -> None:
    if self.tie_breaker in ["random", None]:
      return
    raise ValueError("Tie breaker is not recognized")
