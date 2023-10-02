import numpy as np

class BaseScoring:
  """
  The abstract scoring rule. This class should not be instantiated directly.

  Parameters
  :type tie_breaker: str
  Accepts the following
  - "random": pick from a uniform distribution among the winners
  - "accept": return all winners in an array

  :type zero_indexed: bool
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

  def score(self, scores_by_voter: np.ndarray) -> np.ndarray:
    return np.sum(scores_by_voter, axis=0)

  def swf(self, score: np.ndarray) -> np.ndarray:
    rank = np.argsort(-score)
    return np.array([rank + self.index_fixer, score[rank]])

  def scf(self, score: np.ndarray) -> np.ndarray or int:
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

class Plurality(BaseScoring):
  """
  The plurality voting rule names a single, most-preferred alternative, and the plurality voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with a plurality (greatest number) of votes. Alternately, we can identify a ranking with a plurality ballot for the top-ranked alternative (while we ignore the rest of the ranking).
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

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
    return super().score(scores_by_voter)

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
    score = self.score(profile)
    return super().swf(score)

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
    score = self.score(profile)
    return super().scf(score)

class Borda(BaseScoring):
  """
  The Borda voting rule assigns the jth most preferred alternative a score of M-j, where M is the number of alternatives. The Borda voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest scores.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

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
    scores_by_voter = (profile.shape[1] - profile)
    return super().score(scores_by_voter)

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
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: np.ndarray) -> np.ndarray:
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
    score = self.score(profile)
    return super().scf(score)

class Veto(BaseScoring):
  """
  (Temporary description)
  The veto voting rule, also known as the anti-plurality rule, names a single, least-preferred alternative, and the veto voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the fewest vetoes.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

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
    scores_by_voter = np.where(profile < profile.shape[1], 1, 0)
    return super().score(scores_by_voter)

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
    score = self.score(profile)
    return super().swf(score)

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
    score = self.score(profile)
    return super().scf(score)

class KApproval(BaseScoring):
  """
  (Temporary description)
  The k-approval voting rule names the k most-preferred alternatives, and the k-approval voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest number of approvals.

  Parameters
  :type k: int
  A number greater than 0. If greater than or equal to M, the k-approval rule becomes trivial.
  """
  def __init__(self, k: int, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    if k < 1:
      raise ValueError("k must be greater than 0")
    self.k = k
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: np.ndarray) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Complexity
    O(MN)

    Parameters
    :type profile: np.ndarray

    A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates whether the voter approves of alternative j.

    Returns

    :return np.ndarray
    A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    self._check_profile(profile)
    scores_by_voter = np.where(profile <= self.k, 1, 0)
    return super().score(scores_by_voter)

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
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: np.ndarray) -> np.ndarray:
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
    score = self.score(profile)
    return super().scf(score)

class Harmonic(BaseScoring):
  """
  (Temporary description)
  The harmonic voting rule assigns the jth most preferred alternative a score of 1/j, The harmonic voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest scores.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: np.ndarray) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Complexity
    O(MN)

    Parameters
    :type scores_by_voter: np.ndarray

    A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the score for alternative j by voter i.

    Returns

    :return np.ndarray
    A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    self._check_profile(profile)
    scores_by_voter = 1 / profile
    return super().score(scores_by_voter)

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
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: np.ndarray) -> np.ndarray:
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
    score = self.score(profile)
    return super().scf(score)
