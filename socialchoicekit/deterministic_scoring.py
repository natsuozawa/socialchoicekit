import numpy as np
from typing import Union

from socialchoicekit.utils import check_tie_breaker, break_tie
from socialchoicekit.profile_utils import Profile, CompleteProfile, StrictCompleteProfile, ValuationProfile

"""
Deterministic scoring rules for voting. Definition and explanation taken from the Handbook of Computational Social Choice [BCELP2016]_.
"""

class BaseScoring:
  """
  The abstract scoring rule. This class should not be instantiated directly.

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
    tie_breaker: str="random",
    zero_indexed: bool=False
  ) -> None:
    self.tie_breaker = tie_breaker
    self.index_fixer = 0 if zero_indexed else 1
    check_tie_breaker(self.tie_breaker)

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

  def score(self, scores_by_voter: np.ndarray) -> np.ndarray:
    """
    Common logic for the computing the score.

    Parameters
    ----------
    scores_by_voter: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the score calculated from the voter's preference for alternative j.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    return np.sum(scores_by_voter, axis=0)

  def scf(self, score: np.ndarray) -> Union[np.ndarray, int]:
    """
    Common logic for computing the social choice function.

    Parameters
    ----------
    score : np.ndarray
      A two dimensional (N, M) numpy array where N is the number of alternatives and M is the number of voters. The element at (i, j) indicates the score arising from voter i's ordering of alternative j. Obtain this array by calling a score function in a subclass.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    winners = np.argwhere(score == np.amax(score)).flatten() + self.index_fixer
    return break_tie(winners, self.tie_breaker)

class Plurality(BaseScoring):
  """
  The plurality voting rule names a single, most-preferred alternative, and the plurality voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with a plurality (greatest number) of votes. Alternately, we can identify a ranking with a plurality ballot for the top-ranked alternative (while we ignore the rest of the ranking).

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: Profile) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: Profile
    A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's preference for alternative j, where 1 is the most preferred alternative. If the agent's preference for the alternative is unknown or the alternative is unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    scores_by_voter = np.where(profile.view(np.ndarray) == 1, 1, 0)
    return super().score(scores_by_voter)

  def swf(self, profile: Profile) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(MN + MlogM)

    Parameters
    ----------
    profile: Profile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's preference for alternative j, where 1 is the most preferred alternative. If the agent's preference for the alternative is unknown or the alternative is unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: Profile) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: Profile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's preference for alternative j, where 1 is the most preferred alternative. If the agent's preference for the alternative is unknown or the alternative is unacceptable, the element would be np.nan.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile)
    return super().scf(score)

class Borda(BaseScoring):
  """
  The Borda voting rule assigns the jth most preferred alternative a score of M-j, where M is the number of alternatives. The Borda voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest scores.

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: CompleteProfile) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    scores_by_voter = (profile.shape[1] - profile.view(np.ndarray))
    return super().score(scores_by_voter)

  def swf(self, profile: CompleteProfile) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(MN + MlogM)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: CompleteProfile) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile)
    return super().scf(score)

class Veto(BaseScoring):
  """
  The veto voting rule, also known as the anti-plurality rule, names a single, least-preferred alternative, and the veto voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the fewest vetoes.

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: StrictCompleteProfile) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    # TODO: support all Profiles
    scores_by_voter = np.where(profile.view(np.ndarray) < profile.shape[1], 1, 0)
    return super().score(scores_by_voter)

  def swf(self, profile: StrictCompleteProfile) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(MN + MlogM)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: StrictCompleteProfile) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile)
    return super().scf(score)

class KApproval(BaseScoring):
  """
  The k-approval voting rule names the k most-preferred alternatives, and the k-approval voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest number of approvals.

  Parameters
  ----------
  k : int
    A number greater than 0. If greater than or equal to M, the k-approval rule becomes trivial.

  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, k: int, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    if k < 1:
      raise ValueError("k must be greater than 0")
    self.k = k
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: StrictCompleteProfile) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    scores_by_voter = np.where(profile.view(np.ndarray) <= self.k, 1, 0)
    return super().score(scores_by_voter)

  def swf(self, profile: StrictCompleteProfile) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(MN + MlogM)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: StrictCompleteProfile) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: StrictCompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile)
    return super().scf(score)

class Harmonic(BaseScoring):
  """
  (Temporary description)
  The harmonic voting rule assigns the jth most preferred alternative a score of 1/j, The harmonic voting rule then selects, as the winner(s) of an election (aka the “social choice(s)”) the alternative(s) with the highest scores.

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(self, tie_breaker: str = "random", zero_indexed: bool = False) -> None:
    super().__init__(tie_breaker, zero_indexed)

  def score(self, profile: CompleteProfile) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """
    scores_by_voter = 1 / profile.view(np.ndarray)
    return super().score(scores_by_voter)

  def swf(self, profile: CompleteProfile) -> np.ndarray:
    """
    The social welfare function for this voting rule. Returns a ranked list of alternatives with the scores. Note that tie breaking behavior is undefined.

    Notes
    -----
    Complexity O(MN + MlogM)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    np.ndarray
      A (2, M) array of scores where the element at (0, j) indicates the alternative number for the jth alternative and (1, j) indictes the score for the jth alternative.
    """
    score = self.score(profile)
    return super().swf(score)

  def scf(self, profile: CompleteProfile) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(profile)
    return super().scf(score)

class SocialWelfare(BaseScoring):
  """
  Computes the social welfare for alternatives based on an inputted valuation profile.

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    - "random": pick from a uniform distribution among the winners
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  zero_indexed : bool
    If True, the output of thesocial choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    tie_breaker: str="random",
    zero_indexed: bool=False,
  ):
    super().__init__(tie_breaker, zero_indexed)

  def score(
    self,
    valuation_profile: ValuationProfile,
  ) -> np.ndarray:
    """
    Computes the normalized social welfare for each alternative based on an inputted valuation profile.
    Note that the valuation profile has to be a cardinal profile, not an ordinal profile.

    Parameters
    ----------
    valuation_profile: ValuationProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (M,) array, where M is the number of alternatives. The element at (i,) indicates the social welfare of alternative i.
    """
    return np.nansum(valuation_profile, axis=0) / np.nansum(valuation_profile)

  def scf(
    self,
    valuation_profile: ValuationProfile,
  ) -> Union[np.ndarray, int]:
    """
    The social choice function for this voting rule. Returns a set of alternatives with the highest scores. With a tie breaking rule, returns a single alternative.

    Parameters
    ----------
    valuation_profile: ValuationProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    Union[np.ndarray, int]
      A numpy array of the winning alternative(s) or a single winning alternative.
    """
    score = self.score(valuation_profile)
    return super().scf(score)
