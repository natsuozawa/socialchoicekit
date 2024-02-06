import numpy as np

from socialchoicekit.deterministic_scoring import *
from socialchoicekit.profile_utils import Profile, CompleteProfile, StrictCompleteProfile

"""
Randomized scoring rules for voting. Definition and explanation taken from the Handbook of Computational Social Choice (Brandt, et al. 2016).
"""

class BaseRandomizedScoring:
  """
  The abstract scoring rule. This class should not be instantiated directly.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    voting_rule: BaseScoring,
    zero_indexed: bool=False
  ) -> None:
    self.voting_rule = voting_rule
    self.index_fixer = 0 if zero_indexed else 1

  def score(self, scores_by_voter: np.ndarray) -> np.ndarray:
    """
    The scoring function for this voting rule. Returns a list of alternatives with their scores.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    -------
    np.ndarray
      A (1, M) array of scores where the element at (0, j) indicates the score for alternative j.
    """

    return self.voting_rule.score(scores_by_voter)

  def scf(self, profile: Profile) -> int:
    """
    The social choice function for this voting rule. Returns a single winning alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

    Returns
    -------
    int
      A single winning alternative.

    """
    score = self.score(profile)
    return np.random.choice(np.arange(score.shape[0]), p=score/np.sum(score)) + self.index_fixer

class RandomizedPlurality(BaseRandomizedScoring):
  """
  The randomized plurality voting rule where each alternative has a probability of being selected proportional to its plurality score.

  Access the voting_rule object to access the deterministic plurality voting rule and its methods.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = Plurality(zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)

class RandomizedBorda(BaseRandomizedScoring):
  """
  The randomized Borda voting rule where each alternative has a probability of being selected proportional to its Borda score.

  Access the voting_rule object to access the deterministic Borda voting rule and its methods.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = Borda(zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)

class RandomizedVeto(BaseRandomizedScoring):
  """
  The randomized veto (anti-plurality) voting rule where each alternative has a probability of being selected proportional to its anti-plurality score.

  Access the voting_rule object to access the deterministic veto voting rule and its methods.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = Veto(zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)

class RandomizedKApproval(BaseRandomizedScoring):
  """
  The randomized k-approval voting rule where each alternative has a probability of being selected proportional to its k-approval score.

  Access the voting_rule object to access the deterministic k-approval voting rule and its methods.

  Parameters
  ----------
  k: int
    A number greater than 0. If greater than or equal to M, the k-approval rule becomes trivial.

  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    k: int,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = KApproval(k=k, zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)

class RandomizedHarmonic(BaseRandomizedScoring):
  """
  The randomized harmonic voting rule where each alternative has a probability of being selected proportional to its harmonic score.

  Access the voting_rule object to access the deterministic harmonic voting rule and its methods.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = Harmonic(zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)
