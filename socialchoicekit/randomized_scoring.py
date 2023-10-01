import numpy as np

from .deterministic_scoring import *

class BaseRandomizedScoring:
  """
  The abstract scoring rule. This class should not be instantiated directly.

  Parameters
  :type zero_indexed_output: bool
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
    return self.voting_rule.score(scores_by_voter)

  def scf(self, profile: np.ndarray) -> int:
    score = self.score(profile)
    return np.random.choice(np.arange(score.shape[0]), p=score/np.sum(score)) + self.index_fixer

class RandomizedPlurality(BaseRandomizedScoring):
  """
  The randomized plurality voting rule where each alternative has a probability of being selected proportional to its plurality score.

  Access the voting_rule object to access the deterministic plurality voting rule and its methods.
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
  :type k: int
  A number greater than 0. If greater than or equal to M, the k-approval rule becomes trivial.
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
  """
  def __init__(
    self,
    zero_indexed: bool=False
  ) -> None:
    voting_rule = Harmonic(zero_indexed=zero_indexed)
    super().__init__(voting_rule=voting_rule, zero_indexed=zero_indexed)
