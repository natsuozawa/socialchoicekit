import numpy as np

from socialchoicekit.deterministic_scoring import Plurality
from socialchoicekit.utils import check_tie_breaker, break_tie
from socialchoicekit.profile_utils import CompleteProfile

"""
Deterministic multiround rules for voting. Definition and explanation taken from the Handbook of Computational Social Choice (Brandt, et al. 2016).
"""

class SingleTransferableVote:
  """
  Alternative Vote, Hare (Hare, 1859), Single Transferable Vote (STV), Instant Run-off Voting (IRV), and Ranked Choice Voting (RCV)—and proceeds as follows: at each stage, the alternative with lowest plurality score is dropped from all ballots, and at the first stage for which some alternative x sits atop a majority of the ballots, x is declared the winner.

  Parameters
  ----------

  tie_breaker : {"random", "first"}
    Tie breaker used to drop alternatives, not to select winning alternatives.
    - "random": pick from a uniform distribution among the losers to drop
    - "first": pick the alternative with the lowest index

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
    # TODO: customize this variable.
    self.voting_rule = Plurality(zero_indexed=zero_indexed)
    check_tie_breaker(tie_breaker, include_accept=False)

  def scf(self, profile: CompleteProfile) -> int:
    """
    The social choice function for this voting rule. Returns a single winning alternative.

    Notes
    -----
    Complexity O(MN)

    Parameters
    ----------
    profile: CompleteProfile
      A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative.

    Returns
    -------
    int
      A single winning alternative.
    """
    current_profile = profile
    alternatives = np.arange(profile.shape[1]) + self.index_fixer
    while True:
      score = self.voting_rule.score(CompleteProfile.of(current_profile))
      if alternatives.shape[0] == 1:
        break
      # Access the first element here because np.where returns a tuple.
      candidate_alternatives_to_drop = np.where(score == np.amin(score))[0]
      alternative_to_drop = break_tie(candidate_alternatives_to_drop, self.tie_breaker, include_accept=False)
      dropped_row = np.reshape(current_profile[:, alternative_to_drop], (profile.shape[0], 1))
      current_profile = np.delete(current_profile, alternative_to_drop, axis=1)
      current_profile = np.where(current_profile > dropped_row, current_profile - 1, current_profile)
      alternatives = np.delete(alternatives, alternative_to_drop)
    return alternatives[0]
