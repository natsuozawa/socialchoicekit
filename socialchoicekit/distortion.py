import numpy as np

from typing import Union

from socialchoicekit.deterministic_scoring import SocialWelfare
from socialchoicekit.utils import check_valuation_profile
from socialchoicekit.profile_utils import ValuationProfile, incomplete_valuation_profile_to_complete_valuation_profile

def distortion(
  choice: Union[np.ndarray, int],
  valuation_profile: ValuationProfile,
) -> float:
  """
  This is a utility function to calculate distortion for voting as introduced by Procaccia and Rosenschein (2006)

  Distortion is the worst case ratio between the optimal utility obtainable from cardinal information and the optimal utility obtainable from an algorithm using limited preference information.

  distortion(f(P), v) = (max_{j in A} SW(j|v)) /  SW(f(P) | v)

  Parameters
  ----------
  choice : Union[np.ndarray, int]
    The choice (winner) made by the social choice function (scf) voting the voting rule that is being evaluated, based on limited preference information. Assumed to be 1-indexed.
    The type allows for the output of the scf method of a voting rule to be passed in directly. If multiple choices are given, this function chooses the choice that maximizes the distortion.

  valuation_profile : ValuationProfile
    A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the agent's value for item j. If the agent finds an item unacceptable or the agent's preference is unknown, the element would be np.nan.
    Any np.nan values will be treated as 0.
  """
  check_valuation_profile(valuation_profile, is_complete=False)
  complete_vp = incomplete_valuation_profile_to_complete_valuation_profile(valuation_profile)
  sw = SocialWelfare(tie_breaker="random")
  score = sw.score(complete_vp)
  if isinstance(choice, np.ndarray):
    return np.max(score) / np.min(score[choice - 1])
  return np.max(score) / score[choice - 1]

