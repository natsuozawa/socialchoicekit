import numpy as np

from socialchoicekit.utils import check_tie_breaker, check_profile, check_valuation_profile

class Profile(np.ndarray):
  """
  The generic profile class. In the background, this is just a numpy array.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN.
  """
  def __init__(self):
    raise RuntimeError("Call the 'of' method")

  @staticmethod
  def of(arr: np.ndarray) -> "Profile":
    check_profile(arr, is_complete=False, is_strict=False)
    return arr.view(Profile)

class StrictProfile(Profile):
  """
  Profiles that do not allow ties.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictProfile":
    check_profile(arr, is_complete=False, is_strict=True)
    return arr.view(StrictProfile)

class ProfileWithTies(Profile):
  """
  Profiles that allow ties.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "ProfileWithTies":
    check_profile(arr, is_complete=False, is_strict=False)
    return arr.view(ProfileWithTies)

class CompleteProfile(Profile):
  """
  Profiles that do not have any NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteProfile":
    check_profile(arr, is_complete=True, is_strict=False)
    return arr.view(CompleteProfile)

class IncompleteProfile(Profile):
  """
  Profiles that have NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteProfile":
    check_profile(arr, is_complete=False, is_strict=False)
    return arr.view(IncompleteProfile)

class StrictCompleteProfile(StrictProfile, CompleteProfile):
  """
  Corresponds to SoC (Strict Orders - Complete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictCompleteProfile":
    check_profile(arr, is_complete=True, is_strict=True)
    return arr.view(StrictCompleteProfile)

class StrictIncompleteProfile(StrictProfile, IncompleteProfile):
  """
  Corresponds to SoI (Strict Orders - Incomplete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictIncompleteProfile":
    check_profile(arr, is_complete=False, is_strict=True)
    return arr.view(StrictIncompleteProfile)

class CompleteProfileWithTies(ProfileWithTies, CompleteProfile):
  """
  Corresponds to ToC (Orders with Ties - Complete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteProfileWithTies":
    check_profile(arr, is_complete=True, is_strict=False)
    return arr.view(CompleteProfileWithTies)

class IncompleteProfileWithTies(ProfileWithTies, IncompleteProfile):
  """
  Corresponds to ToI (Orders with Ties - Incomplete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteProfileWithTies":
    check_profile(arr, is_complete=False, is_strict=False)
    return arr.view(IncompleteProfileWithTies)

class ValuationProfile(np.ndarray):
  """
  The generic valuation profile class. In the background, this is just a numpy array.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the utility value (agent's cardinal preference) for alternative or item j. If the value is unknown or the item is unacceptable, the element would be NaN.
  """
  def __init__(self):
    raise RuntimeError("Call the 'of' method")

  @staticmethod
  def of(arr: np.ndarray) -> "ValuationProfile":
    check_valuation_profile(arr, is_complete=False)
    return arr.view(ValuationProfile)

class CompleteValuationProfile(ValuationProfile):
  """
  Valuation profiles that do not have any NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the utility value (agent's cardinal preference) for alternative or item j.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteValuationProfile":
    check_valuation_profile(arr, is_complete=True)
    return arr.view(CompleteValuationProfile)

class IncompleteValuationProfile(ValuationProfile):
  """
  Valuation profiles that have NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the utility value (agent's cardinal preference) for alternative or item j. If the value is unknown or the item is unacceptable, the element would be NaN.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteValuationProfile":
    check_valuation_profile(arr, is_complete=False)
    return arr.view(IncompleteValuationProfile)

def incomplete_profile_to_complete_profile(
  profile: Profile,
  tie_breaker: str = "random",
) -> CompleteProfile:
  """
  Converts an incomplete profile to a complete profile. np.nan values will be assigned a rank such that they are least preferred.

  Parameters
  ----------
  profile: Profile

  tie_breaker: {"random", "first", "accept"}
    - "random": shuffle np.nan items into a random order
    - "first": sort the np.nan items in ascending order
    - "accept": give all np.nan items the same rank - this results in a non-strict profile

  Returns
  -------
  StrictCompleteProfile
    if profile is StrictProfile and tie_breaker is not "accept"
  CompleteProfileWithTies
    otherwise
  """
  check_tie_breaker(tie_breaker, include_accept=True)
  check_profile(profile, is_complete=False, is_strict=False)
  n = profile.shape[0]
  m = profile.shape[1]
  complete_profile = np.array(profile)
  for i in range(n):
    nan_indices = np.where(np.isnan(profile[i]))[0]
    num_nan = len(nan_indices)
    if tie_breaker == "random":
      np.random.shuffle(nan_indices)
    elif tie_breaker == "first":
      nan_indices = np.sort(nan_indices)
    if tie_breaker == "accept":
      complete_profile[i, nan_indices] = m - num_nan + 1
    else:
      # np.arange is not inclusive of the second argument.
      complete_profile[i, nan_indices] = np.arange(m - num_nan + 1, m + 1)
  if tie_breaker != "accept" and isinstance(profile, StrictProfile):
    return StrictCompleteProfile.of(complete_profile)
  return CompleteProfileWithTies.of(complete_profile)

def profile_with_ties_to_strict_profile(
  profile: Profile,
  tie_breaker: str = "random",
):
  """
  Converts a profile with ties to a strict profile. If there are ties, the tie_breaker will be used to break the ties.

  Parameters
  ----------
  profile: Profile

  tie_breaker: {"random", "first"}
    - "random": shuffle the tied items into a random order
    - "first": sort the tied items in ascending order
    accept is not allowed.

  Returns
  -------
  StrictCompleteProfile
    if profile is CompleteProfile
  StrictIncompleteProfile
    otherwise
  """
  check_tie_breaker(tie_breaker, include_accept=False)
  check_profile(profile, is_complete=False, is_strict=False)
  n = profile.shape[0]
  m = profile.shape[1]
  strict_profile = np.array(profile)
  ranked_profile = np.argsort(profile, axis=1)
  for i in range(n):
    r = 0
    while r < m:
      k = 1
      while k < m - r and profile[i, ranked_profile[i, r + k]] == profile[i, ranked_profile[i, r]]:
        k += 1
      num_tied = k
      if num_tied > 1:
        # There is a tie.
        tied_indices = np.array([ranked_profile[i, r + j] for j in range(num_tied)])
        if tie_breaker == "random":
          np.random.shuffle(tied_indices)
        if tie_breaker == "first":
          tied_indices = np.sort(tied_indices)
        strict_profile[i, tied_indices] = np.arange(r + 1, r + num_tied + 1)

      r += num_tied
  if isinstance(profile, CompleteProfile):
    return StrictCompleteProfile.of(strict_profile)
  return StrictIncompleteProfile.of(strict_profile)

def compute_ordinal_profile(cardinal_profile: ValuationProfile) -> StrictProfile:
  """
  Computes the ordinal utility from the inputted cardinal utility. The input cardinal utility does not need to be normalized or complete.

  Parameters
  ----------
  cardinal_profile: ValuationProfile
    A (N, M) array, where N is the number of agents and M is the number of items or alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.

  Returns
  -------
  StrictProfile
    A (N, M) array, where N is the number of agents and M is the number of items or alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.
  """
  n = cardinal_profile.shape[0]
  m = cardinal_profile.shape[1]

  # Sort by descending with np.nan at end
  ranked_profile = np.argsort(cardinal_profile * -1, axis=1).view(np.ndarray)

  # Preserve np.nan
  ans = cardinal_profile.view(np.ndarray) * 0
  for agent in range(n):
    for item_rank in range(m):
      # Preserve np.nan with +=
      ans[agent, ranked_profile[agent, item_rank]] += item_rank + 1
  return StrictProfile.of(ans)
