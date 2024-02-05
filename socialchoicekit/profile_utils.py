import numpy as np

from socialchoicekit.utils import check_tie_breaker, check_profile

class Profile(np.ndarray):
  """
  The generic profile class. In the background, this is just a numpy array.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN.
  """
  def __init__(self):
    raise RuntimeError("Call the 'of' method")

  @staticmethod
  def of(arr: np.ndarray) -> "Profile":
    return arr.view(Profile)

class StrictProfile(Profile):
  """
  Profiles that do not allow ties.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictProfile":
    return arr.view(StrictProfile)

class ProfileWithTies(Profile):
  """
  Profiles that allow ties.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "ProfileWithTies":
    return arr.view(ProfileWithTies)

class CompleteProfile(Profile):
  """
  Profiles that do not have any NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteProfile":
    return arr.view(CompleteProfile)

class IncompleteProfile(Profile):
  """
  Profiles that have NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteProfile":
    return arr.view(IncompleteProfile)

class StrictCompleteProfile(StrictProfile, CompleteProfile):
  """
  Corresponds to SoC (Strict Orders - Complete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictCompleteProfile":
    return arr.view(StrictCompleteProfile)

class StrictIncompleteProfile(StrictProfile, IncompleteProfile):
  """
  Corresponds to SoI (Strict Orders - Incomplete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile does not allow ties (i.e., no two alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "StrictIncompleteProfile":
    return arr.view(StrictIncompleteProfile)

class CompleteProfileWithTies(ProfileWithTies, CompleteProfile):
  """
  Corresponds to ToC (Orders with Ties - Complete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteProfileWithTies":
    return arr.view(CompleteProfileWithTies)

class IncompleteProfileWithTies(ProfileWithTies, IncompleteProfile):
  """
  Corresponds to ToI (Orders with Ties - Incomplete List) in Preflib.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the rank of alternative or item j in the preference list of agent i. The rank is an integer, where 1 is the most preferred. If the rank is unknown or the item is unacceptable, the element would be NaN. The profile allows ties (i.e., two or more alternatives can have the same rank for an agent).
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteProfileWithTies":
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
    return arr.view(ValuationProfile)

class CompleteValuationProfile(ValuationProfile):
  """
  Valuation profiles that do not have any NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the utility value (agent's cardinal preference) for alternative or item j.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "CompleteValuationProfile":
    return arr.view(CompleteValuationProfile)

class IncompleteValuationProfile(ValuationProfile):
  """
  Valuation profiles that have NaN values.

  An (N, M) array, where N is the number of agents and M is the number of alternatives or items. The element at (i, j) indicates the utility value (agent's cardinal preference) for alternative or item j. If the value is unknown or the item is unacceptable, the element would be NaN.
  """
  @staticmethod
  def of(arr: np.ndarray) -> "IncompleteValuationProfile":
    return arr.view(IncompleteValuationProfile)
