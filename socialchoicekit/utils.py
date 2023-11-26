import numpy as np

def check_profile(profile: np.ndarray) -> None:
  """
  Checks that the profile is a numpy array with the correct dimensions.

  Parameters
  ----------
  profile: np.ndarray
    This is the ordinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the voter's preference for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative.

  Raises
  ------
  ValueError
    If the profile is not a numpy array
    If the profile is not two-dimensional.
    If the profile contains NaN values.
    If the profile contains values other than integers from 1 to M.
  """
  if isinstance(profile, np.ndarray):
    if np.ndim(profile) == 2:
      if np.isnan(np.sum(profile)):
        raise ValueError("Profile cannot contain NaN values")
      if np.amin(profile) == 1 and np.amax(profile) == profile.shape[1]:
        return
      raise ValueError("Profile must contain exactly integers from 1 to M")
    raise ValueError("Profile must be a two-dimensional array")
  raise ValueError("Profile is not in a recognized data format")

def check_valuation_profile(
    valuation_profile: np.ndarray,
    is_complete: bool = False
  ) -> None:
  """
  Checks that the valuation profile is a numpy array with the correct dimensions.

  Parameters
  ----------
    valuation_profile: np.ndarray
      This is the (partial) cardinal profile. A (N, M) array, where N is the number of voters and M is the number of alternatives. The element at (i, j) indicates the utility value (voter's cardinal preference) for alternative j. If the value is unknown, the element would be NaN.

    is_complete: bool
      If True, the valuation profile does not have any NaN values. If False, the valuation profile has NaN values.
  """
  if isinstance(valuation_profile, np.ndarray):
    if np.ndim(valuation_profile) == 2:
      if is_complete and np.isnan(np.sum(valuation_profile)):
        raise ValueError("Valuation profile cannot contain NaN values")
      return
    raise ValueError("Profile must be a two-dimensional array")
  raise ValueError("Profile is not in a recognized data format")


def check_tie_breaker(
  tie_breaker: str,
  include_accept: bool = True
) -> None:
  """
  Checks that the tie breaker is valid.

  Parameters
  ----------
  tie_breaker : {"random", "first", "accept"}
    The tie breaker to check.
    - "random": pick from a uniform distribution among the losers to drop
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  include_accept : bool
    If True, "accept" is a valid tie breaker. If False, "accept" is not a valid tie breaker.

  Raises
  ------
  ValueError
    If the tie breaker is not recognized.
  """
  if tie_breaker in ["random", "first"]:
    return
  if include_accept and tie_breaker in ["accept"]:
    return
  raise ValueError("Tie breaker is not recognized")

def break_tie(
  alternatives: np.ndarray,
  tie_breaker: str = "random",
  include_accept: bool = True
) -> int:
  """
  Breaks a tie among winning alternatives according to the tie breaker.

  Parameters
  ----------
  alternatives : np.ndarray
    The alternatives that are tied.

  tie_breaker : {"random", "first", "accept"}
    The tie breaker to use.
    - "random": pick from a uniform distribution among the losers to drop
    - "first": pick the alternative with the lowest index
    - "accept": return all winners in an array

  include_accept : bool
    If True, "accept" is a valid tie breaker. If False, "accept" is not a valid tie breaker.
  """
  if tie_breaker == "random":
    return np.random.choice(alternatives)
  elif tie_breaker == "first":
    return alternatives[0]
  elif tie_breaker == "accept" and include_accept:
    return alternatives
  else:
    raise ValueError("Tie breaker is not recognized")
