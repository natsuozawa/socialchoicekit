import numpy as np

from typing import Union

from socialchoicekit.utils import check_valuation_profile, check_profile

class BaseValuationProfileGenerator:
  """
  The abstract cardinal utility generator. This class should not be instantiated directly.

  The data generator assumes that the utilities are generated for the normalized social choice setting.

  Parameters
  ----------
  seed: Union[int, None]
    The seed for the random number generator. If None, the random number generator will not be seeded.
  """
  def __init__(
    self,
    seed: Union[int, None] = None,
  ):
    self.seed = seed

  def generate(
    self,
    profile: np.ndarray,
  ) -> np.ndarray:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    -------
    profile: np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    raise NotImplementedError

class UniformValuationProfileGenerator(BaseValuationProfileGenerator):
  """
  A simple data generator that first generates a valuation profile (cardinal utilities) based on a uniform probability distribution, and assigns the normalized generated utilities to alternatives based on the inputted profile (ordinal utilities).

  Parameters
  ----------
  high: float
    The upper bound of the uniform distribution. 1 by default. Must be positive.
  low: float
    The lower bound of the uniform distribution. 0 by default. Must be positive.
  seed: Union[int, None]
    The seed for the random number generator. If None, the random number generator will not be seeded.
  """
  def __init__(
    self,
    high: float,
    low: float,
    seed: Union[int, None] = None,
  ):
    if high < low or low < 0:
      raise ValueError("Invalid high and/or low value(s).")
    self.high = high
    self.low = low
    super().__init__(seed=seed)

  def generate(
    self,
    profile: np.ndarray,
  ) -> np.ndarray:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    check_profile(profile, is_complete=False)

    np.random.seed(self.seed)

    n = profile.shape[0]
    m = profile.shape[1]

    ranked_profile = np.argsort(profile, axis=1)

    # Preserve np.nan
    ans = profile * 0

    for agent in range(n):
      num_not_nan = np.count_nonzero(~np.isnan(profile[agent]))
      utilities = np.random.uniform(size=num_not_nan, high=self.high, low=self.low)
      # Descending sort and ormalize
      utilities = np.sort(utilities)[::-1] / np.sum(utilities)
      for item_rank in range(m):
        item = ranked_profile[agent, item_rank]
        if np.isnan(profile[agent, item]):
          break
        ans[agent, item] = utilities[item_rank]
    return ans

class NormalValuationProfileGenerator(BaseValuationProfileGenerator):
  """
  A simple data generator that first generates a valuation profile (cardinal utilities) based on a normal probability distribution, and assigns the normalized generated utilities to alternatives based on the inputted profile (ordinal utilities).

  Parameters
  ----------
  mean: float
    The mean of the normal distribution. 0.5 by default.
  varaince: float
    The variance of the normal distribution. 0.2 by default. Any generated values below 0 will be clipped to 0.
  seed: Union[int, None]
    The seed for the random number generator. If None, the random number generator will not be seeded.
  """
  def __init__(
    self,
    mean: float,
    variance: float,
    seed: Union[int, None] = None,
  ):
    self.mean = mean
    self.variance = variance
    super().__init__(seed=seed)

  def generate(
    self,
    profile: np.ndarray,
  ):
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    ----------
    profile: np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    check_profile(profile, is_complete=False)

    np.random.seed(self.seed)

    n = profile.shape[0]
    m = profile.shape[1]

    ranked_profile = np.argsort(profile, axis=1)

    # Preserve np.nan
    ans = profile * 0

    for agent in range(n):
      num_not_nan = np.count_nonzero(~np.isnan(profile[agent]))
      utilities = np.random.normal(size=num_not_nan, loc=self.mean, scale=np.sqrt(self.variance))
      # Clip negative values to 0
      utilities = np.where(utilities < 0, 0, utilities)
      # Descending sort and normalize
      utilities = np.sort(utilities)[::-1] / np.sum(utilities)
      for item_rank in range(m):
        item = ranked_profile[agent, item_rank]
        if np.isnan(profile[agent, item]):
          break
        ans[agent, item] = utilities[item_rank]
    return ans

def compute_ordinal_profile(cardinal_utility: np.ndarray) -> np.ndarray:
  """
  Computes the ordinal utility from the inputted cardinal utility. The input cardinal utility does not need to be normalized or complete.

  Parameters
  ----------
  cardinal_utility: np.ndarray
    A (N, M) array, where N is the number of agents and M is the number of items or alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.

  Returns
  -------
  np.ndarray
    A (N, M) array, where N is the number of agents and M is the number of items or alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.
  """
  check_valuation_profile(cardinal_utility, is_complete=False)

  n = cardinal_utility.shape[0]
  m = cardinal_utility.shape[1]

  # Sort by descending with np.nan at end
  ranked_profile = np.argsort(cardinal_utility * -1, axis=1)

  # Preserve np.nan
  ans = cardinal_utility * 0
  for agent in range(n):
    for item_rank in range(m):
      # Preserve np.nan with +=
      ans[agent, ranked_profile[agent, item_rank]] += item_rank + 1
  return ans
