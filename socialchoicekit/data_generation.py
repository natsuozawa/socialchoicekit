import numpy as np
from ortools.linear_solver.pywraplp import Solver

from typing import Union

from socialchoicekit.profile_utils import Profile, StrictProfile, ValuationProfile, CompleteValuationProfile

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
    profile: Profile,
  ) -> ValuationProfile:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    -------
    profile: StrictProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    ValuationProfile
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
    profile: StrictProfile,
  ) -> ValuationProfile:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    ----------
    profile: StrictProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    ValuationProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    if self.seed is not None:
      np.random.seed(self.seed)

    n = profile.shape[0]
    m = profile.shape[1]

    ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)

    # Preserve np.nan
    ans = profile.view(np.ndarray) * 0.0

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
    return ValuationProfile.of(ans)

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
    profile: StrictProfile,
  ) -> ValuationProfile:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    ----------
    profile: StrictProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    ValuationProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    if self.seed is not None:
      np.random.seed(self.seed)

    n = profile.shape[0]
    m = profile.shape[1]

    ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)

    # Preserve np.nan
    ans = profile.view(np.ndarray) * 0.0

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
    return ValuationProfile.of(ans)

class WorstDistortionValuationProfileGenerator(BaseValuationProfileGenerator):
  """
  Generates a cardinal profile consistent with the original ordinal profile that maximizes distortion for voting.
  This uses the method introduced in Ebadian et al. (2024)
  """
  def __init__(
    self,
  ):
    super().__init__()

  def generate(
    self,
    profile: StrictProfile,
  ) -> CompleteValuationProfile:
    """
    Generates a cardinal profile based on the inputted ordinal profile.

    Parameters
    ----------
    profile: StrictProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's ordinal utility for alternative j, where 1 is the most preferred alternative and M is the least preferred alternative. If the agent finds an item or alternative unacceptable, the element would be np.nan.

    Returns
    -------
    ValuationProfile
      A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's cardinal utility for alternative j. If the agent finds an item or alternative unacceptable, the element would be np.nan.
    """
    n = profile.shape[0]
    m = profile.shape[1]

    ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)

    solver = Solver.CreateSolver('GLOP')
    if not solver:
      raise Exception('Solver not found.')

    # Variable definitions
    p_hat = []
    for a in range(m):
      p_hat.append(solver.NumVar(0, solver.infinity(), f'p-hat_{a}'))
    delta = [[] for _ in range(n)]
    alpha = [[] for _ in range(n)]
    beta = [[] for _ in range(n)]
    for i in range(n):
      for r in range(m):
        delta[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'delta_{i}_{r}'))
        alpha[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'alpha_{i}_{r}'))
        beta[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'beta_{i}_{r}'))

    # Constraints
    # Main
    for i in range(n):
      # r in [2, m - 1] in 1-indexed is equal to [1, m - 1) is 0-indexed
      for r in range(1, m - 1):
        j = ranked_profile[i, r]
        # There will be no out of bounds error here because r >= 1
        solver.Add(delta[i][profile[i, j]] >= alpha[i][r - 1])
      for r in range(m):
        j = ranked_profile[i, r]
        solver.Add(delta[i][profile[i, j]] >= beta[i][r])
    for a in range(m):
      solver.Add(solver.Sum([delta[i][a] for i in range(n)]) <= 0)
    # Top partial maximums
    for i in range(n):
      for r in range(m):
        if r > 0:
          solver.Add(alpha[i][r] >= alpha[i][r - 1])
          # Rotate the 1/r to the other side to avoid array scalar multiplication
          solver.Add(r * alpha[i][r] >= solver.Sum([-p_hat[ranked_profile[i, l]] for l in range(r)]))
    # Bottom partial sums
    for i in range(n):
      for r in range(m):
        if r < m:
          solver.Add(beta[i][r] >= beta[i][r + 1])
          # Rotate the 1/r to the other side to avoid array scalar multiplication
          solver.Add(r * beta[i][r] >= solver.Sum([1 - p_hat[ranked_profile[i, l]] for l in range(r)]))

    # Variable ranges are hard coded in the variable definitions

    # Objective function
    solver.minimize(solver.Sum(p_hat))

    # Solve
    status = solver.Solve()

    if status != solver.OPTIMAL:
      raise Exception('The problem does not have an optimal solution.')

    # Retrieve values
    distortion = solver.Objective().value()
    p_hat_values = [p_hat[a].solution_value() for a in range(m)]

    # Create cardinal profile
    ans = np.zeros((n, m))

    # TODO: Look at section 3.3 of the paper
    return CompleteValuationProfile.of(np.array(ans))


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
