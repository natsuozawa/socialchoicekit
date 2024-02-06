import numpy as np
from preflibtools.instances import OrdinalInstance

from typing import Union, Callable

from socialchoicekit.profile_utils import ValuationProfile

class Elicitor:
  """
  The Elicitor class responds to queries by the elicitation algorithms. This class is the base class and hence is not meant to be instantiated.

  Parameters
  ----------
  memoize: bool
    IF True, the elicitor will memoize the elicited values. If False, the elicitor may ask repeated questions. When the memoized value is referenced, the elicitation count will not change. True by default.

  zero_indexed : bool
    If True, the input of the elicit function will be zero-indexed. If False, the input will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    memoize: bool = True,
    zero_indexed: bool = False,
  ) -> None:
    self.elicitation_count = 0
    self.memoize = memoize
    if memoize:
      self.memoized_values = {}
    self.index_fixer = 0 if zero_indexed else 1

  def elicit(
    self,
    agent: int,
    alternative: int,
  ) -> float:
    """
    Returns the agent's preference for the alternative.

    Parameters
    ----------
    agent: int
      The agent's index.
    alternative: int
      The alternative's index.

    Returns
    -------
    float
      The agent's preference for the alternative.
    """
    agent += self.index_fixer
    alternative += self.index_fixer
    if self.memoize:
      memoized_value = self.memoized_values.get((agent, alternative))
      if memoized_value is not None:
        return memoized_value
    self.elicitation_count += 1
    elicited_value = self._elicit_impl(agent, alternative)
    if self.memoize:
      self.memoized_values[(agent, alternative)] = elicited_value
    return elicited_value

  def elicit_multiple(
      self,
      agents: np.ndarray,
      alternatives: np.ndarray,
  ) -> np.ndarray:
    """
    Given an agents array and an alternative array both of size N, returns an array of size N containing the elicited values.
    (The ith agent is elicited about the ith alternative.)

    Parameters
    ----------
    agents: np.ndarray
      The agents array. Must contain only integers that correspond to a valid agent.

    alternatives: np.ndarray
      The alternatives array. Must contain only integers that correspond to a valid alternative.

    Returns
    -------
    np.ndarray
      The elicited values. Size is the same as the size of the two input arrays.
    """
    if agents.shape != alternatives.shape:
      raise ValueError("The two input arrays must have the same shape.")

    if not (np.issubdtype(agents.dtype, np.integer) and np.issubdtype(alternatives.dtype, np.integer)):
      raise ValueError("The input arrays must contain only integers.")

    ans = []
    for agent, alternative in zip(agents, alternatives):
      ans.append(self.elicit(agent, alternative))

    return np.array(ans)

  def _elicit_impl(
    self,
    agent: int,
    alternative: int,
  ) -> float:
    # Override this method in the subclass
    raise NotImplementedError

class ValuationProfileElicitor(Elicitor):
  """
  Responds to queries from a valuation profile that is fully pre-populated.

  Parameters
  ----------
  valuation_profile: ValuationProfile
    This is the cardinal profile. A (N, M) array, where N is the number of agents and M is the number of alternatives. The element at (i, j) indicates the agent's preference for alternative j. If the agent finds an alternative unacceptable, the element would be np.nan.

  memoize: bool
    IF True, the elicitor will memoize the elicited values. If False, the elicitor may ask repeated questions. When the memoized value is referenced, the elicitation count will not change. True by default.
  """
  def __init__(
    self,
    valuation_profile: ValuationProfile,
    memoize: bool = True,
  ) -> None:
    self.valuation_profile = valuation_profile
    super().__init__(memoize=memoize, zero_indexed=True)

  def _elicit_impl(
    self,
    agent: int,
    alternative: int,
  ) -> float:
    return self.valuation_profile[agent, alternative]

class SynchronousStdInElicitor(Elicitor):
  """
  Responds to queries by reading each answer from the standard input synchronously.
  Outputs questions in English to the standard output.

  Parameters
  ----------
  zero_indexed : bool
    If True, the input of the elicit function will be zero-indexed. If False, the input will be one-indexed. One-indexed by default.

  memoize: bool
    IF True, the elicitor will memoize the elicited values. If False, the elicitor may ask repeated questions. When the memoized value is referenced, the elicitation count will not change. True by default.
  """
  def __init__(
    self,
    preflib_instance: Union[OrdinalInstance, None] = None,
    memoize: bool = True,
    zero_indexed: bool = False,
  ) -> None:
    self.preflib_instance = preflib_instance
    super().__init__(memoize=memoize, zero_indexed=zero_indexed)

  def _elicit_impl(
    self,
    agent: int,
    alternative: int,
  ) -> float:
    agent_name = agent
    alternative_name = alternative
    if self.preflib_instance is not None:
      alternative_name = self.preflib_instance.alternatives_name[alternative]
    print(f"Agent {agent_name}, what is your preference for alternative {alternative_name}?")
    return float(input())

class LambdaElicitor(Elicitor):
  """
  Responds to queries by calling a user-provided function.

  Parameters
  ----------
  elicitation_function: Callable[[int, int], float]
    A function that takes in the agent's index and the alternative's index and returns the agent's preference for the alternative.

  memoize: bool
    If True, the elicitor will memoize the elicited values. If False, the elicitor may ask repeated questions. When the memoized value is referenced, the elicitation count will not change. True by default.

  zero_indexed : bool
    If True, the input of the elicit function will be zero-indexed. If False, the input will be one-indexed. Zero-indexed by default.
  """
  def __init__(
    self,
    elicitation_function: Callable[[int, int], float],
    memoize: bool = True,
    zero_indexed: bool = True,
  ) -> None:
    self.elicitation_function = elicitation_function
    super().__init__(memoize=memoize, zero_indexed=zero_indexed)

  def _elicit_impl(
    self,
    agent: int,
    alternative: int,
  ) -> float:
    return self.elicitation_function(agent, alternative)
