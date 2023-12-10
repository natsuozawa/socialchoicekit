import numpy as np

from socialchoicekit.bistochastic import birkhoff_von_neumann

class RandomSerialDictatorship:
  """
  Random Serial Dictatorship (Bogomolnaia and Moulin 2001) selects a random agent to select their most preferred item, then selects a random agent from the remaining agents to select their most preferred item, and so on until all agents have selected an item.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
      self,
      zero_indexed: bool = False
  ) -> None:
    self.index_fixer = 0 if zero_indexed else 1

  def scf(self, preference_list: np.ndarray) -> np.ndarray:
    """
    The (provisional) social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    preference_list: np.ndarray
      A M-array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    pref = np.array(preference_list)
    allocation = np.full(preference_list.shape[0], np.nan)

    order = np.arange(pref.shape[0])
    np.random.shuffle(order)

    for agent in order:
      if np.all(np.isnan(pref[agent])):
        continue
      item = np.nanargmin(pref[agent])
      allocation[agent] = item + self.index_fixer
      pref[:, item] = np.nan

    return allocation

class SimultaneousEating:
  """
  Simultaneous Eating (Bogomolnaia and Moulin 2001) is an algorithm for fair random assignment (resource allocation) where the fraction that each agent receives an item in a simultaneous eating setting is translated to the probability that the agent is assigned an item in the resource allocation setting.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
      self,
      zero_indexed: bool = False
  ) -> None:
    self.index_fixer = 0 if zero_indexed else 1

  def bistochastic(
    self,
    preference_list: np.ndarray,
    speeds: np.ndarray
  ) -> np.ndarray:
    """
    The bistochastic matrix outputted by this voting rule on a preference list. This bistochastic matrix can be decomposed with the Birkhoff von Neumann algorithm (implemented in bistochastic.birkhoff_von_neumann) to a convex combination of permuation matrices.

    Parameters
    ----------
    preference_list: np.ndarray
      A M-array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    speeds: np.ndarray
      A N-array, where N is the number of agents. The element at i indicates the speed of agent i. The speed of an agent is the number of items that the agent can eat in one time unit.

    Returns
    -------
    np.ndarray
      A bistochastic matrix.
    """
    pass

  def scf(
    self,
    preference_list: np.ndarray,
    speeds: np.ndarray
  ) -> np.ndarray:
    """
    The (provisional) social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    preference_list: np.ndarray
      A M-array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    speeds: np.ndarray
      A N-array, where N is the number of agents. The element at i indicates the speed of agent i. The speed of an agent is the number of items that the agent can eat in one time unit.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    bistochastic = self.bistochastic(preference_list, np.ones(preference_list.shape[0]))
    decomposition = birkhoff_von_neumann(bistochastic)
    permutation_probabilities = [p for p, _ in decomposition]
    chosen_permutation = decomposition[np.random.choice(1, len(permutation_probabilities), p=permutation_probabilities)][1]
    return np.argmax(chosen_permutation, axis=1) + self.index_fixer

class ProbabilisticSerial:
  """
  Probabilistic Serial (Bogomolnaia and Moulin 2001) is a special case of the simultaneous eating algorithm where all agents have the same eating speed.

  Parameters
  ----------
  zero_indexed : bool
    If True, the output of the social welfare function and social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
      self,
      zero_indexed: bool = False
  ) -> None:
    self.simultaneous_eating = SimultaneousEating(zero_indexed=zero_indexed)

  def bistochastic(self, preference_list: np.ndarray) -> np.ndarray:
    """
    The bistochastic matrix outputted by this voting rule on a preference list. This bistochastic matrix can be decomposed with the Birkhoff von Neumann algorithm (implemented in bistochastic.birkhoff_von_neumann) to a convex combination of permuation matrices.

    Parameters
    ----------
    preference_list: np.ndarray
      A M-array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A bistochastic matrix.
    """
    return self.simultaneous_eating.bistochastic(preference_list, np.ones(preference_list.shape[0]))

  def scf(self, preference_list: np.ndarray) -> np.ndarray:
    """
    The (provisional) social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    preference_list: np.ndarray
      A M-array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    return self.simultaneous_eating.scf(preference_list, np.ones(preference_list.shape[0]))
