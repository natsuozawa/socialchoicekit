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
    n = preference_list.shape[0]
    m = preference_list.shape[1]

    # Element at (i, j) is agent i's j+1th most preferred item (0-indexed alternative number)
    ranked_items = np.argsort(preference_list, axis=1)
    # Element at i is the position of the item in ranked_items that agent i is eating. If agent has nothing else to eat, the element would be np.nan.
    current_position = np.zeros(n)
    # Element at j is the fraction of item j that is remaining. If the item is completely eaten, the element would be np.nan.
    fraction_remaining = np.ones(m)

    bistochastic = np.zeros((n, m))

    while True:
      if np.all(np.isnan(fraction_remaining)):
        break

      # Element at i is the current item that agent i is eating.
      # If there is nothing that the agent can eat, the agent would try to eat their most preferred item (without success).
      # This avoids corner cases.
      current_item = np.where(np.isnan(current_position), ranked_items[:, 0], ranked_items[np.arange(n), current_position.astype(int)])
      # Element at j is the total speed of agents that are currently eating item j
      total_speeds = np.array([np.sum(speeds[current_item == j]) for j in range(m)])

      time_until_completely_eaten = fraction_remaining / total_speeds
      next_completely_eaten_item = np.nanargmin(time_until_completely_eaten)
      time_until_next_item_completely_eaten = time_until_completely_eaten[next_completely_eaten_item]

      bistochastic[np.arange(n), current_item] += np.where(np.isnan(current_position), 0, time_until_next_item_completely_eaten * speeds)
      fraction_remaining = fraction_remaining - total_speeds * time_until_next_item_completely_eaten
      # Compare with some threshold to avoid floating point errors
      fraction_remaining = np.where(fraction_remaining > 1e-9, fraction_remaining, np.nan)

      for agent in range(n):
        while current_position[agent] < m and np.isnan(fraction_remaining[ranked_items[agent, current_position[agent].astype(int)]]):
          current_position[agent] += 1
        if current_position[agent] == m:
          current_position[agent] = np.nan
    return bistochastic

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
    bistochastic = self.bistochastic(preference_list, speeds)
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
