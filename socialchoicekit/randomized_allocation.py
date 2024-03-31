import numpy as np

from socialchoicekit.bistochastic import birkhoff_von_neumann
from socialchoicekit.profile_utils import StrictProfile

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

  def scf(self, profile: StrictProfile) -> np.ndarray:
    """
    The social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    profile: StrictProfile
      A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    pref = np.array(profile.view(np.ndarray))
    allocation = np.full(profile.shape[0], np.nan)

    order = np.arange(pref.shape[0])
    np.random.shuffle(order)

    for agent in order:
      if np.all(np.isnan(pref[agent])):
        continue
      item = np.nanargmin(pref[agent])
      allocation[agent] = int(item) + self.index_fixer
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
    profile: StrictProfile,
    speeds: np.ndarray
  ) -> np.ndarray:
    """
    The bistochastic matrix outputted by this voting rule on a preference profile. This bistochastic matrix can be decomposed with the Birkhoff von Neumann algorithm (implemented in bistochastic.birkhoff_von_neumann) to a convex combination of permuation matrices.

    Parameters
    ----------
    profile: StrictProfile
      An (N, N) array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    speeds: np.ndarray
      A N-array, where N is the number of agents. The element at i indicates the speed of agent i. The speed of an agent is the number of items that the agent can eat in one time unit.

    Returns
    -------
    np.ndarray
      A bistochastic matrix.
    """

    n = profile.shape[0]

    # Element at (i, j) is agent i's j+1th most preferred item (0-indexed alternative number)
    ranked_items = np.argsort(profile, axis=1).view(np.ndarray)
    # Element at i is the position of the item in ranked_items that agent i is eating. If agent has nothing else to eat, the element would be np.nan.
    current_position = np.zeros(n)
    # Element at j is the fraction of item j that is remaining. If the item is completely eaten, the element would be np.nan.
    item_fraction_remaining = np.ones(n)
    # Element at i is the amount of items in total that agent i is eaten. If the agent has finished eating, the value would be np.nan.
    agent_amount_eaten = np.zeros(n)

    bistochastic = np.zeros((n, n))

    while True:
      if np.all(np.isnan(item_fraction_remaining)) or np.all(np.ones(n) <= agent_amount_eaten):
        break

      # Element at i is the current item that agent i is eating.
      # If there is nothing that the agent can eat, the agent would try to eat their most preferred item (without success).
      # This avoids corner cases.
      current_item = np.where(
        np.isnan(current_position),
        np.nan,
        ranked_items[np.arange(n), np.where(np.isnan(current_position), 0, current_position).astype(int)])
      # Element at j is the total speed of agents that are currently eating item j
      total_speeds = np.array([np.sum(speeds[current_item == j]) for j in range(n)])

      # TODO: do a capacity check here np.amax(time_until_completely_eaten * total_speeds, some kind of agg on speeds * (1 - amount_eaten)) < 1
      time_until_agent_finished = (1 - agent_amount_eaten) / speeds
      next_agent_to_finish = np.nanargmin(time_until_agent_finished)
      time_until_next_agent_finished = time_until_agent_finished[next_agent_to_finish]

      time_until_item_finished = item_fraction_remaining / total_speeds
      next_completely_eaten_item = np.nanargmin(time_until_item_finished)
      time_until_next_item_finished = time_until_item_finished[next_completely_eaten_item]

      t = min(time_until_next_agent_finished, time_until_next_item_finished)

      i_indices = np.where(~np.isnan(current_item))[0]
      j_indices = current_item[i_indices].astype(int)
      bistochastic[i_indices, j_indices] += t * speeds[i_indices]
      item_fraction_remaining -= total_speeds * t
      # Compare with some threshold to avoid floating point errors
      item_fraction_remaining = np.where(item_fraction_remaining > 1e-9, item_fraction_remaining, np.nan)
      agent_amount_eaten += speeds * t
      agent_amount_eaten = np.where(agent_amount_eaten < 1 - 1e-9, agent_amount_eaten, np.nan)

      for agent in range(n):
        # Eat the next preferred item that is available
        while current_position[agent] < n and np.isnan(item_fraction_remaining[ranked_items[agent, current_position[agent].astype(int)]]):
          current_position[agent] += 1
        # Theoretically, amount_eaten would never be > 1
        if current_position[agent] == n or np.isnan(agent_amount_eaten[agent]):
          current_position[agent] = np.nan
    return bistochastic

  def scf(
    self,
    profile: StrictProfile,
    speeds: np.ndarray
  ) -> np.ndarray:
    """
    The social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    profile: StrictProfile
      An (N, N) array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    speeds: np.ndarray
      A N-array, where N is the number of agents. The element at i indicates the speed of agent i. The speed of an agent is the number of items that the agent can eat in one time unit.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    bistochastic = self.bistochastic(profile, speeds)
    decomposition = birkhoff_von_neumann(bistochastic)
    permutation_probabilities = [p for p, _ in decomposition]
    chosen_permutation = decomposition[np.random.choice(1, len(permutation_probabilities), p=np.array(permutation_probabilities))][1]
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

  def bistochastic(self, profile: StrictProfile) -> np.ndarray:
    """
    The bistochastic matrix outputted by this voting rule on a preference profile. This bistochastic matrix can be decomposed with the Birkhoff von Neumann algorithm (implemented in bistochastic.birkhoff_von_neumann) to a convex combination of permuation matrices.

    Parameters
    ----------
    profile: StrictProfile
      An (N, N) array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A bistochastic matrix.
    """
    return self.simultaneous_eating.bistochastic(profile, np.ones(profile.shape[0]))

  def scf(self, profile: StrictProfile) -> np.ndarray:
    """
    The social choice function for this voting rule. Returns at most one item allocated for each agent.

    Parameters
    ----------
    profile: StrictProfile
      An (N, N) array, where M is the number of items. The element at (i, j) indicates the voter's preference for item j, where 1 is the most preferred item. If the agent finds an item unacceptable, the element would be np.nan.

    Returns
    -------
    np.ndarray
      A numpy array containing the allocated item for each agent or np.nan if the agent is unallocated.
    """
    return self.simultaneous_eating.scf(profile, np.ones(profile.shape[0]))
