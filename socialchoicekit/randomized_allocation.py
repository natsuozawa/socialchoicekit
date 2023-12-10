import numpy as np

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
