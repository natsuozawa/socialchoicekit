import numpy as np

from preflibtools.instances import OrdinalInstance, CategoricalInstance

from socialchoicekit.utils import check_tie_breaker

def preflib_soc_to_profile(instance: OrdinalInstance) -> np.ndarray:
  """
  Convert a Preflib SoC (Strictly Orders - Complete List) to the profile (Numpy matrix) format.

  For details on Preflib SoC, see https://www.preflib.org/format

  Parameters
  ----------
  soc: OrdinalInstance
    The Preflib SoC to convert. This is included in the preflibtools.instances module. The data_type for this must be soc.

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib SoC.
  """
  if instance.data_type != "soc":
    raise ValueError("The inputted instance is not a SoC (Strictly Orders - Complete List) instance.")

  flattened_order = instance.flatten_strict()
  m = instance.num_alternatives
  arr = []
  for order, multiplicity in flattened_order:
    # Order: strict complete order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    indices = np.array(order) - 1
    preference = np.zeros(m, dtype=int)
    preference[indices] = np.arange(1, m + 1)
    for _ in range(multiplicity):
      arr.append(preference)
  return np.array(arr)

def preflib_soi_to_profile(instance: OrdinalInstance) -> np.ndarray:
  """
  Convert a Preflib SoI (Strictly Orders - Incomplete List) to the profile (Numpy matrix) format.

  For details on Preflib SoC, see https://www.preflib.org/format

  Parameters
  ----------
  soi: OrdinalInstance
    The Preflib SoI to convert. This is included in the preflibtools.instances module. The data_type for this must be soi.

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib SoC.
  """
  if instance.data_type != "soi":
    raise ValueError("The inputted instance is not a SoI (Strictly Orders - Incomplete List) instance.")

  # Note: this prints that we are using flatten_strict on a non-strict order but soi is (supposed to be) strict.
  print("Ignore the warning(s) below:")
  flattened_order = instance.flatten_strict()

  arr = []
  for order, multiplicity in flattened_order:
    indices = np.array(order) - 1
    preference = np.full(instance.num_alternatives, np.nan)
    preference[indices] = np.arange(1, len(o) + 1)
    # Order: strict incomplete order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    for _ in range(multiplicity):
      arr.append(preference)
  return np.array(arr)

def preflib_toc_to_profile(instance: OrdinalInstance, tie_breaker: str = "random") -> np.ndarray:
  """
  Convert a Preflib ToC (Orders with Ties - Complete List) to the profile (Numpy matrix) format.

  For details on Preflib ToC, see https://www.preflib.org/format

  Parameters
  ----------
  toc: OrdinalInstance
    The Preflib ToC to convert. This is included in the preflibtools.instances module. The data_type for this must be toc.

  tie_breaker : {"random", "first", "accept"}
    - "random": shuffle the tied items into a random order
    - "first": sort the tied items in ascending order
    - "accept": do not break ties

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib ToC.
  """
  if instance.data_type != "toc":
    raise ValueError("The inputted instance is not a ToC (Orders with Ties - Complete List) instance.")

  check_tie_breaker(tie_breaker, include_accept=True)

  vote_map = instance.vote_map()
  arr = []
  for order, multiplicity in vote_map.items():
    # Order: complete unflattened order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    preference = np.zeros(instance.num_alternatives, dtype=int)
    current_rank = 1
    for tied_items in order:
      tied_items = np.array(tied_items) - 1
      if tie_breaker == "accept":
        preference[tied_items] = current_rank
      else:
        if tie_breaker == "random":
          np.random.shuffle(tied_items)
        elif tie_breaker == "first":
          tied_items = np.sort(tied_items)
        preference[tied_items] = np.arange(current_rank, len(tied_items) + current_rank)
      current_rank += len(tied_items)
    for _ in range(multiplicity):
      arr.append(preference)
  return np.array(arr)

def preflib_toi_to_profile(instance: OrdinalInstance, tie_breaker: str = "random") -> np.ndarray:
  """
  Convert a Preflib ToI (Orders with Ties - Incomplete List) to the profile (Numpy matrix) format.

  For details on Preflib ToI, see https://www.preflib.org/format

  Parameters
  ----------
  toi: OrdinalInstance
    The Preflib ToI to convert. This is included in the preflibtools.instances module. The data_type for this must be toi.

  tie_breaker : {"random", "first", "accept"}
    - "random": shuffle the tied items into a random order
    - "first": sort the tied items in ascending order
    - "accept": do not break ties

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib ToI.
  """
  if instance.data_type != "toi":
    raise ValueError("The inputted instance is not a ToI (Orders with Ties - Incomplete List) instance.")

  vote_map = instance.vote_map()
  arr = []
  for order, multiplicity in vote_map.items():
    # Order: incomplete unflattened order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    preference = np.full(instance.num_alternatives, np.nan)
    current_rank = 1
    for tied_items in order:
      tied_items = np.array(tied_items) - 1
      if tie_breaker == "accept":
        preference[tied_items] = current_rank
      else:
        if tie_breaker == "random":
          np.random.shuffle(tied_items)
        elif tie_breaker == "first":
          tied_items = np.sort(tied_items)
        preference[tied_items] = np.arange(current_rank, len(tied_items) + current_rank)
      current_rank += len(tied_items)
    for _ in range(multiplicity):
      arr.append(preference)
  return np.array(arr)

def preflib_categorical_to_profile(instance: CategoricalInstance, tie_breaker: str = "random") -> np.ndarray:
  """
  Convert a Preflib categorical instance to the profile (Numpy matrix) format.

  For details on Preflib categorical, see https://www.preflib.org/format

  Parameters
  ----------
  instance: CategoricalInstance
    The Preflib categorical instance to convert. This is included in the preflibtools.instances module.

  tie_breaker : {"random", "first", "accept"}
    - "random": shuffle the tied items into a random order
    - "first": sort the tied items in ascending order
    - "accept": do not break ties

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib categorical instance.
  """
  # This is essentially equal to a toi.
  arr = []
  for p in instance.preferences:
    preference = np.full(instance.num_alternatives, np.nan)
    current_rank = 1
    for tied_items in p:
      # This condition is necessary to avoid indexing errors.
      if len(tied_items) == 0:
        continue
      tied_items = np.array(tied_items) - 1
      if tie_breaker == "accept":
        preference[tied_items] = current_rank
      else:
        if tie_breaker == "random":
          np.random.shuffle(tied_items)
        elif tie_breaker == "first":
          tied_items = np.sort(tied_items)
        preference[tied_items] = np.arange(current_rank, len(tied_items) + current_rank)
      current_rank += len(tied_items)
    for _ in range(instance.multiplicity[p]):
      arr.append(preference)
  return np.array(arr)
