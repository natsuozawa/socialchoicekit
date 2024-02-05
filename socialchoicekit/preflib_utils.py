import numpy as np

from preflibtools.instances import OrdinalInstance

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
  arr = []
  for order, multiplicity in flattened_order:
    # Order: strict complete order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    for _ in range(multiplicity):
      arr.append(order)
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

  m = instance.num_alternatives

  arr = []
  for order, multiplicity in flattened_order:
    # Order: strict incomplete order of the alternatives
    # Multiplicity: the number of agents that had this ordinal preference
    o = order + tuple([np.nan] * (m - len(order)))
    for _ in range(multiplicity):
      arr.append(o)
  return np.array(arr)

def preflib_toc_to_profile(instance: OrdinalInstance, tie_breaker: str = "random") -> np.ndarray:
  """
  Convert a Preflib ToC (Orders with Ties - Complete List) to the profile (Numpy matrix) format.
  Note that this procedure is tie-breaking. Information about ties are not preserved in the converted profile.
  Tie-breaking is done by random, but a tie_breaker may be supplied as an argument.

  For details on Preflib ToC, see https://www.preflib.org/format

  Parameters
  ----------
  toc: OrdinalInstance
    The Preflib ToC to convert. This is included in the preflibtools.instances module. The data_type for this must be toc.

  tie_breaker : {"random", "first", "accept"}
    - "random": shuffle the tied items into a random order
    - "first": sort the tied items in ascending order
    - "accept": keep the order of the tied items as is

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
    flattened_order = []
    for tied_items in order:
      if tie_breaker == "random":
        np.random.shuffle(tied_items)
      elif tie_breaker == "first":
        tied_items = np.sort(tied_items)
      flattened_order += list(tied_items)
    for _ in range(multiplicity):
      arr.append(flattened_order)
  return np.array(arr)
