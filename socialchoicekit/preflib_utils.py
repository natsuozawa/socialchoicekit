import numpy as np

from typing import List, Tuple
from preflibtools.instances import OrdinalInstance

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
