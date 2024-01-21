import numpy as np

from typing import List, Tuple

def preflib_soc_to_profile(soc: List[Tuple[Tuple[int, ...], int]]) -> np.ndarray:
  """
  Convert a Preflib SoC (Strictly Orders - Complete List) to the profile (Numpy matrix) format.

  For details on Preflib SoC, see https://www.preflib.org/format

  Parameters
  ----------
  soc: List[Tuple[Tuple[int, ...], int]]
    The Preflib SoC to convert. This can be obtained by calling flatten_strict() on an OrdinalInstance from preflibtools

  Returns
  -------
  np.ndarray
    The profile (Numpy matrix) format of the Preflib SoC.
  """
  arr = []
  for order, multiplicity in soc:
    for _ in range(multiplicity):
      arr.append(order)
  return np.array(arr)
