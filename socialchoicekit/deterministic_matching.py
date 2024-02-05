import numpy as np

from typing import List, Tuple
import heapq

from socialchoicekit.utils import check_profile

class GaleShapley:
  """
  Resident-oriented Gale Shapley algorithm (RGS) is a deferred acceptance algorithm that finds a stable matching in the two sided matching setting. It is resident optimal.

  Parameters
  ----------
  resident_oriented : bool
    If True, the social choice function will be resident-oriented. If False, the social choice function will be hospital-oriented. Resident-oriented by default.
  zero_indexed : bool
    If True, the output of the social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    resident_oriented: bool = True,
    zero_indexed: bool = False,
  ):
    self.index_fixer = 0 if zero_indexed else 1
    self.resident_oriented = resident_oriented

  def scf(
    self,
    resident_profile: np.ndarray,
    hospital_profile: np.ndarray,
    c: np.ndarray,
  ) -> List[Tuple[int, int]]:
    """
    The (provisional) social choice function for this voting rule. Returns one item allocated for each agent.

    Parameters
    ----------
    resident_profile : np.ndarray
      A (N, M) array, where N is the number of residents and M is the number of hospitals. The element at (i, j) indicates the resident's preference for hospital j, where 1 is the most preferred hospital. If the resident finds a hospital unacceptable, the element would be np.nan.

    hospital_profile : np.ndarray
      A (M, N) array, where M is the number of hospitals and N is the number of residents. The element at (i, j) indicates the hospital's preference for resident j, where 1 is the most preferred resident. If the hospital finds a resident unacceptable, the element would be np.nan.

    c: np.ndarray
      A M-array containing the capacities of the hospitals.

    Returns
    -------
    List[Tuple[int, int]]
      A list containing assignments (resident, hospital) for each assignment.
    """
    check_profile(resident_profile, is_complete=False)
    check_profile(hospital_profile, is_complete=False)

    n = resident_profile.shape[0]
    m = resident_profile.shape[1]

    if n != hospital_profile.shape[1] or m != hospital_profile.shape[0]:
      raise ValueError("The resident profile and hospital profile dimensions do not match.")

    # Decrease by one because we will be using 0-indexing to access the ranked versions of these profiles.
    resident_profile = resident_profile - 1
    hospital_profile = hospital_profile - 1

    # NaN will be put last.
    ranked_resident_profile = np.argsort(resident_profile, axis=1)
    ranked_hospital_profile = np.argsort(hospital_profile, axis=1)

    if self.resident_oriented:
      # Key: resident, value = the last hospital the resident applied to
      resident_applications = {}

      # Key: hospital, value = list of residents the hospital is matched to,
      # where each resident is expressed as the ranked position for that hospital.
      # In resident-oriented Gale Shapley this is a priority queue.
      hospital_waiting_lists = {i: [] for i in range(m)}

      # Initially, everyone applies.
      next_current_applicants = np.ones(n, dtype=int)

      while True:
        if np.all(next_current_applicants != 1):
          break

        # Copy because we don't want the modification to take effect until the next iteration of the loop.
        current_applicants = np.array(next_current_applicants)

        # resident, next_hospital, dropped_resident are 0-indexed positions originally supplied in the input.
        # last_applied_hospital_rank is a 0-indexed position in the ranked resident profile.
        for resident in range(n):
          if current_applicants[resident] == 0 or current_applicants[resident] == 2:
            # Resident already has a match or rejection is confirmed.
            continue

          last_applied_hospital_rank = resident_applications.get(resident, -1)
          next_hospital = ranked_resident_profile[resident, last_applied_hospital_rank + 1]
          if np.isnan(resident_profile[resident, next_hospital]):
            # Candidate has applied to all hospitables they find acceptable. (Yet have not gotten accepted into any)
            next_current_applicants[resident] = 2
            continue

          resident_applications[resident] = last_applied_hospital_rank + 1

          if np.isnan(hospital_profile[next_hospital, resident]):
            # Candidate is unacceptable to the hospital. Auto-rejected.
            continue

          hospital_waiting_list = hospital_waiting_lists.get(next_hospital, [])
          # Negate resident rank because heapq is a min heap.
          heapq.heappush(hospital_waiting_list, int(hospital_profile[next_hospital, resident] * -1))
          next_current_applicants[resident] = 0

          if len(hospital_waiting_list) <= c[next_hospital]:
            # Hospital has not reached capacity yet.
            continue

          # Hospital has reached capacity.
          # Revert back from negated resident rank
          dropped_resident = ranked_hospital_profile[next_hospital, heapq.heappop(hospital_waiting_list) * -1]
          next_current_applicants[dropped_resident] = 1

      ans = []
      for hospital in range(m):
        for resident_rank in hospital_waiting_lists.get(hospital, []):
          # Revert back from negated resident rank
          ans.append((int(ranked_hospital_profile[hospital, resident_rank * -1]) + self.index_fixer, hospital + self.index_fixer))
      return ans

    else:
      # Key: resident, value = the last resident the hospital offered to
      hospital_offers = {}

      resident_waiting_lists = {i: -1 for i in range(n)}

      # np.nan if hospital is terminally undersubscribed.
      hospital_accepted_offers = np.zeros(m, dtype=int)
      current_offerers = np.ones(m, dtype=int)

      while True:
        current_offerers = np.where(current_offerers == 2, 2, np.where(c == hospital_accepted_offers, 0, 1))
        if np.all(current_offerers != 1):
          break

        # hospital, next_resident, dropped_hospital are 0-indexed positions originally supplied in the input.
        # last_applied_resident_rank is a 0-indexed position in the ranked resident profile.
        for hospital in range(m):
          if current_offerers[hospital] == 0 or current_offerers[hospital] == 2:
            # Hospital already has a match or undersubscription is confirmed.
            continue

          last_applied_resident_rank = hospital_offers.get(hospital, -1)
          next_resident = ranked_hospital_profile[hospital, last_applied_resident_rank + 1]
          if np.isnan(hospital_profile[hospital, next_resident]):
            # Hospital has offered to all residents they find acceptable. (Yet are undersubscribed)
            current_offerers[hospital] = 2

          hospital_offers[hospital] = last_applied_resident_rank + 1

          if np.isnan(resident_profile[next_resident, hospital]):
            # Hospital is unacceptable to the resident. Auto-rejected.
            continue

          # Negate resident rank because heapq is a min heap.
          current_accepted_hospital = resident_waiting_lists[next_resident]
          if current_accepted_hospital == -1 or resident_profile[next_resident, hospital] < resident_profile[next_resident, current_accepted_hospital]:
            # Resident has not received any offers yet or the hospital is more preferred than the resident's current offer.
            hospital_accepted_offers[hospital] += 1
            hospital_accepted_offers[current_accepted_hospital] -= 1
            resident_waiting_lists[next_resident] = hospital

      ans = []
      for resident in range(n):
        hospital = resident_waiting_lists.get(resident, -1)
        if hospital == -1:
          continue
        ans.append((resident + self.index_fixer, hospital + self.index_fixer))
      return ans



