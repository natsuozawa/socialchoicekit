import numpy as np

from typing import List, Tuple, Optional, Dict
import heapq
import sys

from socialchoicekit.profile_utils import StrictProfile, StrictCompleteProfile, CompleteValuationProfile, compute_ordinal_profile
from socialchoicekit.utils import check_valuation_profile, check_profile
from socialchoicekit.flow import ford_fulkerson

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
    resident_profile: StrictProfile,
    hospital_profile: StrictProfile,
    c: np.ndarray,
  ) -> List[Tuple[int, int]]:
    """
    The (provisional) social choice function for this voting rule. Returns one item allocated for each agent.

    Parameters
    ----------
    resident_profile StrictProfile
      A (N, M) array, where N is the number of residents and M is the number of hospitals. The element at (i, j) indicates the resident's preference for hospital j, where 1 is the most preferred hospital. If the resident finds a hospital unacceptable, the element would be np.nan.

    hospital_profile: StrictProfile
      A (M, N) array, where M is the number of hospitals and N is the number of residents. The element at (i, j) indicates the hospital's preference for resident j, where 1 is the most preferred resident. If the hospital finds a resident unacceptable, the element would be np.nan.

    c: np.ndarray
      A M-array containing the capacities of the hospitals.

    Returns
    -------
    List[Tuple[int, int]]
      A list containing assignments (resident, hospital) for each assignment.
    """
    n = resident_profile.shape[0]
    m = resident_profile.shape[1]

    if n != hospital_profile.shape[1] or m != hospital_profile.shape[0]:
      raise ValueError("The resident profile and hospital profile dimensions do not match.")

    # Decrease by one because we will be using 0-indexing to access the ranked versions of these profiles.
    rprofile = resident_profile.view(np.ndarray) - 1
    hprofile = hospital_profile.view(np.ndarray) - 1

    # NaN will be put last.
    ranked_rprofile = np.argsort(rprofile, axis=1)
    ranked_hprofile = np.argsort(hprofile, axis=1)

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
          next_hospital = ranked_rprofile[resident, last_applied_hospital_rank + 1]
          if np.isnan(rprofile[resident, next_hospital]):
            # Candidate has applied to all hospitables they find acceptable. (Yet have not gotten accepted into any)
            next_current_applicants[resident] = 2
            continue

          resident_applications[resident] = last_applied_hospital_rank + 1

          if np.isnan(hprofile[next_hospital, resident]):
            # Candidate is unacceptable to the hospital. Auto-rejected.
            continue

          hospital_waiting_list = hospital_waiting_lists.get(next_hospital, [])
          # Negate resident rank because heapq is a min heap.
          heapq.heappush(hospital_waiting_list, int(hprofile[next_hospital, resident] * -1))
          next_current_applicants[resident] = 0

          if len(hospital_waiting_list) <= c[next_hospital]:
            # Hospital has not reached capacity yet.
            continue

          # Hospital has reached capacity.
          # Revert back from negated resident rank
          dropped_resident = ranked_hprofile[next_hospital, heapq.heappop(hospital_waiting_list) * -1]
          next_current_applicants[dropped_resident] = 1

      ans = []
      for hospital in range(m):
        for resident_rank in hospital_waiting_lists.get(hospital, []):
          # Revert back from negated resident rank
          ans.append((int(ranked_hprofile[hospital, resident_rank * -1]) + self.index_fixer, hospital + self.index_fixer))
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
          next_resident = ranked_hprofile[hospital, last_applied_resident_rank + 1]
          if np.isnan(hprofile[hospital, next_resident]):
            # Hospital has offered to all residents they find acceptable. (Yet are undersubscribed)
            current_offerers[hospital] = 2

          hospital_offers[hospital] = last_applied_resident_rank + 1

          if np.isnan(rprofile[next_resident, hospital]):
            # Hospital is unacceptable to the resident. Auto-rejected.
            continue

          # Negate resident rank because heapq is a min heap.
          current_accepted_hospital = resident_waiting_lists[next_resident]
          if current_accepted_hospital == -1 or rprofile[next_resident, hospital] < rprofile[next_resident, current_accepted_hospital]:
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

class Irving:
  """
  Algorithm for computing an optimal stable matching introduced in Irving et al. (1987) and modified for cardinal utilities (called weighted preference lists in the paper).
  This algorithm works with a simplified version of the hospital resident problem (HR) where each hospital can only take one resident, and the number of hospitals and residents are equal. We call this the stable marriage problem (SM).
  We replace residents with men and hospitals with women.
  The algorithm also will only work with complete valuation profiles.
  """
  def __init__(
    self,
  ):
    pass

  def scf(
    self,
    valuation_profile_1: CompleteValuationProfile,
    valuation_profile_2: CompleteValuationProfile,
    profile_1: Optional[StrictCompleteProfile] = None,
    profile_2: Optional[StrictCompleteProfile] = None,
  ) -> List[Tuple[int, int]]:
    """
    The social choice function for this voting rule. Returns a stable matching that optimizes social welfare based on the given valuation profile.
    The optional ordinal profile parameters will be useful if the valuation profile(s) provided are simulated (or estimated) and contains ties.
    The ordinal profile(s) will be used to maintain stability.
    The Irving algorithm assumes a strict ordering of preferences to create rotations. If a strict complete ordinal profile is not given, the ordinal profile will be automatically computed from the valuation profile (ties will be randomly broken).
    To break ties in some other way, use profile_utils.compute_ordinal_profile.

    Parameters
    ----------
    valuation_profile_1: CompleteValuationProfile
      A (N, N) array, where N is the number of men and also the number of women. The element at (i, j) indicates the ith man's cardinal preference for woman j.
    valuation_profile_2: CompleteValuationProfile
      A (N, N) array, where N is the number of women and also the number of men. The element at (i, j) indicates the ith woman's cardinal preference for man j.
    profile_1: Optional[StrictCompleteProfile]
      An optional (N, N) array, where N is the number of men and also the number of women. The element at (i, j) indicates the ith man's ordinal preference for woman j. 1 is the most preferred.
      If None, then the ordinal profile will be computed from the valuation profile.
    profile_2: Optional[StrictCompleteProfile]
      An optional (N, N) array, where N is the number of women and also the number of men. The element at (i, j) indicates the ith woman's ordinal preference for man j. 1 is the most preferred.
      If None, then the ordinal profile will be computed from the valuation profile.

    Returns
    -------
    List[Tuple[int, int]]
      A list containing assignments (resident, hospital) for each assignment.
    """
    check_valuation_profile(valuation_profile_1, is_complete=True)
    check_valuation_profile(valuation_profile_2, is_complete=True)
    if isinstance(profile_1, StrictCompleteProfile):
      check_profile(profile_1, is_complete=True, is_strict=True)
      ordinal_profile_1 = profile_1.view(np.ndarray)
    else:
      ordinal_profile_1 = compute_ordinal_profile(valuation_profile_1).view(np.ndarray)
    if isinstance(profile_2, StrictCompleteProfile):
      check_profile(profile_2, is_complete=True, is_strict=True)
      ordinal_profile_2 = profile_2.view(np.ndarray)
    else:
      ordinal_profile_2 = compute_ordinal_profile(valuation_profile_2).view(np.ndarray)

    n = valuation_profile_1.shape[0]
    assert (n, n) == valuation_profile_1.shape
    assert (n, n) == valuation_profile_2.shape
    assert (n, n) == ordinal_profile_1.shape
    assert (n, n) == ordinal_profile_2.shape

    # 0-indexed
    ranked_profile_1 = np.argsort(valuation_profile_1, axis=1)
    ranked_profile_2 = np.argsort(valuation_profile_2, axis=1)
    prof_1 = ordinal_profile_1 - 1
    prof_2 = ordinal_profile_2 - 1

    # Get the male optimal stable matching.
    stable_marriage = GaleShapley(resident_oriented=True, zero_indexed=True).scf(
      StrictCompleteProfile.of(ordinal_profile_1),
      StrictCompleteProfile.of(ordinal_profile_2),
      np.ones(n, dtype=int)
    )
    # Capacity requriement is tested in TestDeterministicMatching.

    # Check each man is matched to exactly one woman and vice versa.
    assert len(stable_marriage) == n
    assert len(set([i for i, _ in stable_marriage])) == n
    assert len(set([j for _, j in stable_marriage])) == n

    # Reconstruct preference lists. (0-indexed)
    # We first cut [0, matched_woman) from the man's preference lists because of Property 3 in Irving et al (1987): in the male optimal solution, every man is matched to the first woman on his shortlist.
    preference_lists_1 = {i: ranked_profile_1[i, prof_1[i, j]:] for i, j in stable_marriage}
    # We first cut (matched_man, n-1] from the woman's preference lists because of Property 2 in Irving et al (1987): in the male optimal solution, every woman is matched to the last man on her shortlist.
    preference_lists_2 = {j: ranked_profile_2[j, :prof_2[j, i]] for i, j in stable_marriage}

    # We then reduce the shortlist to enforce the first statement of Property 2.
    # This is O(n^3) using a naive approach.
    new_preference_lists_1 = {}
    new_preference_lists_2 = {}
    for i in range(n):
      new_preference_lists_1[i] = np.array([])
      for j in preference_lists_1[i]:
        if i in preference_lists_2[j]:
          new_preference_lists_1[i] = np.append(preference_lists_1[i], j)
    for j in range(n):
      new_preference_lists_2[j] = np.array([])
      for i in preference_lists_2[j]:
        if j not in new_preference_lists_1[i]:
          new_preference_lists_2[j] = np.append(new_preference_lists_2[j], i)

    preference_lists_1 = new_preference_lists_1
    preference_lists_2 = new_preference_lists_2

    rotations, eliminating_rotations_of_pair = self.find_all_rotations_and_eliminations(preference_lists_1, preference_lists_2)

    # Construct P'
    # Nodes: rotation
    # Edges: From rule 1 and 2
    # Rule 1: If (m, w) is a member of a rotation, say pi,and w’ is the first woman
    # below w in m’s list such that (m, w’) is a member of some other rotation,
    # say rho, then P’ contains adirected edgefrom pi to rho.
    # Rule 2: If (m, w’) is not a member of any rotation, but is eliminated by some rotation,
    # say pi, and w is the first woman above w’ in m’s list such that (m, w) is
    # a member of some rotation, say rho, then P’ contains a directed edge from pi to rho.
    P_prime = {pi: [] for pi in range(len(rotations))}

    rotation_of_pair = {}
    for index, rotation in enumerate(rotations):
      for i, j in rotation:
        rotation_of_pair[(i, j)] = index

    # We use i, j instead of m, w here.
    for i in range(n):
      for index, j in enumerate(preference_lists_1[i]):
        if index == len(preference_lists_1[i]) - 1:
          # Cannot create edges from the last woman on the preference list.
          # So skip.
          continue
        if (i, j) not in rotation_of_pair:
          # We cannot construct (m, w) which is in a rotation.
          continue
        pi = rotation_of_pair[(i, j)]
        j_prime = preference_lists_1[i][index + 1]
        if (i, j_prime) in rotation_of_pair:
          # Rule 1 is satisfied.
          rho = rotation_of_pair[(i, j_prime)]
        elif (i, j_prime) in eliminating_rotations_of_pair:
          # Rule 2 is satisfied.
          rho = eliminating_rotations_of_pair[(i, j_prime)]
        P_prime[pi].append(rho)

    # source s: -1, sink t: -2
    # Elements represent (destination, capacity)
    network: Dict[int, List[Tuple[int, int]]] = {-1: [], -2: []}
    for pi in P_prime:
      network[pi] = [(rho, sys.maxsize) for rho in P_prime[pi]]
      w = self.rotation_weight(rotations[pi], valuation_profile_1, valuation_profile_2)
      if w < 0:
        network[pi].append((-2, int(-w)))
      elif w > 0:
        network[-1].append((pi, int(w)))

    max_flow = ford_fulkerson(network, -1, -2)
    # TODO: modify FF implementation to return min_cut
    return []

  def find_all_rotations_and_eliminations(
    self,
    preference_lists_1: Dict[int, np.ndarray],
    preference_lists_2: Dict[int, np.ndarray],
  ) -> Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
    """
    This is an internal routine to find the set of al rotations that we can obtain by eliminating some rotations. This includes the rotations that are already exposed in a stable matching.
    We also note for each pair if there is a rotation that eliminates it.
    The parameters indicate the reduced preference lists at the time of finding a stable matching.

    Complexity
    ----------
    O(n^3)

    Parameters
    ----------
    preference_lists_1: Dict[int, np.ndarray]
      A dictionary where the key is an integer indicating a man in 0-index. The value is an array of integers. The kth element indicates the man's kth most preferred woman in his shortlist in 0-index.
      This shortlist must be reduced.
      The dictionary must contain n keys and each preference list must be at most n long.
    preference_lists_2: Dict[int, np.ndarray]
      A dictionary where the key is an integer indicating a woman in 0-index. The value is an array of integers. The kth element indicates the woman's kth most preferred man in her shortlist in 0-index.
      This shortlist must be reduced.
      The dictionary must contain n keys and each preference list must be at most n long.

    Returns
    -------
    Tuple[List[List[Tuple[int, int]]], Dict[Tuple[int, int], int]
      Each item is described below.

    List[List[Tuple[int, int]]]
      A list containing all the rotations reachable in the stable matching. Each rotation is a list of 0-indexed man-woman pairs.

    Dict[Tuple[int, int], int]
      A map from a 0-indexed man-woman pair (m, w) to the 0-indexed index (in the first item of the returned tuple) of the rotation that eliminates it.
    """
    n = len(preference_lists_1)
    assert n == len(preference_lists_2)

    ans = []

    # Use binary indicator representation to allow for faster access to see if an element is in the preference list.
    # This technique allows for the entire routine to be O(n^3).
    # 1 to indicate that the pair is still in the preference list. 0 to indicate otherwise.
    preference_matrix_1 = {(i, j): 1 for i in range(n) for j in preference_lists_1[i]}
    preference_matrix_2 = {(j, i): 1 for j in range(n) for i in preference_lists_2[j]}

    # Male preference list is incomplete to the right of the start pointer
    # and is indicated by [start_pointer, r)
    # where r is the original length of the preference list.
    # Female preference list is complete and is indicated by [0, end_pointer]

    # No node can be in two cycles at once in G(S).
    # Therefore, no man or woman is in two rotations at once.
    # Hence, we eliminate all the rotations in the same level simultaneously to expose a new set of rotations.

    eliminating_rotations_of_pair = {}
    current_rotation = -1
    while True:
      rotations = self.find_rotations(preference_lists_1, preference_lists_2)
      if len(rotations) == 0:
        break
      ans += rotations
      # The outer two loops are O(n) combined
      # because we only update the preference lists once for each person in each level.
      for rotation in rotations:
        current_rotation += 1
        # Eliminate.
        r = len(rotation)
        for i in range(r):
          m_i_minus_1 = rotation[(i - 1) % r][0]
          w_i = rotation[i][1]
          k = len(preference_lists_2[w_i])
          # This part is O(n) in total of all levels.
          while k >= 0:
            if preference_lists_2[w_i][k] == m_i_minus_1:
              preference_lists_2[w_i] = preference_lists_2[w_i][:k]
              break
            preference_matrix_2[(w_i, preference_lists_2[w_i][k])] = 0
            eliminating_rotations_of_pair[(preference_lists_2[w_i][k], w_i)] = current_rotation
            k -= 1
    # Eliminate male preference lists. O(n^2)
    for i in range(n):
      k = 0
      while k < len(preference_lists_1[i]):
        j = preference_lists_1[i][k]
        in_preference_list = preference_matrix_2.get((j, i), 0)
        if in_preference_list:
          preference_lists_1[i] = preference_lists_1[i][k:]
          break
        preference_matrix_1[(i, j)] = 0
        k += 1
    return ans, eliminating_rotations_of_pair

  def find_rotations(
    self,
    preference_lists_1: Dict[int, np.ndarray],
    preference_lists_2: Dict[int, np.ndarray],
  ) -> List[List[Tuple[int, int]]]:
    """
    This is an internal routine to find the set of all rotations that are exposed in a stable matching, given the preference lists.
    We find the solution by constructing the graph G(S) as described in Irving et al. (1987) and finding all cycles in G(S).

    Complexity
    ----------
    O(n)

    Parameters
    ----------
    preference_lists_1: Dict[int, np.ndarray]
      A dictionary where the key is an integer indicating a man in 0-index. The value is an array of integers. The kth element indicates the man's kth most preferred woman in his shortlist in 0-index.
      Each man's shortlist does not have to be fully reduced. Only the first and second elements are used.
      The dictionary must contain n keys and each preference list must be at most n long.
    preference_lists_2: Dict[int, np.ndarray]
      A dictionary where the key is an integer indicating a woman in 0-index. The value is an array of integers. The kth element indicates the woman's kth most preferred man in her shortlist in 0-index.
      Each woman's shortlist does not have to be reduced. Only the last element is used.
      The dictionary must contain n keys and each preference list must be at most n long.

    Returns
    -------
    List[List[Tuple[int, int]]]
      A list containing all the rotations in exposed the stable matching. Each rotation is a list of 0-indexed man-woman pairs.
    """
    # Graph G(S)
    # Nodes: man (0-indexed)
    # Edges: betwen man i and man i' if the woman who is second on man i's preference list
    # has man i' at the top of her preference list.
    # Note that in this graph, each node has at most one outgoing edge.
    n = len(preference_lists_1)
    assert n == len(preference_lists_2)
    G = {i: [] for i in range(n)}
    for i in range(n):
      if len(preference_lists_1[i]) <= 1:
        continue
      j = preference_lists_1[i][1]
      i_prime = preference_lists_2[j][-1]
      if i != i_prime:
        G[i].append(i_prime)

    # Find all cycles in G(S)
    # We exploit the fact that G(S) is a directed graph with at most one outgoing edge from each node.
    visited = [False] * n
    start_point = 0
    cycles = []
    while start_point < n:
      if visited[start_point]:
        start_point += 1
        continue
      cycle = []
      current_node = start_point
      while not visited[current_node]:
        visited[current_node] = True
        next_node = G[current_node][0]
        cycle.append((current_node, preference_lists_1[current_node][0]))
        current_node = next_node
      index = cycle.index((current_node, preference_lists_1[current_node][0]))
      cycles.append(cycle[index:])
    return cycles

  def rotation_weight(
    self,
    rotation: List[Tuple[int, int]],
    valuation_profile_1: CompleteValuationProfile,
    valuation_profile_2: CompleteValuationProfile,
  ) -> float:
    """
    The weight of a rotation as defined in Irving et al. (1987).
    The weight of the rotation must be smaller than sys.maxsize to work with Irving.

    Parameters
    ----------
    rotation: List[Tuple[int, int]]
      Rotations of the form [(m_0, w_0), ..., (m_{r-1}, w_{r-1})] where m_i, w_i are 0-indexed.

    valuation_profile_1: CompleteValuationProfile
      The male valuation profile.

    valuation_profile_1: CompleteValuationProfile
      The female valuation profile.

    Returns
    -------
    float
      The weight of the rotation.
    """
    r = len(rotation)
    ans = 0
    for i in range(r):
      ans += valuation_profile_1[rotation[i][0], rotation[i][1]] - valuation_profile_1[rotation[i][0], rotation[(i + 1) % r][1]]
      ans -= valuation_profile_2[rotation[i][1], rotation[i][0]] - valuation_profile_2[rotation[i][1], rotation[(i - 1) % r][0]]
    return ans
