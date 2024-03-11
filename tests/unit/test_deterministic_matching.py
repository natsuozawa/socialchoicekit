import numpy as np

import pytest

from typing import Dict, Tuple, List
import sys

from socialchoicekit.deterministic_matching import GaleShapley, Irving
from socialchoicekit.profile_utils import StrictIncompleteProfile, StrictCompleteProfile, CompleteValuationProfile, is_consistent_valuation_profile
from socialchoicekit.flow import ford_fulkerson

class TestDeterministicMatching:
  # Example from Handbook of Computational Social Choice, Chapter 14.
  @pytest.fixture
  def basic_resident_profile_1(self):
    return StrictIncompleteProfile.of(np.array([
      [1, 2, np.nan],
      [1, 2, 3],
      [2, 1, 3],
      [2, 1, np.nan],
    ]))

  @pytest.fixture
  def basic_hospital_profile_1(self):
    return StrictIncompleteProfile.of(np.array([
      [3, 2, 1, 4],
      [3, 1, 2, 4],
      [np.nan, 1, 2, np.nan],
    ]))

  @pytest.fixture
  def basic_c_1(self):
    return np.array([1, 2, 1])

  def test_gale_shapley_basic_profile_1(self, basic_resident_profile_1, basic_hospital_profile_1, basic_c_1):
    rogs = GaleShapley(resident_oriented=True)
    rogs_assignments = rogs.scf(basic_resident_profile_1, basic_hospital_profile_1, basic_c_1)
    assert len(rogs_assignments) == 3
    assert (1, 2) in rogs_assignments
    assert (2, 1) in rogs_assignments
    assert (3, 2) in rogs_assignments

    hogs = GaleShapley(resident_oriented=False)
    hogs_assignments = hogs.scf(basic_resident_profile_1, basic_hospital_profile_1, basic_c_1)
    assert len(hogs_assignments) == 3
    assert (3, 2) in hogs_assignments
    assert (2, 1) in hogs_assignments
    assert (1, 2) in hogs_assignments

  def test_gale_shapley_capacity_requirement(self, basic_resident_profile_1, basic_hospital_profile_1):
    m = basic_hospital_profile_1.shape[1]

    rogs = GaleShapley(resident_oriented=True)
    rogs_assignments = rogs.scf(basic_resident_profile_1, basic_hospital_profile_1, np.ones(3))
    rogs_hospital_assignment_counter = {hospital: 0 for hospital in range(1, m + 1)}
    for h, _ in rogs_assignments:
      rogs_hospital_assignment_counter[h] += 1
      assert rogs_hospital_assignment_counter[h] == 1

    hogs = GaleShapley(resident_oriented=False)
    hogs_assignments = hogs.scf(basic_resident_profile_1, basic_hospital_profile_1, np.ones(3))
    hogs_hospital_assignment_counter = {hospital: 0 for hospital in range(1, m + 1)}
    for h, _ in hogs_assignments:
      hogs_hospital_assignment_counter[h] += 1
      assert hogs_hospital_assignment_counter[h] == 1

  @pytest.fixture
  def profiles_2(self):
    # Example given in Irving, et al. (1987)
    ranked_ordinal_profile_1 = np.array([
      [3, 1, 5, 7, 4, 2, 8, 6],
      [6, 1, 3, 4, 8, 7, 5, 2],
      [7, 4, 3, 6, 5, 1, 2, 8],
      [5, 3, 8, 2, 6, 1, 4, 7],
      [4, 1, 2, 8, 7, 3, 6, 5],
      [6, 2, 5, 7, 8, 4, 3, 1],
      [7, 8, 1, 6, 2, 3, 4, 5],
      [2, 6, 7, 1, 8, 3, 4, 5],
    ]) - 1
    ranked_ordinal_profile_2 = np.array([
      [4, 3, 8, 1, 2, 5, 7, 6],
      [3, 7, 5, 8, 6, 4, 1, 2],
      [7, 5, 8, 3, 6, 2, 1, 4],
      [6, 4, 2, 7, 3, 1, 5, 8],
      [8, 7, 1, 5, 6, 4, 3, 1],
      [5, 4, 7, 6, 2, 8, 3, 1],
      [1, 4, 5, 6, 2, 8, 3, 7],
      [2, 5, 4, 3, 7, 8, 1, 6],
    ]) - 1

    # Custom
    # Use the borda-like weights that Irving et al. (1987) used.
    ranked_cardinal_profile_1 = np.array([
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ])

    ranked_cardinal_profile_2 = np.array([
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ])

    # ranked_cardinal_profile_1 = np.array([
    #   [12, 12, 12, 0, 0, 0, 0, 0],
    #   [36, 0, 0, 0, 0, 0, 0, 0],
    #   [15, 15, 6, 0, 0, 0, 0, 0],
    #   [9, 9, 9, 9, 0, 0, 0, 0],
    #   [7, 7, 7, 7, 2, 2, 2, 2],
    #   [7, 7, 7, 7, 2, 2, 2, 2],
    #   [12, 12, 12, 0, 0, 0, 0, 0],
    #   [18, 18, 0, 0, 0, 0, 0, 0],
    # ])
    # ranked_cardinal_profile_2 = np.array([
    #   [6, 6, 4, 4, 4, 4, 4, 4],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [5, 5, 5, 5, 5, 5, 5, 1],
    #   [5, 5, 5, 5, 4, 4, 4, 4],
    #   [5, 5, 5, 5, 5, 5, 5, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    # ])

    # Fix into form accepted by Profile, ValuationProfile
    ordinal_profile_1 = np.argsort(ranked_ordinal_profile_1, axis=1)
    ordinal_profile_2 = np.argsort(ranked_ordinal_profile_2, axis=1)
    cardinal_profile_1 = np.zeros(ranked_cardinal_profile_1.shape)
    cardinal_profile_2 = np.zeros(ranked_cardinal_profile_2.shape)
    cardinal_profile_1 = np.take_along_axis(ranked_cardinal_profile_1, ordinal_profile_1, axis=1)
    cardinal_profile_2 = np.take_along_axis(ranked_cardinal_profile_2, ordinal_profile_2, axis=1)

    return StrictCompleteProfile.of(ordinal_profile_1 + 1), StrictCompleteProfile.of(ordinal_profile_2 + 1), CompleteValuationProfile.of(cardinal_profile_1), CompleteValuationProfile.of(cardinal_profile_2)

  @pytest.fixture
  def initial_preference_lists_2(self):
    preference_list_1 = {
      0: np.array([3, 1, 5, 7, 4]) - 1,
      1: np.array([1, 3, 4, 8, 7]) - 1,
      2: np.array([7, 4, 3, 1, 2, 8]) -  1,
      3: np.array([5, 8, 6, 1, 4, 7]) - 1,
      4: np.array([4, 2, 8, 7, 3, 6, 5]) - 1,
      5: np.array([6, 5, 7, 4, 3]) - 1,
      6: np.array([8, 6, 2, 3, 4, 5]) - 1,
      7: np.array([2, 7, 1, 3, 5]) - 1,
    }

    preference_list_2 = {
      0: np.array([4, 3, 8, 1, 2]) - 1,
      1: np.array([3, 7, 5, 8]) - 1,
      2: np.array([7, 5, 8, 3, 6, 2, 1]) - 1,
      3: np.array([6, 4, 2, 7, 3, 1, 5]) - 1,
      4: np.array([8, 7, 1, 5, 6, 4]) - 1,
      5: np.array([5, 4, 7, 6]) - 1,
      6: np.array([1, 4, 5, 6, 2, 8, 3]) - 1,
      7: np.array([2, 5, 4, 3, 7]) - 1,
    }

    return preference_list_1, preference_list_2

  @pytest.fixture
  def exposed_rotations_2(self):
    return [[(0, 2), (1, 0)], [(2, 6), (4, 3), (7, 1)], [(3, 4), (6, 7), (5, 5)]]

  @pytest.fixture
  def all_rotations_2(self):
    return [
      [(0, 2), (1, 0)],
      [(2, 6), (4, 3), (7, 1)],
      [(3, 4), (6, 7), (5, 5)],
      [(0, 0), (5, 4), (7, 6)],
      [(1, 2), (2, 3)],
      [(3, 7), (6, 5), (4, 1)],
      [(2, 2), (7, 0)],
      [(1, 3), (4, 7), (5, 6)],
      [(0, 4), (4, 6), (7, 2)],
      [(2, 0), (6, 1), (4, 2), (3, 5)],
    ]

  @pytest.fixture
  def sparsest_rotation_poset_graph_2(self, all_rotations_2):
    return {
      0: [3, 4],
      1: [3, 4, 5],
      2: [3, 5],
      3: [6, 7],
      4: [6, 7],
      5: [7],
      6: [8],
      7: [8],
      8: [9],
      9: [],
    }

  def test_profile_consistency_2(self, profiles_2):
    ordinal_profile_1, ordinal_profile_2, cardinal_profile_1, cardinal_profile_2 = profiles_2
    assert is_consistent_valuation_profile(cardinal_profile_1, ordinal_profile_1)
    assert is_consistent_valuation_profile(cardinal_profile_2, ordinal_profile_2)

  def test_gale_shapley_2(self, profiles_2):
    ordinal_profile_1, ordinal_profile_2, _, _ = profiles_2
    stable_marriage = GaleShapley(
      resident_oriented=True, zero_indexed=True
    ).scf(
      ordinal_profile_1, ordinal_profile_2,
      np.ones(ordinal_profile_1.shape[0], dtype=int),
    )
    assert (0, 2) in stable_marriage
    assert (1, 0) in stable_marriage
    assert (2, 6) in stable_marriage
    assert (3, 4) in stable_marriage
    assert (4, 3) in stable_marriage
    assert (5, 5) in stable_marriage
    assert (6, 7) in stable_marriage
    assert (7, 1) in stable_marriage

  def test_find_initial_preference_lists_2(self, profiles_2, initial_preference_lists_2):
    ordinal_profile_1, ordinal_profile_2, _, _ = profiles_2
    shortlist_1, shortlist_2 = initial_preference_lists_2

    irving = Irving()

    stable_marriage = GaleShapley(
      resident_oriented=True, zero_indexed=True
    ).scf(
      ordinal_profile_1, ordinal_profile_2,
      np.ones(ordinal_profile_1.shape[0], dtype=int),
    )

    preference_list_1, preference_list_2 = irving.find_initial_preference_lists(
      stable_marriage,
      ordinal_profile_1 - 1,
      ordinal_profile_2 - 1
    )

    for i in shortlist_1.keys():
      assert np.issubdtype(preference_list_1[i].dtype, np.integer)
      assert np.array_equal(preference_list_1[i], shortlist_1[i])
    for i in shortlist_2.keys():
      assert np.issubdtype(preference_list_2[i].dtype, np.integer)
      assert np.array_equal(preference_list_2[i], shortlist_2[i])

  def test_find_rotations_2(self, initial_preference_lists_2, exposed_rotations_2):
    shortlist_1, shortlist_2 = initial_preference_lists_2
    irving = Irving()
    rotations = irving.find_rotations(shortlist_1, shortlist_2)

    # We must compare in this way because there are multiple valid orders of pairs in each rotation.
    assert all(
      [any(
        [any(
          [pair in answer_rotation for pair in solved_rotation]
        ) for answer_rotation in exposed_rotations_2]
      ) for solved_rotation in rotations]
    )

    assert len(rotations) == len(exposed_rotations_2)

  def test_find_all_rotations_and_eliminations_2(self, initial_preference_lists_2, all_rotations_2):
    shortlist_1, shortlist_2 = initial_preference_lists_2
    irving = Irving()
    # We don't test eliminations for now.
    rotations, _ = irving.find_all_rotations_and_eliminations(shortlist_1, shortlist_2)

    assert all(
      [any(
        [any(
          [pair in answer_rotation for pair in solved_rotation]
        ) for answer_rotation in all_rotations_2]
      ) for solved_rotation in rotations]
    )

    assert len(rotations) == len(all_rotations_2)

  def test_construct_sparse_rotation_poset_graph(self, initial_preference_lists_2, all_rotations_2, sparsest_rotation_poset_graph_2):
    shortlist_1, shortlist_2 = initial_preference_lists_2
    irving = Irving()

    # Copy shortlist_1
    preference_lists_1 = {i: np.array(shortlist_1[i]) for i in range(len(shortlist_1))}
    rotations, eliminations = irving.find_all_rotations_and_eliminations(shortlist_1, shortlist_2)
    P_prime = irving.construct_sparse_rotation_poset_graph(rotations, preference_lists_1, eliminations)

    # Check that this is a supergraph of the sparsest rotation poset graph.
    # We have to do this in a verbose way because the ordering of the rotations and the pairs in the rotations do not necessarily align.
    expected_to_actual_mapping = {}
    for i, rotation_expected in enumerate(all_rotations_2):
      for j, rotation_actual in enumerate(rotations):
        is_match = all([pair in rotation_actual for pair in rotation_expected])
        if is_match:
          expected_to_actual_mapping[i] = j
          break

    for i, destinations in sparsest_rotation_poset_graph_2.items():
      for j in destinations:
        assert expected_to_actual_mapping[j] in P_prime[expected_to_actual_mapping[i]]

  def test_rotation_weight_2(self, profiles_2, all_rotations_2):
    _, _, cardinal_profile_1, cardinal_profile_2 = profiles_2
    irving = Irving()
    rotation_weights = [irving.rotation_weight(all_rotations_2[i], cardinal_profile_1, cardinal_profile_2) for i in range(10)]
    # The rotation weights are taken from Irving, et al. (1987)
    # We believe there is a mistake in the original paper for the 7th rotation. We have corrected it here from -1 to -3.
    # This changes the result of the maximum weight closed subset.
    assert rotation_weights == [0, -1, -2, 2, 2, -1, -3, 0, 1, 0]

  def test_maximum_weight_closed_subset_2(self, profiles_2, all_rotations_2, sparsest_rotation_poset_graph_2):
    _, _, cardinal_profile_1, cardinal_profile_2 = profiles_2
    irving = Irving()
    # maximum_weight_closed_subset = irving.find_maximum_weight_closed_subset(sparsest_rotation_poset_graph_2, all_rotations_2, cardinal_profile_1, cardinal_profile_2)
    # Due to the mistake mentioned above, we cannot compare the maximum weight to the one found in the paper.

    def adjusted_rotation_weights(rotation_index):
      if rotation_index == 6:
        return -1
      else:
        return irving.rotation_weight(
          all_rotations_2[rotation_index],
          cardinal_profile_1,
          cardinal_profile_2
        )

    # Here we use the original weight from the paper by copy pasting the code.
    # === Begin Duplication ===
    network: Dict[int, List[Tuple[int, int]]] = {-1: [], -2: []}
    temp_maximum_weight_closed_subset = set()
    for pi in sparsest_rotation_poset_graph_2:
      network[pi] = [(rho, sys.maxsize) for rho in sparsest_rotation_poset_graph_2[pi]]
      w = adjusted_rotation_weights(pi)
      if w > 0:
        network[pi].append((-2, int(w)))
        temp_maximum_weight_closed_subset.add(pi)
      elif w < 0:
        network[-1].append((pi, int(-w)))

    flow, min_cut = ford_fulkerson(network, -1, -2)
    min_cut.remove(-1)

    maximum_weight_closed_subset = set()
    for positive_node in temp_maximum_weight_closed_subset:
      if positive_node not in min_cut:
        maximum_weight_closed_subset.add(positive_node)

    while True:
      continue_loop = False
      for rho in sparsest_rotation_poset_graph_2.keys():
        if rho in maximum_weight_closed_subset:
          continue
        if len(set(sparsest_rotation_poset_graph_2[rho]).intersection(maximum_weight_closed_subset)) > 0:
          maximum_weight_closed_subset.add(rho)
          continue_loop = True
      if not continue_loop:
        break
    # === End Duplication ===

    rotation_weights = [adjusted_rotation_weights(rotation_index) for rotation_index in maximum_weight_closed_subset]
    assert sum(rotation_weights) == 1

  def test_irving_2(self, profiles_2):
    ordinal_profile_1, ordinal_profile_2, cardinal_profile_1, cardinal_profile_2 = profiles_2
    irving = Irving()
    irving.scf(
      cardinal_profile_1,
      cardinal_profile_2,
      ordinal_profile_1,
      ordinal_profile_2,
    )
