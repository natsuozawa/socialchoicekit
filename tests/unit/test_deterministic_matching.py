import numpy as np

import pytest

from socialchoicekit.deterministic_matching import GaleShapley

class TestDeterministicMatching:
  # Example from Handbook of Computational Social Choice, Chapter 14.
  @pytest.fixture
  def basic_resident_profile_1(self):
    return np.array([
      [1, 2, np.nan],
      [1, 2, 3],
      [2, 1, 3],
      [2, 1, np.nan],
    ])

  @pytest.fixture
  def basic_hospital_profile_1(self):
    return np.array([
      [3, 2, 1, 4],
      [3, 1, 2, 4],
      [np.nan, 1, 2, np.nan],
    ])

  @pytest.fixture
  def basic_c_1(self):
    return np.array([1, 2, 1])

  def test_basic_profile_1(self, basic_resident_profile_1, basic_hospital_profile_1, basic_c_1):
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


