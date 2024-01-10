import numpy as np
import pytest

from socialchoicekit.deterministic_allocation import MaximumWeightMatching

class TestDeterministicAllocation:
  @pytest.fixture
  def basic_valuation_profile_1(self):
    return np.array([
      [0.5, 0.3, 0.1, 0.2],
      [0.1, 0.7, 0.1, 0.1],
      [0.2, 0.1, 0.4, 0.3],
      [0.1, 0.1, 0.4, 0.4],
    ])

  def test_maximum_weight_matching_basic_1(self, basic_valuation_profile_1):
    mwm = MaximumWeightMatching()
    allocation = mwm.scf(basic_valuation_profile_1)
    assert np.all(allocation == np.array([0, 1, 2, 3]))

  @pytest.fixture
  def basic_valuation_profile_2(self):
    return np.array([
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
    ])

  def test_maximum_weight_matching_basic_2(self, basic_valuation_profile_2):
    mwm = MaximumWeightMatching()
    allocation = mwm.scf(basic_valuation_profile_2)
    # Check that an allocation is returned.
    assert np.all(np.sort(allocation) == np.array([0, 1, 2, 3]))

  @pytest.fixture
  def basic_valuation_profile_3(self):
    return np.array([
      [np.nan, np.nan, 1, np.nan, np.nan],
      [0.2, 0.1, 0.5, 0.3, np.nan],
      [0.6, 0.1, np.nan, 0.3, np.nan],
      [0.6, np.nan, 0.3, np.nan, 0.1],
      [0.4, 0.1, 0.4, 0.1, np.nan]
    ])

  def test_maximum_weight_matching_basic_3(self, basic_valuation_profile_3):
    mwm = MaximumWeightMatching()
    allocation = mwm.scf(basic_valuation_profile_3)
    assert np.all(allocation == np.array([2, 3, 0, 4, 1]))

  @pytest.fixture
  def invalid_valuation_profile_1(self):
    return np.array([
      [np.nan, 1],
      [np.nan, 1]
    ])

  def test_maximum_weight_matching_invalid_1(self, invalid_valuation_profile_1):
    mwm = MaximumWeightMatching()
    with pytest.raises(ValueError):
      mwm.scf(invalid_valuation_profile_1)
