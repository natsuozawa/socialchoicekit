import numpy as np

import pytest

from socialchoicekit.data_generation import *
from socialchoicekit.utils import check_valuation_profile
from socialchoicekit.profile_utils import StrictCompleteProfile, StrictIncompleteProfile, IncompleteValuationProfile, compute_ordinal_profile

class TestDataGeneration:
  @pytest.fixture
  def ordinal_profile_1(self):
    return StrictIncompleteProfile.of(np.array([
      [1, 5, 2, 3, 4],
      [4, 5, 3, 1, 2],
      [1, np.nan, 4, 2, 3]
    ]))

  @pytest.fixture
  def cardinal_profile_1(self):
    return IncompleteValuationProfile.of(np.array([
      [0.9, 0, 0.05, 0.04, 0.01],
      [0.1, 0.05, 0.15, 0.4, 0.3],
      [0.7, np.nan, 0.01, 0.2, 0.09]
    ]))

  @pytest.fixture
  def ordinal_profile_2(self):
    return StrictCompleteProfile.of(np.array([
      [1, 4, 2, 3],
      [1, 3, 4, 2],
    ]))

  def test_uniform_valuation_profile_generator_1(self, ordinal_profile_1):
    uvpg = UniformValuationProfileGenerator(high=1, low=0)
    valuation_profile = uvpg.generate(ordinal_profile_1)
    check_valuation_profile(valuation_profile, is_complete=False)
    assert np.allclose(np.ones(ordinal_profile_1.shape[0]), np.nansum(valuation_profile, axis=1))
    assert np.allclose(compute_ordinal_profile(valuation_profile), ordinal_profile_1, equal_nan=True)

  def test_normal_valuation_profile_generator_1(self, ordinal_profile_1):
    # Use a small variance so we can reliably compute the ordinal profile
    # (We cannot do this when multiple values are clipped to zero)
    nvpg = NormalValuationProfileGenerator(mean=0.5, variance=0.01)
    valuation_profile = nvpg.generate(ordinal_profile_1)
    check_valuation_profile(valuation_profile, is_complete=False)
    assert np.allclose(np.ones(ordinal_profile_1.shape[0]), np.nansum(valuation_profile, axis=1))
    assert np.allclose(compute_ordinal_profile(valuation_profile), ordinal_profile_1, equal_nan=True)

  def test_uniform_valuation_profile_generator_2(self, ordinal_profile_2):
    uvpg = UniformValuationProfileGenerator(high=1, low=0)
    valuation_profile = uvpg.generate(ordinal_profile_2)
    check_valuation_profile(valuation_profile, is_complete=True)
    assert np.allclose(np.ones(ordinal_profile_2.shape[0]), np.nansum(valuation_profile, axis=1))

  def test_normal_valuation_profile_generator_2(self, ordinal_profile_2):
    nvpg = NormalValuationProfileGenerator(mean=0.5, variance=0.01)
    valuation_profile = nvpg.generate(ordinal_profile_2)
    check_valuation_profile(valuation_profile, is_complete=True)
    assert np.allclose(np.ones(ordinal_profile_2.shape[0]), np.nansum(valuation_profile, axis=1))

  def test_uniform_valuation_profile_generator_invalid_range(self, ordinal_profile_1):
    with pytest.raises(ValueError):
      UniformValuationProfileGenerator(high=-1, low=1)
