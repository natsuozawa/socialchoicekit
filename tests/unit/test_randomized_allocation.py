import pytest
import numpy as np

from socialchoicekit.randomized_allocation import RandomSerialDictatorship, ProbabilisticSerial, SimultaneousEating
from socialchoicekit.profile_utils import StrictIncompleteProfile

class TestRandomizedAllocation:
  @pytest.fixture
  def basic_profile_1(self):
    return StrictIncompleteProfile.of(np.array([
      [1, 2, 3, 4],
      [2, 1, np.nan, np.nan],
      [3, np.nan, 2, 1],
    ]))

  def test_random_serial_dictatorship_basic_1(self, basic_profile_1):
    rsd = RandomSerialDictatorship()
    allocation = rsd.scf(basic_profile_1)
    assert np.all(allocation == np.array([1, 2, 4]))

  @pytest.fixture
  def basic_profile_2(self):
    return StrictIncompleteProfile.of(np.array([
      [1, 2, 3, 4],
      [2, 1, np.nan, np.nan],
      [3, np.nan, 2, 1],
      [np.nan, np.nan, np.nan, np.nan],
    ]))

  def test_random_serial_dictatorship_basic_2(self, basic_profile_2):
    rsd = RandomSerialDictatorship()
    allocation = rsd.scf(basic_profile_2)
    assert np.all(allocation[0:3] == np.array([1, 2, 4])) and np.isnan(allocation[3])

  @pytest.fixture
  def basic_profile_3(self):
    return StrictIncompleteProfile.of(np.array([
      [1.0, 2.0, 3.0, 4.0],
      [1, 2, 3, np.nan],
      [3, np.nan, 2, 1],
      [2, 1, 3, np.nan],
    ]))

  def test_probabilistic_serial_3(self, basic_profile_3):
    ps = ProbabilisticSerial()
    bistochastic = ps.bistochastic(basic_profile_3)
    assert np.all(bistochastic == np.array([
      [1/2, 1/6, 1/3, 0],
      [1/2, 1/6, 1/3, 0],
      [0, 0, 0, 1],
      [0, 2/3, 1/3, 0],
    ]))

  @pytest.fixture
  def speeds_3(self):
    return np.array([1, 1, 2, 2])

  def test_simultaneous_eating_3(self, basic_profile_3, speeds_3):
    se = SimultaneousEating()
    bistochastic = se.bistochastic(basic_profile_3, speeds_3)
    assert np.all(bistochastic == np.array([
      [1/2, 0, 1/2, 0],
      [1/2, 0, 1/2, 0],
      [0, 0, 0, 1],
      [0, 1, 0, 0],
    ]))
