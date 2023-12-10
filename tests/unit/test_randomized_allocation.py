import pytest
import numpy as np

from socialchoicekit.randomized_allocation import RandomSerialDictatorship

class TestRandomizedAllocation:
  @pytest.fixture
  def basic_preference_list_1(self):
    return np.array([
      [1, 2, 3, 4],
      [2, 1, np.nan, np.nan],
      [3, np.nan, 2, 1]
    ])

  def test_random_serial_dictatorship_basic_1(self, basic_preference_list_1):
    rsd = RandomSerialDictatorship()
    allocation = rsd.scf(basic_preference_list_1)
    assert np.all(allocation == np.array([1, 2, 4]))


  @pytest.fixture
  def basic_preference_list_2(self):
    return np.array([
      [1, 2, 3, 4],
      [2, 1, np.nan, np.nan],
      [3, np.nan, 2, 1],
      [np.nan, np.nan, np.nan, np.nan]
    ])

  def test_random_serial_dictatorship_basic_2(self, basic_preference_list_2):
    rsd = RandomSerialDictatorship()
    allocation = rsd.scf(basic_preference_list_2)
    assert np.all(allocation[0:3] == np.array([1, 2, 4])) and np.isnan(allocation[3])
