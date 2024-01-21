import numpy as np
import pytest

from socialchoicekit.elicitation_allocation import LambdaTSF
from socialchoicekit.elicitation_utils import ValuationProfileElicitor

class TestElicitationAllocation:
  @pytest.fixture
  def basic_profile_1(self):
    return np.array([
      [1, 2, 4, 3],
      [2, 1, 3, 4],
      [3, 4, 1, 2],
      [4, 3, 2, 1],
    ])

  @pytest.fixture
  def basic_valuation_profile_1(self):
    return np.array([
      [0.5, 0.3, 0.1, 0.2],
      [0.1, 0.7, 0.1, 0.1],
      [0.2, 0.1, 0.4, 0.3],
      [0.1, 0.1, 0.4, 0.4],
    ])

  def test_lambda_tsf_basic_1(self, basic_profile_1, basic_valuation_profile_1):
    ltsf = LambdaTSF(lambda_=2)
    vpe = ValuationProfileElicitor(basic_valuation_profile_1)
    allocation = ltsf.scf(basic_profile_1, vpe)
    assert np.all(allocation == np.array([1, 2, 3, 4]))

  @pytest.fixture
  def basic_profile_3(self):
    return np.array([
      [np.nan, np.nan, 1, np.nan, np.nan],
      [3, 4, 1, 2, np.nan],
      [1, 3, np.nan, 2, np.nan],
      [1, np.nan, 2, np.nan, 3],
      [2, 4, 1, 3, np.nan]
    ])

  @pytest.fixture
  def basic_valuation_profile_3(self):
    return np.array([
      [np.nan, np.nan, 1, np.nan, np.nan],
      [0.2, 0.1, 0.5, 0.3, np.nan],
      [0.6, 0.1, np.nan, 0.3, np.nan],
      [0.6, np.nan, 0.3, np.nan, 0.3],
      [0.4, 0.1, 0.4, 0.1, np.nan],
    ])

  def test_lambda_tsf_basic_3(self, basic_profile_3, basic_valuation_profile_3):
    ltsf = LambdaTSF(lambda_=3)
    vpe = ValuationProfileElicitor(basic_valuation_profile_3)
    allocation = ltsf.scf(basic_profile_3, vpe)
    assert np.all(allocation == np.array([3, 4, 1, 5, 2]))

  @pytest.fixture
  def invalid_profile_1(self):
    return np.array([
      [np.nan, 1],
      [np.nan, 1],
    ])
  @pytest.fixture
  def invalid_valuation_profile_1(self):
    return np.array([
      [np.nan, 1],
      [np.nan, 1],
    ])

  def test_lambda_tsf_invalid_1(self,invalid_profile_1,  invalid_valuation_profile_1):
    ltsf = LambdaTSF(lambda_=1)
    vpe = ValuationProfileElicitor(invalid_valuation_profile_1)
    with pytest.raises(ValueError):
      ltsf.scf(invalid_profile_1, vpe)
