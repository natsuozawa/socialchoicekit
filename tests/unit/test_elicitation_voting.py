import numpy as np
import pytest

from socialchoicekit.elicitation_voting import LambdaPRV, KARV
from socialchoicekit.elicitation_utils import ValuationProfileElicitor

class TestElicitationVoting:
  @pytest.fixture
  def basic_profile_1(self):
    return np.array([
      [1, 4, 3, 2],
      [4, 2, 1, 3],
      [4, 3, 2, 1],
      [3, 4, 2, 1]
    ])

  @pytest.fixture
  def basic_valuation_profile_1(self):
    return np.array([
      [0.5, 0.1, 0.1, 0.3],
      [0.2, 0.2, 0.4, 0.2],
      [0.1, 0.3, 0.3, 0.3],
      [0.2, 0.1, 0.3, 0.4],
    ])

  def test_lambda_prv_basic_1(self, basic_profile_1, basic_valuation_profile_1):
    lprv_1 = LambdaPRV(lambda_=1)
    lprv_2 = LambdaPRV(lambda_=2, tie_breaker="first")
    lprv_3 = LambdaPRV(lambda_=3)
    lprv_4 = LambdaPRV(lambda_=4)
    vpe = ValuationProfileElicitor(basic_valuation_profile_1)
    score_1 = lprv_1.score(basic_profile_1, vpe)
    assert np.allclose(score_1, np.array([0.5, 0, 0.4, 0.7]))
    assert lprv_1.scf(basic_profile_1, vpe) == 4
    score_2 = lprv_2.score(basic_profile_1, vpe)
    assert np.allclose(score_2, np.array([0.5, 0.2, 1.0, 1.0]))
    assert lprv_2.scf(basic_profile_1, vpe) == 3
    score_3 = lprv_3.score(basic_profile_1, vpe)
    assert np.allclose(score_3, np.array([0.7, 0.5, 1.1, 1.2]))
    assert lprv_3.scf(basic_profile_1, vpe) == 4
    score_4 = lprv_4.score(basic_profile_1, vpe)
    assert np.allclose(score_4, np.array([1.0, 0.7, 1.1, 1.2]))
    assert lprv_4.scf(basic_profile_1, vpe) == 4

  def test_lambda_prv_with_invalid_lambda(self, basic_profile_1, basic_valuation_profile_1):
    with pytest.raises(ValueError):
      LambdaPRV(lambda_=0)
    with pytest.raises(ValueError):
      lprv_5 = LambdaPRV(lambda_=5)
      vpe = ValuationProfileElicitor(basic_valuation_profile_1)
      lprv_5.scf(basic_profile_1, vpe)

  def test_karv_basic_1(self, basic_profile_1, basic_valuation_profile_1):
    karv_1 = KARV(k = 1)
    karv_2 = KARV(k = 2)
    vpe = ValuationProfileElicitor(basic_valuation_profile_1)
    assert karv_1.scf(basic_profile_1, vpe) == 4
    karv_2 = KARV(k = 2)
    assert karv_2.scf(basic_profile_1, vpe) == 4

  def test_karv_with_invalid_k(self, basic_profile_1, basic_valuation_profile_1):
    with pytest.raises(ValueError):
      KARV(k = 0)
    with pytest.raises(ValueError):
      karv_5 = KARV(k = 5)
      vpe = ValuationProfileElicitor(basic_valuation_profile_1)
      karv_5.scf(basic_profile_1, vpe)

