import numpy as np
import pytest

from socialchoicekit.elicitation_utils import ValuationProfileElicitor, LambdaElicitor

class TestElicitationUtils:
  @pytest.fixture
  def basic_profile_1(self):
    return np.array([
      [1, 2, 3],
      [3, 1, 2],
      [1, 2, 3],
    ])

  def test_memoization(self, basic_profile_1):
    vpe_1 = ValuationProfileElicitor(basic_profile_1)
    vpe_1.elicit_multiple(np.array([0, 1, 1]), np.array([0, 2, 0]))
    assert vpe_1.elicitation_count == 3
    vpe_1.elicit(0, 0)
    assert vpe_1.elicitation_count == 3

    vpe_2 = ValuationProfileElicitor(basic_profile_1, memoize=False)
    vpe_2.elicit_multiple(np.array([0, 1, 1]), np.array([0, 2, 0]))
    assert vpe_2.elicitation_count == 3
    vpe_2.elicit(0, 0)
    assert vpe_2.elicitation_count == 4

  def test_lambda_elicitor(self, basic_profile_1):
    vpe = ValuationProfileElicitor(basic_profile_1)
    # Use private method here only for test purposes
    le = LambdaElicitor(elicitation_function=lambda i, j : vpe._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
    assert le.elicit(0, 0) == 1 and le.elicitation_count == 1
    assert le.elicit(1, 2) == 2 and le.elicitation_count == 2
    le.elicit_multiple(np.array([0, 1]), np.array([0, 2]))
    assert le.elicitation_count == 2




