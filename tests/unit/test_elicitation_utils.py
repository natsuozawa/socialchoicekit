import numpy as np
import pytest

from socialchoicekit.elicitation_utils import ValuationProfileElicitor, SynchronousStdInElicitor, LambdaElicitor
from socialchoicekit.profile_utils import CompleteValuationProfile

class TestElicitationUtils:
  @pytest.fixture
  def basic_profile_1(self):
    return CompleteValuationProfile.of(np.array([
      [1, 2, 3],
      [3, 1, 2],
      [1, 2, 3],
    ]))

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

  def test_synchronous_stdin_elicitor(self, agh_course_selection_instance, monkeypatch, capfd):
    ssie = SynchronousStdInElicitor(preflib_instance=agh_course_selection_instance)
    monkeypatch.setattr("builtins.input", lambda: 1)
    assert ssie.elicit(0, 0) == 1
    assert ssie.elicit(0, 1) == 1
    assert ssie.elicit(0, 0) == 1
    assert ssie.elicitation_count == 2
    out, _ = capfd.readouterr()
    assert out == "Agent 1, what is your preference for alternative Course 1?\nAgent 1, what is your preference for alternative Course 2?\n"


  def test_lambda_elicitor(self, basic_profile_1):
    vpe = ValuationProfileElicitor(basic_profile_1)
    # Use private method here only for test purposes
    le = LambdaElicitor(elicitation_function=lambda i, j : vpe._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
    assert le.elicit(0, 0) == 1 and le.elicitation_count == 1
    assert le.elicit(1, 2) == 2 and le.elicitation_count == 2
    le.elicit_multiple(np.array([0, 1]), np.array([0, 2]))
    assert le.elicitation_count == 2




