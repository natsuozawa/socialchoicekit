import numpy as np
import pytest

from socialchoicekit.elicitation_utils import *
from socialchoicekit.profile_utils import CompleteValuationProfile

class TestElicitationUtils:
  @pytest.fixture
  def basic_profile_1(self):
    return CompleteValuationProfile.of(np.array([
      [1, 2, 3],
      [3, 1, 2],
      [1, 2, 3],
    ]))

  @pytest.fixture
  def basic_profile_2(self):
    return CompleteValuationProfile.of(np.array([
      [0.1, 0.5, 0.4],
      [0.5, 0, 0.5],
      [0.9, 0.05, 0.05],
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

  def test_valuation_profile_elicitor(self, basic_profile_1, basic_profile_2):
    vpe1 = ValuationProfileElicitor(basic_profile_1)
    assert vpe1.elicit(0, 0) == 1
    assert vpe1.elicit(1, 2) == 2
    vpe2 = ValuationProfileElicitor(basic_profile_2)
    assert vpe2.elicit(0, 0) == 0.1
    assert vpe2.elicit(1, 1) == 0
    assert vpe2.elicit(2, 0) == 0.9

  def test_synchronous_stdin_elicitor(self, agh_course_selection_instance, monkeypatch, capfd):
    ssie = SynchronousStdInElicitor(preflib_instance=agh_course_selection_instance)
    monkeypatch.setattr("builtins.input", lambda: 1)
    assert ssie.elicit(0, 0) == 1
    assert ssie.elicit(0, 1) == 1
    assert ssie.elicit(0, 0) == 1
    assert ssie.elicitation_count == 2
    out, _ = capfd.readouterr()
    assert out == "Agent 1, what is your preference for alternative Course 1?\nAgent 1, what is your preference for alternative Course 2?\n"

  def test_lambda_elicitor(self, basic_profile_1, basic_profile_2):
    vpe1 = ValuationProfileElicitor(basic_profile_1)
    # Use private method here only for test purposes
    le1 = LambdaElicitor(elicitation_function=lambda i, j : vpe1._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
    assert le1.elicit(0, 0) == 1 and le1.elicitation_count == 1
    assert le1.elicit(1, 2) == 2 and le1.elicitation_count == 2
    le1.elicit_multiple(np.array([0, 1]), np.array([0, 2]))
    assert le1.elicitation_count == 2
    vpe2 = ValuationProfileElicitor(basic_profile_2)
    le2 = LambdaElicitor(elicitation_function=lambda i, j : vpe2._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
    assert le2.elicit(0, 0) == 0.1
    assert le2.elicit(1, 1) == 0
    assert le2.elicit(2, 0) == 0.9

  def test_integer_valuation_profile_elicitor(self, basic_profile_1, basic_profile_2):
    ivpe = IntegerValuationProfileElicitor(basic_profile_1)
    assert ivpe.elicit(0, 0) == 1
    assert ivpe.elicit(1, 2) == 2
    ivpe.elicit_multiple(np.array([0, 1]), np.array([0, 2]))
    assert ivpe.elicitation_count == 2

    with pytest.raises(ValueError):
      ivpe2 = IntegerValuationProfileElicitor(basic_profile_2)
      ivpe2.elicit(0, 2)

  def test_integer_synchronous_stdin_elicitor(self, agh_course_selection_instance, monkeypatch, capfd):
    issie1 = IntegerSynchronousStdInElicitor(preflib_instance=agh_course_selection_instance)
    monkeypatch.setattr("builtins.input", lambda: 1)
    assert issie1.elicit(0, 0) == 1
    assert issie1.elicit(0, 1) == 1
    assert issie1.elicit(0, 0) == 1
    assert issie1.elicitation_count == 2
    out1, _ = capfd.readouterr()
    assert out1 == "Agent 1, what is your preference for alternative Course 1?\nAgent 1, what is your preference for alternative Course 2?\n"
    issie2 = IntegerSynchronousStdInElicitor(preflib_instance=agh_course_selection_instance)
    monkeypatch.setattr("builtins.input", lambda: 0.1)
    with pytest.raises(ValueError):
      issie2.elicit(0, 0)

  def test_integer_lambda_elicitor(self, basic_profile_1, basic_profile_2):
    ivpe1 = IntegerValuationProfileElicitor(basic_profile_1)
    ile1 = IntegerLambdaElicitor(elicitation_function=lambda i, j : ivpe1._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
    assert ile1.elicit(0, 0) == 1
    assert ile1.elicit(1, 2) == 2
    ile1.elicit_multiple(np.array([0, 1]), np.array([0, 2]))
    assert ile1.elicitation_count == 2

    with pytest.raises(ValueError):
      ivpe2 = IntegerValuationProfileElicitor(basic_profile_2)
      ile2 = IntegerLambdaElicitor(elicitation_function=lambda i, j : ivpe2._elicit_impl(i - 1, j - 1), memoize=True, zero_indexed=False)
      ile2.elicit(0, 2)
