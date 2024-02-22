import numpy as np
import pytest

from socialchoicekit.profile_utils import *

class TestProfileUtils:
  @pytest.fixture
  def strict_incomplete_profile_1(self):
    return StrictIncompleteProfile.of(np.array([
      [1, np.nan, 2, 3, np.nan],
      [3, 4, np.nan, 1, 2],
      [1, 5, 4, 3, 2],
      [np.nan, np.nan, np.nan, np.nan, 1],
      [4, 2, 3, 1, np.nan]
    ]))

  @pytest.fixture
  def strict_complete_profile_1(self):
    return StrictCompleteProfile.of(np.array([
      [1, 4, 2, 3, 5],
      [3, 4, 5, 1, 2],
      [1, 5, 4, 3, 2],
      [2, 3, 4, 5, 1],
      [4, 2, 3, 1, 5],
    ]))

  @pytest.fixture
  def complete_profile_with_ties_1(self):
    return CompleteProfileWithTies.of(np.array([
      [1, 4, 2, 3, 4],
      [3, 4, 5, 1, 2],
      [1, 5, 4, 3, 2],
      [2, 2, 2, 2, 1],
      [4, 2, 3, 1, 5],
    ]))

  @pytest.fixture
  def incomplete_profile_with_ties_1(self):
    # Completing this profile will NOT result in complete_profile_with_ties_1
    return IncompleteProfileWithTies.of(np.array([
      [1, np.nan, 2, 2, np.nan],
      [3, 4, np.nan, 1, 1],
      [1, 5, 4, 3, 2],
      [np.nan, np.nan, np.nan, np.nan, 1],
      [4, 2, 3, 1, np.nan],
    ]))

  def test_incomplete_profile_to_complete_profile_1(
    self,
    strict_incomplete_profile_1,
    strict_complete_profile_1,
    incomplete_profile_with_ties_1,
    complete_profile_with_ties_1,
  ):
    completed_profile_first = incomplete_profile_to_complete_profile(strict_incomplete_profile_1, tie_breaker="first")
    assert np.all(completed_profile_first == strict_complete_profile_1)
    assert isinstance(completed_profile_first, StrictCompleteProfile)
    completed_profile_accept = incomplete_profile_to_complete_profile(strict_incomplete_profile_1, tie_breaker="accept")
    assert np.all(completed_profile_accept == complete_profile_with_ties_1)
    assert isinstance(completed_profile_accept, CompleteProfileWithTies)

    completed_profile_random = incomplete_profile_to_complete_profile(incomplete_profile_with_ties_1, tie_breaker="random")
    assert np.amax(completed_profile_random[0, :]) == 5
    assert np.all(completed_profile_random[0, :] != 3)
    assert np.all(completed_profile_random[1, :] != 2)
    assert np.all(completed_profile_random[4] == np.array([4, 2, 3, 1, 5]))
    assert isinstance(completed_profile_random, CompleteProfileWithTies)

  def test_profile_with_ties_to_strict_profile_1(
    self,
    complete_profile_with_ties_1,
    strict_complete_profile_1,
    incomplete_profile_with_ties_1,
    strict_incomplete_profile_1,
  ):
    strict_complete_profile_first = profile_with_ties_to_strict_profile(complete_profile_with_ties_1, tie_breaker="first")
    assert np.all(strict_complete_profile_first == strict_complete_profile_1)
    assert isinstance(strict_complete_profile_first, StrictCompleteProfile)

    strict_incomeplete_profile_first = profile_with_ties_to_strict_profile(incomplete_profile_with_ties_1, tie_breaker="first")
    assert np.array_equal(strict_incomeplete_profile_first,strict_incomplete_profile_1, equal_nan=True)
    assert isinstance(strict_incomeplete_profile_first, StrictIncompleteProfile)

    strict_complete_profile_random = profile_with_ties_to_strict_profile(complete_profile_with_ties_1, tie_breaker="random")
    assert np.sum(strict_complete_profile_random) == np.sum(strict_complete_profile_1)
