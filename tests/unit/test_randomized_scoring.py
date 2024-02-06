import numpy as np

import pytest

from socialchoicekit.randomized_scoring import *
from socialchoicekit.profile_utils import StrictCompleteProfile, CompleteValuationProfile

class TestRandomizedScoring:
  @pytest.fixture
  def profile_1(self):
    return StrictCompleteProfile.of(np.array([
      [1, 2, 3, 4],
      [1, 3, 2, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ]))

  def test_plurality_1(self, profile_1):
    voting_rule = RandomizedPlurality()
    assert voting_rule.scf(profile_1) == 1

  def test_veto_1(self, profile_1):
    voting_rule = RandomizedVeto()
    assert voting_rule.scf(profile_1) != 4
