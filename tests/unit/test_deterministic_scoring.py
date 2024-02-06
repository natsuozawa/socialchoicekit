import pytest
import numpy as np

from socialchoicekit.deterministic_scoring import *
from socialchoicekit.profile_utils import IncompleteValuationProfile

class TestDeterministicScoring:
  def test_plurality_a(self, profile_a):
    voting_rule = Plurality(tie_breaker="first")
    assert voting_rule.scf(profile_a) == 3

  def test_plurality_b(self, profile_b):
    voting_rule = Plurality(tie_breaker="first")
    assert voting_rule.scf(profile_b) == 7

  def test_borda_a(self, profile_a):
    voting_rule = Borda(tie_breaker="first")
    assert voting_rule.scf(profile_a) == 3

  def test_borda_b(self, profile_b):
    voting_rule = Borda(tie_breaker="first")
    assert voting_rule.scf(profile_b) == 7

  def test_veto_a(self, profile_a):
    voting_rule = Veto(tie_breaker="first")
    assert voting_rule.scf(profile_a) == 3

  def test_veto_b(self, profile_b):
    voting_rule = Veto(tie_breaker="first")
    assert voting_rule.scf(profile_b) == 1

  def test_k_approval_a(self, profile_a):
    voting_rule = KApproval(k=3, tie_breaker="first")
    assert voting_rule.scf(profile_a) == 3

  def test_k_approval_b(self, profile_b):
    voting_rule = KApproval(k=3, tie_breaker="first")
    assert voting_rule.scf(profile_b) == 7

  def test_harmonic_a(self, profile_a):
    voting_rule = Harmonic(tie_breaker="first")
    assert voting_rule.scf(profile_a) == 3

  def test_harmonic_b(self, profile_b):
    voting_rule = Harmonic(tie_breaker="first")
    assert voting_rule.scf(profile_b) == 7

  @pytest.fixture
  def cardinal_profile_1(self):
    return IncompleteValuationProfile.of(np.array([
      [0.9, 0, 0.05, 0.04, 0.01],
      [0.1, 0.05, 0.15, 0.4, 0.3],
      [0.7, np.nan, 0.01, 0.2, 0.09]
    ]))

  @pytest.fixture
  def social_welfare_1(self):
    unnormalized_social_welfare = np.array([1.7, 0.05, 0.21, 0.64, 0.4])
    return unnormalized_social_welfare / np.sum(unnormalized_social_welfare)

  def test_social_welfare_1(self, cardinal_profile_1, social_welfare_1):
    sw = SocialWelfare()
    score = sw.score(cardinal_profile_1)
    assert np.allclose(score, social_welfare_1)
