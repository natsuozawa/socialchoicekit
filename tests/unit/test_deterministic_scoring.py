import pytest
import numpy as np

from socialchoicekit.deterministic_scoring import *

class TestDeterministicScoring():
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
