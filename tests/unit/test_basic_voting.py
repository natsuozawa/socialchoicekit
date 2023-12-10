from socialchoicekit.deterministic_scoring import Plurality
import numpy as np
from typing import cast
import pytest

class TestBasicVoting:
  def test_tie_breaker(self, profile_tie):
    voting_rule_without_tie_breaker = Plurality(tie_breaker="accept")
    # We safely assume that scf does not return int when tie_breaker is set to "accept"
    winners: np.ndarray = cast(np.ndarray, voting_rule_without_tie_breaker.scf(profile_tie))
    assert set(winners) == set(profile_tie[0])
    voting_rule_random = Plurality(tie_breaker="random")
    winner = voting_rule_random.scf(profile_tie)
    new_winner = winner
    assert winner in winners
    for _ in range(int(1e8)):
      # Check to see that there is a different winner selected at least once
      new_winner = voting_rule_random.scf(profile_tie)
      if new_winner != winner:
        break
    assert new_winner != winner
    assert new_winner in winners

  def test_zero_indexed(self, profile_b):
    voting_rule_zero_indexed = Plurality(zero_indexed=True)
    voting_rule_one_indexed = Plurality(zero_indexed=False)
    winner_zero_indexed = voting_rule_zero_indexed.scf(profile_b)
    winner_one_indexed = voting_rule_one_indexed.scf(profile_b)
    assert winner_zero_indexed == winner_one_indexed - 1

  def test_empty(self, profile_empty):
    voting_rule = Plurality()
    with pytest.raises(ValueError):
      voting_rule.scf(profile_empty)

  def test_1d(self, profile_1d):
    voting_rule = Plurality()
    with pytest.raises(ValueError):
      voting_rule.scf(profile_1d)

  def test_3d(self, profile_3d):
    voting_rule = Plurality()
    with pytest.raises(ValueError):
      voting_rule.scf(profile_3d)

  # def test_repeat(self, profile_repeat):
  #   voting_rule = Plurality()
  #   with pytest.raises(ValueError):
  #     voting_rule.scf(profile_repeat)

  def test_negative(self, profile_negative):
    voting_rule = Plurality()
    with pytest.raises(ValueError):
      voting_rule.scf(profile_negative)

  def test_invalid_alternative(self, profile_invalid_alternative):
    voting_rule = Plurality()
    with pytest.raises(ValueError):
      voting_rule.scf(profile_invalid_alternative)
