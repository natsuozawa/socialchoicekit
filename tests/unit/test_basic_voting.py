from socialchoicekit.deterministic_scoring import Plurality
import numpy as np
from typing import cast
import pytest

class TestBasicVoting:
  @pytest.fixture
  def profile_single(self):
    return np.array([[1]])

  @pytest.fixture
  def profile_empty(self):
    return np.array([[]])

  @pytest.fixture
  def profile_1d(self):
    return np.array([1, 2, 3, 4, 5])

  @pytest.fixture
  def profile_3d(self):
    return np.array([
      [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
      [[1, 3, 2], [3, 1, 2], [3, 1, 2]],
      [[2, 1, 3], [2, 1, 3], [2, 3, 1]],
    ])

  @pytest.fixture
  def profile_repeat(self):
    return np.array([
      [1, 2, 4, 3, 8, 5, 3, 7],
      [4, 5, 1, 2, 4, 6, 8, 3],
      [3, 7, 1, 2, 4, 6, 8, 5],
    ])

  @pytest.fixture
  def profile_negative(self, profile_a):
    return profile_a - 2

  @pytest.fixture
  def profile_invalid_alternative(self, profile_a):
    return profile_a + 1

  @pytest.fixture
  def profile_tie(self):
    return np.array([
      [1, 2],
      [2, 1],
    ])

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
