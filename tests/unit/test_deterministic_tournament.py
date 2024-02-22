import numpy as np

from preflibtools.properties.pairwisecomparisons import copeland_scores

from socialchoicekit.deterministic_tournament import Copeland
from socialchoicekit.preflib_utils import preflib_soc_to_profile

class TestDeterministicTournament:
  def test_copeland_1(self, profile_a):
    voting_rule = Copeland(tie_breaker="first")
    score = voting_rule.score(profile_a)
    # Compare against hand calculated Borda scores
    assert score[0] == -2
    assert score[1] == -1

  def test_preflib_soc_copeland(self, agh_course_selection_instance):
    soc = agh_course_selection_instance
    voting_rule = Copeland(tie_breaker="first")
    profile = preflib_soc_to_profile(soc)
    score = voting_rule.score(profile)
    preflib_copeland_score_map = copeland_scores(soc)
    preflib_copeland_score = np.zeros(soc.num_alternatives)
    for i in range(soc.num_alternatives):
      for j in range(soc.num_alternatives):
        if i == j:
          continue
        if preflib_copeland_score_map[i + 1][j + 1] > 0:
          preflib_copeland_score[i] += 1
        elif preflib_copeland_score_map[i + 1][j + 1] < 0:
          preflib_copeland_score[i] -= 1
    assert np.array_equal(score, preflib_copeland_score)
