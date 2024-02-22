import numpy as np

from socialchoicekit.deterministic_multiround import SingleTransferableVote

class TestDeterministicMultiround:
  def test_single_transferable_vote_a(self, profile_a):
    voting_rule = SingleTransferableVote(tie_breaker="first")
    # Compare with hand calculated winner
    winner = voting_rule.scf(profile_a)
    assert winner == 5

  def test_single_transferable_vote_b(self, profile_b):
    voting_rule = SingleTransferableVote(tie_breaker="first")
    # Compare with hand calculated winner
    winner = voting_rule.scf(profile_b)
    assert winner == 7
