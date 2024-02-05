import pytest

from socialchoicekit.preflib_utils import *
from socialchoicekit.utils import check_profile

class TestPrefLibUtils:
  def test_preflib_soc_to_profile(self, agh_course_selection_instance):
    soc = agh_course_selection_instance
    profile = preflib_soc_to_profile(soc)
    check_profile(profile, is_complete=True)
    assert profile.shape == (agh_course_selection_instance.num_voters, agh_course_selection_instance.num_alternatives)

  def test_preflib_soc_to_profile_with_invalid_instance(self, apa_election_instance):
    with pytest.raises(ValueError):
      preflib_soc_to_profile(apa_election_instance)

  def test_preflib_soi_to_profile(self, apa_election_instance):
    soi = apa_election_instance
    profile_1 = preflib_soi_to_profile(soi)
    check_profile(profile_1, is_complete=False)
    assert profile_1.shape == (apa_election_instance.num_voters, apa_election_instance.num_alternatives)
    profile_2 = preflib_soi_to_profile(soi)
    check_profile(profile_2, is_complete=False)
    assert profile_2.shape == (apa_election_instance.num_voters, apa_election_instance.num_alternatives)

  def test_preflib_soi_to_profile_with_invalid_instance(self, agh_course_selection_instance):
    with pytest.raises(ValueError):
      preflib_soi_to_profile(agh_course_selection_instance)

  def test_preflib_toc_to_profile(self, burlington_election_instance):
    toc = burlington_election_instance
    profile = preflib_toc_to_profile(toc, tie_breaker="first")
    check_profile(profile, is_complete=True)
    assert profile.shape == (burlington_election_instance.num_voters, burlington_election_instance.num_alternatives)

  def test_preflib_toc_to_profile_with_invalid_instance(self, apa_election_instance):
    with pytest.raises(ValueError):
      preflib_toc_to_profile(apa_election_instance, tie_breaker="first")

  def test_preflib_toi_to_profile(self, aspen_election_instance):
    toi = aspen_election_instance
    profile_1 = preflib_toi_to_profile(toi, tie_breaker="first")
    check_profile(profile_1, is_complete=False)
    assert profile_1.shape == (aspen_election_instance.num_voters, aspen_election_instance.num_alternatives)
    profile_2 = preflib_toi_to_profile(toi, tie_breaker="random")
    check_profile(profile_2, is_complete=False)
    assert profile_2.shape == (aspen_election_instance.num_voters, aspen_election_instance.num_alternatives)

  def test_preflib_toi_to_profile_with_invalid_instance(self, agh_course_selection_instance):
    with pytest.raises(ValueError):
      preflib_toi_to_profile(agh_course_selection_instance, tie_breaker="first")
