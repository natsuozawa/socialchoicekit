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
    profile = preflib_soi_to_profile(soi)
    check_profile(profile, is_complete=False)
    assert profile.shape == (apa_election_instance.num_voters, apa_election_instance.num_alternatives)

  def test_preflib_soi_to_profile_with_invalid_instance(self, agh_course_selection_instance):
    with pytest.raises(ValueError):
      preflib_soi_to_profile(agh_course_selection_instance)

  def test_preflib_toc_to_profile(self, burlington_election_instance):
    toc = burlington_election_instance
    profile_1 = preflib_toc_to_profile(toc, tie_breaker="first")
    check_profile(profile_1, is_complete=True, is_strict=True)
    assert profile_1.shape == (burlington_election_instance.num_voters, burlington_election_instance.num_alternatives)
    profile_2 = preflib_toc_to_profile(toc, tie_breaker="random")
    check_profile(profile_2, is_complete=True, is_strict=True)
    assert profile_2.shape == (burlington_election_instance.num_voters, burlington_election_instance.num_alternatives)
    profile_3 = preflib_toc_to_profile(toc, tie_breaker="accept")
    check_profile(profile_3, is_complete=True, is_strict=False)
    # check_profile may be True with is_strict=False but not necessarily
    assert profile_3.shape == (burlington_election_instance.num_voters, burlington_election_instance.num_alternatives)

  def test_preflib_toc_to_profile_with_invalid_instance(self, apa_election_instance):
    with pytest.raises(ValueError):
      preflib_toc_to_profile(apa_election_instance, tie_breaker="first")

  def test_preflib_toi_to_profile(self, aspen_election_instance):
    toi = aspen_election_instance
    profile_1 = preflib_toi_to_profile(toi, tie_breaker="first")
    check_profile(profile_1, is_complete=False, is_strict=True)
    assert profile_1.shape == (aspen_election_instance.num_voters, aspen_election_instance.num_alternatives)
    profile_2 = preflib_toi_to_profile(toi, tie_breaker="random")
    check_profile(profile_2, is_complete=False, is_strict=True)
    assert profile_2.shape == (aspen_election_instance.num_voters, aspen_election_instance.num_alternatives)
    profile_3 = preflib_toi_to_profile(toi, tie_breaker="accept")
    check_profile(profile_3, is_complete=False, is_strict=False)
    assert profile_3.shape == (aspen_election_instance.num_voters, aspen_election_instance.num_alternatives)

  def test_preflib_toi_to_profile_with_invalid_instance(self, agh_course_selection_instance):
    with pytest.raises(ValueError):
      preflib_toi_to_profile(agh_course_selection_instance, tie_breaker="first")

  def test_preflib_categorical_to_profile(self, french_president_election_instance):
    cat = french_president_election_instance
    profile_1 = preflib_categorical_to_profile(cat, tie_breaker="first")
    # This dataset happens to be complete. Categorical preference data can be incomplete in general.
    check_profile(profile_1, is_complete=True)
    assert profile_1.shape == (french_president_election_instance.num_voters, french_president_election_instance.num_alternatives)
    profile_2 = preflib_categorical_to_profile(cat, tie_breaker="random")
    check_profile(profile_2, is_complete=True)
    assert profile_2.shape == (french_president_election_instance.num_voters, french_president_election_instance.num_alternatives)
    profile_3 = preflib_categorical_to_profile(cat, tie_breaker="accept")
    check_profile(profile_3, is_complete=True, is_strict=False)
    assert profile_3.shape == (french_president_election_instance.num_voters, french_president_election_instance.num_alternatives)
