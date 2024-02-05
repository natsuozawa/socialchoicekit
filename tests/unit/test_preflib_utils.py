import pytest

from socialchoicekit.preflib_utils import *
from socialchoicekit.utils import check_profile

class TestPrefLibUtils:
  def test_preflib_soc_to_profile(self, agh_course_selection_instance):
    soc = agh_course_selection_instance
    profile = preflib_soc_to_profile(soc)
    check_profile(profile, is_complete=True)
    assert profile.shape == (agh_course_selection_instance.num_voters, agh_course_selection_instance.num_alternatives)

  def test_preflib_soi_to_profile(self, apa_election_instance):
    soi = apa_election_instance
    profile = preflib_soi_to_profile(soi)
    check_profile(profile, is_complete=False)
    assert profile.shape == (apa_election_instance.num_voters, apa_election_instance.num_alternatives)

  def test_preflib_soc_to_profile_with_invalid_instance(self, apa_election_instance):
    with pytest.raises(ValueError):
      preflib_soc_to_profile(apa_election_instance)

  def test_preflib_soi_to_profile_with_invalid_instance(self, agh_course_selection_instance):
    with pytest.raises(ValueError):
      preflib_soi_to_profile(agh_course_selection_instance)
