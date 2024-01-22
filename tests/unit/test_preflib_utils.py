from socialchoicekit.preflib_utils import *
from socialchoicekit.utils import check_profile

class TestPrefLibUtils:
  def test_preflib_soc_to_profile(self, agh_course_selection_instance):
    soc = agh_course_selection_instance.flatten_strict()
    profile = preflib_soc_to_profile(soc)
    check_profile(profile)
    assert profile.shape == (agh_course_selection_instance.num_voters, agh_course_selection_instance.num_alternatives)
