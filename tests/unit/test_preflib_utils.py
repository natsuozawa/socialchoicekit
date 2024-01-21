import pytest
from preflibtools.instances import OrdinalInstance

from socialchoicekit.preflib_utils import *
from socialchoicekit.utils import check_profile

class TestPrefLibUtils:
  @pytest.fixture
  def agh_course_selection_instance(self):
    instance = OrdinalInstance()
    instance.parse_url("https://www.preflib.org/static/data/agh/00009-00000001.soc")
    return instance

  def test_preflib_soc_to_profile(self, agh_course_selection_instance):
    soc = agh_course_selection_instance.flatten_strict()
    profile = preflib_soc_to_profile(soc)
    check_profile(profile)
    assert profile.shape == (agh_course_selection_instance.num_voters, agh_course_selection_instance.num_alternatives)
