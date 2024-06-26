import numpy as np
import pytest

from preflibtools.instances import OrdinalInstance, CategoricalInstance

from socialchoicekit.profile_utils import StrictCompleteProfile
@pytest.fixture
def profile_a():
  return StrictCompleteProfile.of(np.array([
    [1, 2, 4, 6, 8, 5, 3, 7],
    [4, 5, 1, 2, 7, 3, 8, 6],
    [4, 2, 3, 5, 1, 6, 7, 8],
    [8, 7, 6, 5, 4, 3, 2, 1],
    [8, 4, 2, 5, 1, 7, 6, 3],
    [5, 8, 1, 4, 2, 7, 6, 3],
  ]))

@pytest.fixture
def profile_b():
  return StrictCompleteProfile.of(np.array([
    [3, 2, 4, 6, 8, 5, 1, 7],
    [3, 6, 7, 2, 4, 5, 1, 8],
    [1, 8, 7, 3, 4, 5, 2, 6],
    [5, 6, 1, 4, 8, 7, 3, 2],
    [3, 6, 7, 2, 4, 5, 1, 8],
    [4, 3, 2, 1, 5, 6, 7, 8],
  ]))

@pytest.fixture
def agh_course_selection_instance():
  instance = OrdinalInstance()
  instance.parse_url("https://www.preflib.org/static/data/agh/00009-00000001.soc")
  return instance

@pytest.fixture
def apa_election_instance():
  instance = OrdinalInstance()
  instance.parse_url("https://www.preflib.org/static/data/apa/00028-00000001.soi")
  return instance

@pytest.fixture
def burlington_election_instance():
  instance = OrdinalInstance()
  instance.parse_url("https://www.preflib.org/static/data/burlington/00005-00000001.toc")
  return instance

@pytest.fixture
def aspen_election_instance():
  instance = OrdinalInstance()
  instance.parse_url("https://www.preflib.org/static/data/aspen/00016-00000002.toi")
  return instance

@pytest.fixture
def french_president_election_instance():
  instance = CategoricalInstance()
  instance.parse_url("https://www.preflib.org/static/data/frenchapproval/00026-00000001.cat")
  return instance
