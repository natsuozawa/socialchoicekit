import numpy as np
import pytest

from preflibtools.instances import OrdinalInstance
@pytest.fixture
def profile_a():
  return np.array([
    [1, 2, 4, 6, 8, 5, 3, 7],
    [4, 5, 1, 2, 7, 3, 8, 6],
    [4, 2, 3, 5, 1, 6, 7, 8],
    [8, 7, 6, 5, 4, 3, 2, 1],
    [8, 4, 2, 5, 1, 7, 6, 3],
    [5, 8, 1, 4, 2, 7, 6, 3],
  ])

@pytest.fixture
def profile_b():
  return np.array([
    [3, 2, 4, 6, 8, 5, 1, 7],
    [3, 6, 7, 2, 4, 5, 1, 8],
    [1, 8, 7, 3, 4, 5, 2, 6],
    [5, 6, 1, 4, 8, 7, 3, 2],
    [3, 6, 7, 2, 4, 5, 1, 8],
    [4, 3, 2, 1, 5, 6, 7, 8],
  ])


@pytest.fixture
def agh_course_selection_instance():
  instance = OrdinalInstance()
  instance.parse_url("https://www.preflib.org/static/data/agh/00009-00000001.soc")
  return instance
