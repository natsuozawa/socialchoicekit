import numpy as np
import pytest

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
def profile_single():
  return np.array([[1]])

@pytest.fixture
def profile_empty():
  return np.array([[]])

@pytest.fixture
def profile_1d():
  return np.array([1, 2, 3, 4, 5])

@pytest.fixture
def profile_3d():
  return np.array([
    [[1, 2, 3], [2, 3, 1], [3, 1, 2]],
    [[1, 3, 2], [3, 1, 2], [3, 1, 2]],
    [[2, 1, 3], [2, 1, 3], [2, 3, 1]],
  ])

@pytest.fixture
def profile_repeat():
  return np.array([
    [1, 2, 4, 3, 8, 5, 3, 7],
    [4, 5, 1, 2, 4, 6, 8, 3],
    [3, 7, 1, 2, 4, 6, 8, 5],
  ])

@pytest.fixture
def profile_negative(profile_a):
  return profile_a - 2

@pytest.fixture
def profile_invalid_alternative(profile_a):
  return profile_a + 1

@pytest.fixture
def profile_tie():
  return np.array([
    [1, 2],
    [2, 1],
  ])

@pytest.fixture
def bistochastic_matrix_1():
  return 0.4 * np.array([
    [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]
  ]) + 0.6 * np.array([
    [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]
  ])

@pytest.fixture
def bistochastic_matrix_2():
  return 0.5 * np.array([
    [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]
  ]) + 0.6 * np.array([
    [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]
  ]) - 0.1 * np.array([
    [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]
  ])

@pytest.fixture
def bipartite_graph_undirected():
  return {
    0: [3, 4, 5, 6],
    1: [3, 5],
    2: [4, 6],
    3: [0, 1],
    4: [0, 2],
    5: [0, 1],
    6: [0, 2]
  }

@pytest.fixture
def bipartite_graph_directed():
  return {
    0: [3, 4, 5, 6],
    1: [3, 5],
    2: [4, 6],
    3: [],
    4: [],
    5: [],
    6: []
  }

@pytest.fixture
def flow_network_integral_basic():
  """
  A basic general (non bipartite) flow network with integral capacities.
  """
  return ({
    0: [(1, 1), (2, 1)],
    1: [(2, 1), (3, 1)],
    2: [(3, 1)],
    3: []
  }, 0, 3)
