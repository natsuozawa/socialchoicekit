import pytest
import numpy as np

from socialchoicekit.bistochastic import *

class TestBistochastic:
  @pytest.fixture
  def bistochastic_matrix_1(self):
    return 0.4 * np.array([
      [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]
    ]) + 0.6 * np.array([
      [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]
    ])

  @pytest.fixture
  def bistochastic_matrix_2(self):
    return 0.5 * np.array([
      [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]
    ]) + 0.6 * np.array([
      [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]
    ]) - 0.1 * np.array([
      [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]
    ])

  def test_positivity_graph_without_negative_values(self, bistochastic_matrix_1):
    G = positivity_graph(bistochastic_matrix_1)
    assert isinstance(G, dict)
    assert G.keys() == set(range(0, 8))
    assert G[0] == [5, 6]
    assert G[1] == [4, 7]
    assert G[4] == [1, 2]
    assert G[5] == [0, 3]

  def test_positivity_graph_with_negative_values(self, bistochastic_matrix_2):
    G = positivity_graph(bistochastic_matrix_2)
    assert isinstance(G, dict)
    assert G.keys() == set(range(0, 8))
    assert G[0] == [5, 6]
    assert G[1] == [4, 7]
    assert G[4] == [1, 2]
    assert G[5] == [0, 3]

  def test_birkhoff_von_neumann_1(self, bistochastic_matrix_1):
    result = birkhoff_von_neumann(bistochastic_matrix_1)
    assert sum([z for (z, _) in result]) == pytest.approx(1)
    assert all([P.shape[0] == P.shape[1] and P.shape[0] == bistochastic_matrix_1.shape[0] for (_, P) in result])
    assert all([np.all(np.sum(P, axis=0) == np.ones(P.shape[0])) for (_, P) in result])
    assert all([np.all(np.sum(P, axis=1) == np.ones(P.shape[0])) for (_, P) in result])
    assert all([np.all(np.unique(P) == np.array([0, 1])) for (_, P) in result])
