import pytest
import numpy as np

from socialchoicekit.bistochastic import positivity_graph

class TestBistochastic():
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
