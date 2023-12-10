import pytest
import numpy as np

from socialchoicekit.flow import convert_bipartite_graph_to_flow_network

class TestFlow():
  def test_convert_bipartite_graph_to_flow_network_undirected(self, bipartite_graph_undirected):
    network = convert_bipartite_graph_to_flow_network(bipartite_graph_undirected, list(range(0, 3)), list(range(3, 7)))
    assert isinstance(network, dict)
    assert network.keys() == set(range(-2, 7))
    assert network[-2] == []
    assert network[-1] == [(i, 1) for i in range(0, 3)]
    assert network[0] == [(3, 1), (4, 1), (5, 1), (6, 1)]
    assert network[1] == [(3, 1), (5, 1)]
    assert network[3] == [(-2, 1)]

  def test_convert_bipartite_graph_to_flow_network_directed(self, bipartite_graph_directed):
    network = convert_bipartite_graph_to_flow_network(bipartite_graph_directed, list(range(0, 3)), list(range(3, 7)))
    assert isinstance(network, dict)
    assert network.keys() == set(range(-2, 7))
    assert network[-2] == []
    assert network[-1] == [(i, 1) for i in range(0, 3)]
    assert network[0] == [(3, 1), (4, 1), (5, 1), (6, 1)]
    assert network[1] == [(3, 1), (5, 1)]
    assert network[3] == [(-2, 1)]
