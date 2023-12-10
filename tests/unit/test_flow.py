import pytest
import numpy as np

from socialchoicekit.flow import convert_bipartite_graph_to_flow_network, ford_fulkerson

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

  def test_ford_fulkerson_integral_1(self, flow_network_integral_1):
    network, s, t = flow_network_integral_1
    flow = ford_fulkerson(network, s, t)
    assert isinstance(flow, dict)
    for (u, v) in flow.keys():
      # Here, due to the nature of the basic flow network as only having binary capacities, the capacity can be assumed to be always 1.
      assert (v, 1) in network[u]
    assert flow[(0, 1)] == 1
    assert flow[(0, 2)] == 1
    assert flow[(1, 2)] == 0
    assert flow[(1, 3)] == 1
    assert flow[(2, 3)] == 1

  def test_ford_fulkerson_integral_2(self, flow_network_integral_2):
    network, s, t = flow_network_integral_2
    flow = ford_fulkerson(network, s, t)
    assert flow[(0, 1)] + flow[(0, 2)] == 13

