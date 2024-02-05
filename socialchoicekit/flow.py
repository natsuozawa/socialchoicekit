import sys, copy
from typing import Dict, List, Tuple, Union
import numpy as np

from socialchoicekit.utils import check_bipartite_graph

def ford_fulkerson(G: Dict[int, List[Tuple[int, int]]], s: int, t: int) -> Dict[Tuple[int, int], int]:
  """
  The Ford Fulkerson algorithm for computing the maximum flow in a flow network (with depth first search)

  This implementation only works with integral capacities, and we use this to ensure that the algorithm terminates.

  Parameters
  ----------
  G : Dict[int, List[Tuple[int, int]]]
    A flow network of the form {i: [(j, c), (k, c), ...]} where i is the index of a vertex and [(j, c), (k, c), ...] are the indices of the vertices that i is connected to along with the capacity of the edge.
    The value of the capacities cannot exceed sys.maxsize.
  s : int
    The index of the source vertex.
  t : int
    The index of the sink vertex.

  Returns
  -------
  Dict[Tuple[int, int], int]
    The flow network with the maximum flow. The flow network is of the form {(i, j): f} where f is the flow from vertex i to vertex j.
    This flow network includes paths in the original graph where the flow is zero.
  """
  # Residual graph
  G_f = copy.deepcopy(G)
  flow = dict()

  for i in G.keys():
    for j, _ in G[i]:
      flow[(i, j)] = 0
      flow[(j, i)] = 0
      if all([v != i for (v, _) in G_f[j]]):
        # Add the reverse edge if it does not exist in the residual graph.
        G_f[j] += [(i, 0)]

  while True:
    path_from_sink_to_source = dfs_path(G_f, s, t, {i: 0 for i in G_f.keys()})
    if path_from_sink_to_source is None:
      # Only return the flow in the original graph
      flow_final = dict()
      for i in G.keys():
        for j, _ in G[i]:
          flow_final[(i, j)] = flow[(i, j)]
      return flow_final

    path, c_f_p = path_from_sink_to_source
    for i in range(len(path) - 1):
      u = path[i]
      v = path[i + 1]
      flow[(u, v)] += c_f_p
      flow[(v, u)] -= c_f_p
      # Here, the flow network is only updated for the original network.
      # The residual network in addition gets a flow from v to u with capacity addition of -c_f_p.
      # Update c_f, the capacities of the residual graph. c_f = c - f
      # In effect, this changes by +- c_f_p
      G_f[u] = [(w, c_f - c_f_p) if w == v else (w, c_f) for (w, c_f) in G_f[u]]
      G_f[v] = [(w, c_f + c_f_p) if w == u else (w, c_f) for (w, c_f) in G_f[v]]

def dfs_path(G: Dict[int, List[Tuple[int, int]]], current: int, sink: int, visited: Dict[int, int]) -> Union[Tuple[List[int], int], None]:
  """
  Finds a path from the given vertex to the sink vertex.

  Parameters
  ----------
  G : Dict[int, List[Tuple[int, int]]]
    A flow network of the form {i: [(j, c), (k, c), ...]} where i is the index of a vertex and [(j, c), (k, c), ...] are the indices of the vertices that i is connected to along with the capacity of the edge.
    The value of the capacities cannot exceed sys.maxsize.

  current : int
    The index of the current vertex.

  sink : int
    The index of the sink vertex.

  visited : Dict[int, int]
    A dictionary of the form {i: j} where i is the index of a vertex and j is 0 if i is not visited, 1 if i is visited.

  Returns
  -------
  Tuple[List[int], int]
    A tuple of the form (path, capacity) where path is a list of vertices in the path from the current vertex to the sink vertex and capacity is the capacity of the path.

  None
    If there is no path from the current vertex to the sink vertex.
  """
  if current == sink:
    # Undo visit so other path searches could visit
    visited[current] = 0
    return ([current], sys.maxsize)
  candidates = G[current]
  best_path = None
  best_capacity = 0
  for (v, c) in candidates:
    if visited[v] != 0:
      continue
    if c > 0:
      visited[v] = 1
      subpath = dfs_path(G, v, sink, visited)
      if subpath is not None:
        path, capacity = subpath
        # As a heuristic, return the path with the best capacity.
        if min(capacity, c) > best_capacity:
          best_path = [current] + path
          best_capacity = min(capacity, c)
  if best_path is not None:
    visited[current] = 0
    return (best_path, best_capacity)
  return None

def convert_bipartite_graph_to_flow_network(G: Dict[int, List[int]], X: list, Y: list) -> Dict[int, List[Tuple[int, int]]]:
  """
  Converts a bipartite graph to a flow network by performing the following.
  - Add a source vertex s and a sink vertex t. In this implementation, s = -1, t = -2.
  - Add an edge from s to each vertex in X.
  - Add an edge from each vertex in Y to t.
  - For each edge in the unweighted graph, assign capacity 1 to each edge.

  Must run utils.convert_bipartite_graph on G, X, Y before calling this function.

  Parameters
  ----------
  G : Dict[int, List[int]]
    A bipartite graph of the form {i: [j, k, ...]} where i is the index of a vertex and [j, k, ...] are the indices of the vertices that i is connected to.
    The graph may be undirected (as in for every edges from x to y there is an edge from y to x) or directed. If it is directed, then the edges are assumed to be directed from X to Y.

  X : list
    The list of the left vertices (in the first partition) in the bipartite graph G.

  Y : list
    The list of the right vertices (in the second partition) in the bipartite graph G.

  Returns
  -------
  Dict[int, List[Tuple[int, int]]
    A graph of the form {i: [(j, c), (k, c), ...]} where i is the index of a vertex and [(j, c), (k, c), ...] are the indices of the vertices that i is connected to along with the capacity of the edge.
  """
  # This is a shallow copy
  network = dict()

  for v in X:
    network[v] = [(y, 1) for y in G.get(v, [])]
  network[-1] = [(x, 1) for x in X]
  network[-2] = []

  # Remove all edges that originate from Y, as we assume edges are directed from X to Y.
  # Add edges from Y to sink.
  for v in Y:
    network[v] = [(-2, 1)]
  return network

def maximum_cardinality_matching_bipartite(G: Dict[int, List[int]], X: list, Y: list) -> List[Tuple[int, int]]:
  """
  The maximum cardinality matching on a bipartite graph. This runs the Ford Fulkerson algorithm (with depth first search).

  Parameters
  ----------
  G : Dict[int, List[int]]
    A dictionary of the form {i: [j, k, ...]} where i is the index of a vertex and [j, k, ...] are the indices of the vertices that i is connected to.
    This graph may be directed or undirected. If it is directed, then the edges are assumed to be directed from X to Y.
    This graph must be bipartite.

  X : list
    The list of the left vertices (in the first partition) in the bipartite graph G.

  Y : list
    The list of the right vertices (in the second partition) in the bipartite graph G.

  Returns
  -------
  List[Tuple[int, int]]
    A list of tuples of the form (i, j) where i is a vertex in X and j is a vertex in Y. This represents the maximum cardinality matching.
  """
  check_bipartite_graph(G, X, Y)
  network = convert_bipartite_graph_to_flow_network(G, X, Y)
  flow = ford_fulkerson(network, -1, -2)

  matchings = []
  for x in X:
    matched_y = G[x][np.argmax(np.array([flow[(x, y)] for y in G[x]]))]
    if (flow[x, matched_y] == 1):
      matchings.append((x, matched_y))
  return matchings

