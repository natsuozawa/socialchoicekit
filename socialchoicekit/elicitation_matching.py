import numpy as np

from typing import List, Tuple

from socialchoicekit.deterministic_matching import Irving
from socialchoicekit.elicitation_utils import IntegerElicitor, IntegerSynchronousStdInElicitor
from socialchoicekit.profile_utils import IntegerValuationProfile, StrictCompleteProfile

class DoubleLambdaTSF:
  """
  Double Lambda-Threshold Step Function is the provisional name for applying the binary search-based elicitation technique to get a good distortion for stable matching.
  The algorithm partitions alternatives into lambda + 1 sets for evey agent to create a simulated value function using binary search.
  As the cardinal algorithm, this uses Irving's algorithm for stable matching.

  Parameters
  ----------
  lambda_1 : int
    The number of positions to query for profile_1.

  lambda_2 : int
    The number of positions to query for profile_2.

  zero_indexed : bool
    If True, the output of the social choice function will be zero-indexed. If False, the output will be one-indexed. One-indexed by default.
  """
  def __init__(
    self,
    lambda_1: int = 1,
    lambda_2: int = 1,
    zero_indexed: bool = False,
  ):
    if lambda_1 < 1 or lambda_2 < 1:
      raise ValueError("Invalid lambda")
    self.lambda_1 = lambda_1
    self.lambda_2 = lambda_2
    self.irving = Irving(zero_indexed=zero_indexed)

  def scf(
    self,
    profile_1: StrictCompleteProfile,
    profile_2: StrictCompleteProfile,
    elicitor_1: IntegerElicitor = IntegerSynchronousStdInElicitor(),
    elicitor_2: IntegerElicitor = IntegerSynchronousStdInElicitor(),
  ) -> List[Tuple[int, int]]:
    """
    The social choice function for this voting rule. Returns agent to agent pairs where each agent is only matched once.

    Parameters
    ----------
    profile_1: StrictCompleteProfile
      A (N, N) array, where N is the number of agents in the first group and also the number of agents in the second group. The element at (i, j) indicates the agent i's preference for agent j, where 1 is the most preferred agent. Here, agent i belongs to the first group and agent j belongs to the second group.

    profile_2: StrictCompleteProfile
      A (N, N) array, where N is the number of agents in the second group and also the number of agents in the first group. The element at (i, j) indicates the agent i's preference for agent j, where 1 is the most preferred agent. Here, agent i belongs to the second group and agent j belongs to the first group.

    elicitor_1: IntegerElicitor
      The elicitor that will be used to query the agents in the first group. By default, IntegerSynchronousStdInElicitor is used.
      Memoization should be enabled for this elicitor.

    elicitor_2: IntegerElicitor
      The elicitor that will be used to query the agents in the first group. By default, IntegerSynchronousStdInElicitor is used.
      Memoization should be enabled for this elicitor.

    Returns
    -------
    List[Tuple[int, int]]
      A list containing assignments (agent_1, agent_2) where agent_1 from the first group is matched to agent_2 from the second group.
      This list will contain N tuples, where N is the number of agents in each group.
    """
    v_tildes = self.get_simulated_cardinal_profiles(profile_1, profile_2, elicitor_1, elicitor_2)
    return self.irving.scf(v_tildes[0], v_tildes[1], profile_1, profile_2)

  def get_simulated_cardinal_profiles(
    self,
    profile_1: StrictCompleteProfile,
    profile_2: StrictCompleteProfile,
    elicitor_1: IntegerElicitor = IntegerSynchronousStdInElicitor(),
    elicitor_2: IntegerElicitor = IntegerSynchronousStdInElicitor(),
  ) -> Tuple[IntegerValuationProfile, IntegerValuationProfile]:
    """
    Obtain the two simulated cardinal profiles.

    Parameters
    ----------
    profile_1: StrictCompleteProfile
      A (N, N) array, where N is the number of agents in the first group and also the number of agents in the second group. The element at (i, j) indicates the agent i's preference for agent j, where 1 is the most preferred agent. Here, agent i belongs to the first group and agent j belongs to the second group.

    profile_2: StrictCompleteProfile
      A (N, N) array, where N is the number of agents in the second group and also the number of agents in the first group. The element at (i, j) indicates the agent i's preference for agent j, where 1 is the most preferred agent. Here, agent i belongs to the second group and agent j belongs to the first group.

    elicitor_1: IntegerElicitor
      The elicitor that will be used to query the agents in the first group. By default, IntegerSynchronousStdInElicitor is used.
      Memoization should be enabled for this elicitor.

    elicitor_2: IntegerElicitor
      The elicitor that will be used to query the agents in the first group. By default, IntegerSynchronousStdInElicitor is used.
      Memoization should be enabled for this elicitor.

    Returns
    -------
    Tuple[IntegerValuationProfile, IntegerValuationProfile]
      The first IntegerValuationProfile is the simulated profile for the first group.
      The second IntegerValuationProfile is the simulated profile for the second group.
    """
    n = profile_1.shape[0]
    assert profile_1.shape == (n, n)
    assert profile_2.shape == (n, n)

    profiles = [profile_1, profile_2]
    lambdas = [self.lambda_1, self.lambda_2]
    elicitors = [elicitor_1, elicitor_2]
    v_tildes = []

    if lambdas[0] > n or lambdas[1] > n:
      raise ValueError("Invalid lambda")

    for k, profile in enumerate(profiles):
      # Element at (i, j) is agent i's j+1th most preferred alternative (0-indexed alternative number)
      ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)
      # Element at i is agent i's favorite alternative
      v_favorite = elicitors[k].elicit_multiple(np.arange(n), ranked_profile[:, 0])

      # We have this as an inner function because it currently needs to access the ranked_profile array.
      # We've modified this binary search from the paper for our implementation.
      def binary_search(i: int, left: int, right: int, alpha: int, v: float):
        if right - left <= 1:
          return left
        # This will never be more than m - 1, even if we start with beta = m.
        mid = (right + left) // 2
        u = elicitors[k].elicit(i, ranked_profile[i, mid])
        if u >= v / alpha:
          return binary_search(i, mid, right, alpha, v)
        else:
          return binary_search(i, left, mid, alpha, v)

      # Element at (i, j) is the simulated welfare of alternative j for agent i
      v_tilde = np.zeros((n, n))
      v_tilde[np.arange(n), ranked_profile[:, 0]] = v_favorite
      # Element at i is the least preferred alternative (0-indexed alternative number) in agent i's lambda-acceptable set
      # Add a very small threshold to distinguish between unacceptable alterantives and alternatives that did not fit in any acceptable set.
      Q_prev = np.zeros(n)
      for l in range(1, lambdas[k] + 1):
        alpha_l = n ** (l / (lambdas[k] + 1))
        p_star = np.array([binary_search(i, 0, n, alpha_l, v_favorite[i]) for i in range(n)])
        memoized_v = np.array([elicitors[k].elicit(i, ranked_profile[i, p_star[i]]) for i in range(n)])
        j_indices = np.concatenate([ranked_profile[i, np.arange(Q_prev[i] + 1, p_star[i] + 1, dtype=int)] for i in range(n)])
        i_indices = np.concatenate([np.ones(int(p_star[i] - Q_prev[i]), dtype=int) * i for i in range(n)])
        # Use the lowest returned cardinal value greater than alpha_l
        # to create an integer valuation profile.
        v_tilde[(i_indices, j_indices)] = memoized_v[i_indices]
        Q_prev = p_star
      v_tildes.append(v_tilde.astype(int))

    return (IntegerValuationProfile.of(v_tildes[0]), IntegerValuationProfile.of(v_tildes[1]))
