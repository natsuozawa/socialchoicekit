import numpy as np
from ortools.linear_solver.pywraplp import Solver, Variable

from typing import Union, Tuple, List

from socialchoicekit.deterministic_scoring import SocialWelfare
from socialchoicekit.utils import check_valuation_profile, check_profile
from socialchoicekit.profile_utils import StrictCompleteProfile, ValuationProfile, incomplete_valuation_profile_to_complete_valuation_profile

def distortion(
  choice: Union[np.ndarray, int],
  valuation_profile: ValuationProfile,
) -> float:
  """
  This is a utility function to calculate distortion as introduced by Procaccia and Rosenschein (2006)

  Distortion is the worst case ratio between the optimal utility obtainable from cardinal information and the optimal utility obtainable from an algorithm using limited preference information.

  distortion(f(P), v) = (max_{j in A} SW(j|v)) /  SW(f(P) | v)

  Parameters
  ----------
  choice : Union[np.ndarray, int]
    The choice (winner) made by the social choice function (scf) voting the voting rule that is being evaluated, based on limited preference information. Assumed to be 1-indexed.
    The type allows for the output of the scf method of a voting rule to be passed in directly. If multiple choices are given, this function chooses the choice that maximizes the distortion.

  valuation_profile : ValuationProfile
    A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the agent's value for item j. If the agent finds an item unacceptable or the agent's preference is unknown, the element would be np.nan.
    Any np.nan values will be treated as 0.
  """
  check_valuation_profile(valuation_profile, is_complete=False)
  complete_vp = incomplete_valuation_profile_to_complete_valuation_profile(valuation_profile)
  sw = SocialWelfare(tie_breaker="random")
  score = sw.score(complete_vp)
  if isinstance(choice, np.ndarray):
    return np.max(score) / np.min(score[choice - 1])
  return np.max(score) / score[choice - 1]

def optimal_distortion(profile: StrictCompleteProfile) -> float:
  """
  This algorithm from Ebadian et al. (2024) calculates the best possible distortion for a given ordinal profile.
  Given an ordinal profile, there are many cardinal profiles that are consistent with the ordinal profile (ie follow the same ordered structure).
  For any voting rule, we can calculate its distortion on an ordinal profile as follows.
  Given a voting rule, we can calculate its distortion on each cardinal profile consistent with the ordinal profile. Let the maximum of these distortions be the distortion of the voting rule on the ordinal profile.
  Then, the best possible distortion for a given ordinal profile is given by the voting rule which has the lowest distortion on the ordinal profile.
  The resulting distortion on the ordinal profile then becomes a lower bound on the distortion on this ordinal profile for any algorithm.
  Note that given an ordinal profile (or a cardinal profile) and a fixed (randomized) voting rule,
  the distortion can be higher than this optimal distortion.

  Parameters
  ----------
  profile : StrictCompleteProfile
    A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the agent's ranking of item j. The rankings are 1-indexed.

  Returns
  -------
  float
    The optimal distortion.
  """
  check_profile(profile)
  solver, _ = optimal_distortion_lp(profile)
  return solver.Objective().Value()

def optimal_distortion_lp(
  profile: StrictCompleteProfile,
) -> Tuple[Solver, Tuple[List[Variable], List[List[Variable]], List[List[Variable]], List[List[Variable]]]]:
  """
  This is a subroutine used to calculate the optimal distortion for a given ordinal profile along with the probability distribution representing a randomized voting rule that would generate this distortion.
  This is an implementation of the linear program presented by Ebadian et al. (2024).
  Given an ordinal profile, there are many cardinal profiles that are consistent with the ordinal profile (ie follow the same ordered structure).
  For any voting rule, we can calculate its distortion on an ordinal profile as follows.
  Given a voting rule, we can calculate its distortion on each cardinal profile consistent with the ordinal profile. Let the maximum of these distortions be the distortion of the voting rule on the ordinal profile.
  Then, the best possible distortion for a given ordinal profile is given by the voting rule which has the lowest distortion on the ordinal profile.
  The resulting distortion on the ordinal profile then becomes a lower bound on the distortion on this ordinal profile for any algorithm.
  Note that given an ordinal profile (or a cardinal profile) and a fixed (randomized) voting rule,
  the distortion can be higher than this optimal distortion.

  Parameters
  ----------
  profile : StrictCompleteProfile
    A (N, M) array, where N is the number of agents and M is the number of items. The element at (i, j) indicates the agent's ranking of item j. The rankings are 1-indexed.

  Returns
  -------
  Tuple[Solver, Tuple[List[Variable], List[List[Variable]], List[List[Variable]], List[List[Variable]]]]:
    Since this is long, we will explain each component in detail below.

  Solver
    The MPSolver instance that has finished solving the linear program. See Google OR-Tools documentation for more information on how to process this object.
    https://developers.google.com/optimization/lp/lp_example
    https://developers.google.com/optimization/reference/python/linear_solver/pywraplp
    https://or-tools.github.io/docs/pdoc/ortools/linear_solver/pywraplp.html

  List[Variable]
    p_hat. A list of M elements.

  List[List[Variable]]
    delta. An (N, M) list.

  List[List[Variable]]
    alpha. An (N, M) list.

  List[List[Variable]]
    beta. An (N, M) list.
  """
  n = profile.shape[0]
  m = profile.shape[1]

  ranked_profile = np.argsort(profile, axis=1).view(np.ndarray)
  # 0-indexed prof to be used for calculation
  prof = profile.view(np.ndarray) - 1

  solver: Solver = Solver.CreateSolver('GLOP')
  if not solver:
    raise Exception('Solver not found.')

  # Variable definitions
  p_hat = []
  for a in range(m):
    p_hat.append(solver.NumVar(0, solver.infinity(), f'p-hat_{a}'))
  delta = [[] for _ in range(n)]
  alpha = [[] for _ in range(n)]
  beta = [[] for _ in range(n)]
  for i in range(n):
    for r in range(m):
      delta[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'delta_{i}_{r}'))
      alpha[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'alpha_{i}_{r}'))
      beta[i].append(solver.NumVar(-solver.infinity(), solver.infinity(), f'beta_{i}_{r}'))

  # Constraints
  # Main
  for i in range(n):
    # r in [2, m - 1] in 1-indexed is equal to [1, m - 1) is 0-indexed
    for r in range(1, m - 1):
      j = ranked_profile[i, r]
      # There will be no out of bounds error here because r >= 1
      solver.Add(delta[i][prof[i, j]] >= alpha[i][r - 1])
    for r in range(m):
      j = ranked_profile[i, r]
      solver.Add(delta[i][prof[i, j]] >= beta[i][r])
  for a in range(m):
    solver.Add(solver.Sum([delta[i][a] for i in range(n)]) <= 0)
  # Top partial maximums
  for i in range(n):
    for r in range(m):
      if r > 0:
        solver.Add(alpha[i][r] >= alpha[i][r - 1])
        # Rotate the 1/r to the other side to avoid array scalar multiplication
        solver.Add(r * alpha[i][r] >= solver.Sum([-p_hat[ranked_profile[i, l]] for l in range(r)]))
  # Bottom partial sums
  for i in range(n):
    for r in range(m):
      if r < m - 1:
        solver.Add(beta[i][r] >= beta[i][r + 1])
        # Rotate the 1/r to the other side to avoid array scalar multiplication
        solver.Add(r * beta[i][r] >= solver.Sum([1 - p_hat[ranked_profile[i, l]] for l in range(r)]))

  # Variable ranges are hard coded in the variable definitions

  # Objective function
  solver.Minimize(solver.Sum(p_hat))

  # Solve
  status = solver.Solve()

  if status != solver.OPTIMAL:
    raise Exception('The problem does not have an optimal solution.')

  # Explicitly define types at the end because operator overloads don't work with
  # explicitly defined variables for some reason.
  _p_hat: List[Variable] = p_hat
  _delta: List[List[Variable]] = delta
  _alpha: List[List[Variable]] = alpha
  _beta: List[List[Variable]] = beta

  return solver, (_p_hat, _delta, _alpha, _beta)
