import numpy as np
import pytest

from socialchoicekit.distortion import distortion, optimal_distortion, optimal_distortion_lp
from socialchoicekit.deterministic_scoring import Plurality
from socialchoicekit.profile_utils import CompleteValuationProfile, StrictCompleteProfile

class TestDistortion:
  @pytest.fixture
  def cardinal_profile_1(self):
    return CompleteValuationProfile.of(np.array([
      [0, 0, 0, 0.5, 0.5],
      [0, 0, 0, 0.5, 0.5],
      [0, 0, 0.5, 0.5, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
    ]))

  @pytest.fixture
  def ordinal_profile_1(self):
    return StrictCompleteProfile.of(np.array([
      [3, 4, 5, 1, 2],
      [3, 4, 5, 1, 2],
      [3, 4, 2, 1, 5],
      [1, 2, 3, 4, 5],
      [1, 2, 3, 4, 5],
    ]))

  def test_distortion_1(self, cardinal_profile_1, ordinal_profile_1):
    plurality = Plurality()
    dist = distortion(plurality.scf(ordinal_profile_1), cardinal_profile_1)
    assert np.allclose(dist, 2 / 1.5)

  @pytest.fixture
  def cardinal_profile_2(self):
    return CompleteValuationProfile.of(np.array([
      [1, 0, 0, 0, 0],
      [0.2, 0, 0, 0.3, 0.5],
      [0.2, 0, 0, 0.3, 0.5],
      [1, 0, 0, 0, 0],
      [0.1, 0.4, 0.3, 0.2, 0],
      [0.1, 0.4, 0.3, 0.2, 0],
    ]))

  @pytest.fixture
  def ordinal_profile_2(self):
    return StrictCompleteProfile.of(np.array([
      [1, 4, 5, 3, 2],
      [3, 5, 4, 2, 1],
      [3, 5, 4, 2, 1],
      [1, 4, 5, 3, 2],
      [4, 1, 2, 3, 5],
      [4, 1, 2, 3, 5],
    ]))

  def test_distortion_2(self, cardinal_profile_2, ordinal_profile_2):
    # Test that distortion works with np.ndarray as input and chooses the alternative with the worst distortion
    plurality = Plurality(tie_breaker="accept")
    winners = plurality.scf(ordinal_profile_2)
    assert isinstance(winners, np.ndarray)
    assert len(winners) == 3
    dist = distortion(winners, cardinal_profile_2)
    assert np.allclose(dist, 2.6 / 0.8)

  @pytest.fixture
  def ordinal_profile_3(self):
    return StrictCompleteProfile.of(np.array([
      [1, 2],
      [2, 1],
      [1, 2],
    ]))

  def test_optimal_distortion_3(self, ordinal_profile_3):
    n = ordinal_profile_3.shape[0]
    m = ordinal_profile_3.shape[1]

    solver, (p_hat, delta, alpha, beta) = optimal_distortion_lp(ordinal_profile_3)
    num_constraints = solver.NumConstraints()
    num_delta_constraints = n * (m - 2) + n * m + 1
    num_top_partial_maximums = n * (m - 2) + n * (m - 1)
    num_bottom_partial_maximums = n * (m - 1) + n * m
    # +1 for the sum constraint on the deltas
    # Don't count the variable range constraints as they are included in the variable definitions
    assert num_constraints == num_delta_constraints + 1 + num_top_partial_maximums + num_bottom_partial_maximums

    num_variables = solver.NumVariables()
    # delta, alpha, beta are n * m each and p_hat is m
    assert num_variables == 3 * n * m + m

    p_hat_values = [ph.solution_value() for ph in p_hat]
    delta_values = [[delta[i][r].solution_value() for r in range(m)] for i in range(n)]
    alpha_values = [[alpha[i][r].solution_value() for r in range(m)] for i in range(n)]
    beta_values = [[beta[i][r].solution_value() for r in range(m)] for i in range(n)]

    # In the ordinal_profile_3 case, the dichotomous profile can either be:
    # [[1, 0], [0, 1], [1, 0]] or all 0.5
