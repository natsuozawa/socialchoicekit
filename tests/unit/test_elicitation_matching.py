import numpy as np

import pytest

from socialchoicekit.profile_utils import StrictCompleteProfile, IntegerValuationProfile
from socialchoicekit.elicitation_utils import IntegerValuationProfileElicitor
from socialchoicekit.elicitation_matching import DoubleLambdaTSF
from socialchoicekit.deterministic_matching import Irving

class TestElicitationMatching:
  # Copied from TestDeterministicMatching, but might change later.
  @pytest.fixture
  def profiles_1(self):
    # Example given in Irving, et al. (1987)
    ranked_ordinal_profile_1 = np.array([
      [3, 1, 5, 7, 4, 2, 8, 6],
      [6, 1, 3, 4, 8, 7, 5, 2],
      [7, 4, 3, 6, 5, 1, 2, 8],
      [5, 3, 8, 2, 6, 1, 4, 7],
      [4, 1, 2, 8, 7, 3, 6, 5],
      [6, 2, 5, 7, 8, 4, 3, 1],
      [7, 8, 1, 6, 2, 3, 4, 5],
      [2, 6, 7, 1, 8, 3, 4, 5],
    ]) - 1
    ranked_ordinal_profile_2 = np.array([
      [4, 3, 8, 1, 2, 5, 7, 6],
      [3, 7, 5, 8, 6, 4, 1, 2],
      [7, 5, 8, 3, 6, 2, 1, 4],
      [6, 4, 2, 7, 3, 1, 5, 8],
      [8, 7, 1, 5, 6, 4, 3, 1],
      [5, 4, 7, 6, 2, 8, 3, 1],
      [1, 4, 5, 6, 2, 8, 3, 7],
      [2, 5, 4, 3, 7, 8, 1, 6],
    ]) - 1

    # Custom
    # Use the borda-like weights that Irving et al. (1987) used.
    ranked_cardinal_profile_1 = np.array([
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ])

    ranked_cardinal_profile_2 = np.array([
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ])

    # ranked_cardinal_profile_1 = np.array([
    #   [12, 12, 12, 0, 0, 0, 0, 0],
    #   [36, 0, 0, 0, 0, 0, 0, 0],
    #   [15, 15, 6, 0, 0, 0, 0, 0],
    #   [9, 9, 9, 9, 0, 0, 0, 0],
    #   [7, 7, 7, 7, 2, 2, 2, 2],
    #   [7, 7, 7, 7, 2, 2, 2, 2],
    #   [12, 12, 12, 0, 0, 0, 0, 0],
    #   [18, 18, 0, 0, 0, 0, 0, 0],
    # ])
    # ranked_cardinal_profile_2 = np.array([
    #   [6, 6, 4, 4, 4, 4, 4, 4],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [5, 5, 5, 5, 5, 5, 5, 1],
    #   [5, 5, 5, 5, 4, 4, 4, 4],
    #   [5, 5, 5, 5, 5, 5, 5, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    #   [8, 7, 6, 5, 4, 3, 2, 1],
    # ])

    # Fix into form accepted by Profile, ValuationProfile
    ordinal_profile_1 = np.argsort(ranked_ordinal_profile_1, axis=1)
    ordinal_profile_2 = np.argsort(ranked_ordinal_profile_2, axis=1)
    cardinal_profile_1 = np.zeros(ranked_cardinal_profile_1.shape)
    cardinal_profile_2 = np.zeros(ranked_cardinal_profile_2.shape)
    cardinal_profile_1 = np.take_along_axis(ranked_cardinal_profile_1, ordinal_profile_1, axis=1)
    cardinal_profile_2 = np.take_along_axis(ranked_cardinal_profile_2, ordinal_profile_2, axis=1)

    return StrictCompleteProfile.of(ordinal_profile_1 + 1), StrictCompleteProfile.of(ordinal_profile_2 + 1), IntegerValuationProfile.of(cardinal_profile_1), IntegerValuationProfile.of(cardinal_profile_2)

  @pytest.fixture
  def profiles_2(self):
        # Example given in Irving, et al. (1987) with modified utilities.
    ranked_ordinal_profile_1 = np.array([
      [3, 1, 5, 7, 4, 2, 8, 6],
      [6, 1, 3, 4, 8, 7, 5, 2],
      [7, 4, 3, 6, 5, 1, 2, 8],
      [5, 3, 8, 2, 6, 1, 4, 7],
      [4, 1, 2, 8, 7, 3, 6, 5],
      [6, 2, 5, 7, 8, 4, 3, 1],
      [7, 8, 1, 6, 2, 3, 4, 5],
      [2, 6, 7, 1, 8, 3, 4, 5],
    ]) - 1
    ranked_ordinal_profile_2 = np.array([
      [4, 3, 8, 1, 2, 5, 7, 6],
      [3, 7, 5, 8, 6, 4, 1, 2],
      [7, 5, 8, 3, 6, 2, 1, 4],
      [6, 4, 2, 7, 3, 1, 5, 8],
      [8, 7, 1, 5, 6, 4, 3, 1],
      [5, 4, 7, 6, 2, 8, 3, 1],
      [1, 4, 5, 6, 2, 8, 3, 7],
      [2, 5, 4, 3, 7, 8, 1, 6],
    ]) - 1

    # Custom
    ranked_cardinal_profile_1 = np.array([
      [12, 12, 12, 0, 0, 0, 0, 0],
      [36, 0, 0, 0, 0, 0, 0, 0],
      [15, 15, 6, 0, 0, 0, 0, 0],
      [9, 9, 9, 9, 0, 0, 0, 0],
      [7, 7, 7, 7, 2, 2, 2, 2],
      [7, 7, 7, 7, 2, 2, 2, 2],
      [12, 12, 12, 0, 0, 0, 0, 0],
      [18, 18, 0, 0, 0, 0, 0, 0],
    ])
    ranked_cardinal_profile_2 = np.array([
      [6, 6, 4, 4, 4, 4, 4, 4],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [5, 5, 5, 5, 5, 5, 5, 1],
      [5, 5, 5, 5, 4, 4, 4, 4],
      [5, 5, 5, 5, 5, 5, 5, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ])

    # Fix into form accepted by Profile, ValuationProfile
    ordinal_profile_1 = np.argsort(ranked_ordinal_profile_1, axis=1)
    ordinal_profile_2 = np.argsort(ranked_ordinal_profile_2, axis=1)
    cardinal_profile_1 = np.zeros(ranked_cardinal_profile_1.shape)
    cardinal_profile_2 = np.zeros(ranked_cardinal_profile_2.shape)
    cardinal_profile_1 = np.take_along_axis(ranked_cardinal_profile_1, ordinal_profile_1, axis=1)
    cardinal_profile_2 = np.take_along_axis(ranked_cardinal_profile_2, ordinal_profile_2, axis=1)

    return StrictCompleteProfile.of(ordinal_profile_1 + 1), StrictCompleteProfile.of(ordinal_profile_2 + 1), IntegerValuationProfile.of(cardinal_profile_1), IntegerValuationProfile.of(cardinal_profile_2)


  def test_double_lambda_tsf_1(self, profiles_1):
    profile_1, profile_2, cardinal_profile_1, cardinal_profile_2 = profiles_1
    ivpe_1 = IntegerValuationProfileElicitor(cardinal_profile_1)
    ivpe_2 = IntegerValuationProfileElicitor(cardinal_profile_2)
    for lambda_ in range(2, 8):
      dlt = DoubleLambdaTSF(lambda_1=lambda_, lambda_2=lambda_, zero_indexed=True)
      stable_matching = dlt.scf(profile_1, profile_2, ivpe_1, ivpe_2)
      # Check cardinal value with respect to the original valuation profiles, not the simulated valuation profiles.
      expected_value = Irving.stable_matching_value([(0, 0), (1, 3), (2, 2), (3, 4), (4, 1), (5, 5), (6, 7), (7, 6)], cardinal_profile_1, cardinal_profile_2)
      actual_value = Irving.stable_matching_value(stable_matching, cardinal_profile_1, cardinal_profile_2)
      # We can only check this upper bound because even when we let lambda = n, the generated bucket will not correspond to each integer between [1, n]
      assert expected_value >= actual_value

  def test_double_lambda_tsf_2(self, profiles_2):
    profile_1, profile_2, cardinal_profile_1, cardinal_profile_2 = profiles_2
    ivpe_1 = IntegerValuationProfileElicitor(cardinal_profile_1)
    ivpe_2 = IntegerValuationProfileElicitor(cardinal_profile_2)
    for lambda_ in range(2, 8):
      dlt = DoubleLambdaTSF(lambda_1=lambda_, lambda_2=lambda_, zero_indexed=True)
      elicitation_stable_matching = dlt.scf(profile_1, profile_2, ivpe_1, ivpe_2)
      irving = Irving(zero_indexed=True)
      irving_stable_matching = irving.scf(cardinal_profile_1, cardinal_profile_2, profile_1, profile_2)
      # Check cardinal value with respect to the original valuation profiles, not the simulated valuation profiles.
      expected_value = Irving.stable_matching_value(irving_stable_matching, cardinal_profile_1, cardinal_profile_2)
      actual_value = Irving.stable_matching_value(elicitation_stable_matching, cardinal_profile_1, cardinal_profile_2)
      assert expected_value >= actual_value
