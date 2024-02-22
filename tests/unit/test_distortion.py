import numpy as np
import pytest

from socialchoicekit.distortion import distortion
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
