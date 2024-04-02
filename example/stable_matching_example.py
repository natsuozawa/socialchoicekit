import numpy as np

from socialchoicekit.profile_utils import StrictCompleteProfile, IntegerValuationProfile, is_consistent_valuation_profile
from socialchoicekit.elicitation_utils import IntegerValuationProfileElicitor
from socialchoicekit.deterministic_matching import GaleShapley, Irving
from socialchoicekit.elicitation_matching import DoubleLambdaTSF

sigma_1 = StrictCompleteProfile.of(np.array([
  [4, 3, 2, 1, 5, 6],
  [5, 2, 1, 6, 3, 4],
  [6, 1, 4, 5, 2, 3],
  [4, 1, 3, 5, 6, 2],
  [2, 5, 6, 3, 1, 4],
  [1, 6, 5, 3, 4, 2],
]))

sigma_2 = StrictCompleteProfile.of(np.array([
  [4, 2, 1, 3, 5, 6],
  [3, 4, 5, 6, 2, 1],
  [5, 6, 3, 4, 1, 2],
  [6, 5, 4, 3, 2, 1],
  [2, 4, 5, 1, 6, 3],
  [1, 2, 4, 5, 3, 6],
]))

v_1 = IntegerValuationProfile.of(np.array([
  [1, 1, 3, 5, 0, 0],
  [1, 2, 2, 1, 2, 2],
  [1, 3, 1, 1, 2, 2],
  [1, 3, 2, 1, 1, 2],
  [3, 1, 1, 1, 3, 1],
  [4, 0, 0, 2, 1, 3],
]))

v_2 = IntegerValuationProfile.of(np.array([
  [0, 5, 5, 0, 0, 0],
  [3, 0, 0, 0, 3, 4],
  [0, 0, 0, 0, 10, 0],
  [0, 0, 0, 0, 3, 7],
  [2, 0, 0, 7, 0, 1],
  [9, 1, 0, 0, 0, 0],
]))

assert is_consistent_valuation_profile(v_1, sigma_1)
assert is_consistent_valuation_profile(v_2, sigma_2)

n = 6
gs = GaleShapley(resident_oriented=True, zero_indexed=True)

# X-optimal
M_x = gs.scf(sigma_1, sigma_2, np.ones(n))
# Y-optimal
M_y = gs.scf(sigma_2, sigma_1, np.ones(n))

x_shortlists, y_shortlists = Irving().find_initial_preference_lists(M_x, sigma_1 - 1, sigma_2 - 1)
irving = Irving(zero_indexed=True)
exposed_rotations = irving.find_rotations(x_shortlists, y_shortlists)
all_rotations, _ = irving.find_all_rotations_and_eliminations(x_shortlists, y_shortlists)
M_i = irving.scf(v_1, v_2, sigma_1, sigma_2)

sw_i = Irving.stable_matching_value(M_i, v_1, v_2)
sw_x = Irving.stable_matching_value(M_x, v_1, v_2)
sw_y = Irving.stable_matching_value(M_y, v_1, v_2)

print(sw_i, sw_x, sw_y)

ivpe_1 = IntegerValuationProfileElicitor(v_1)
ivpe_2 = IntegerValuationProfileElicitor(v_2)
dltsf = DoubleLambdaTSF(1, 1, True)
v_tildes = dltsf.get_simulated_cardinal_profiles(sigma_1, sigma_2, ivpe_1, ivpe_2)
M_d = dltsf.scf(sigma_1, sigma_2, ivpe_1, ivpe_2)
sw_d = Irving.stable_matching_value(M_d, v_1, v_2)
weights = [Irving().rotation_weight(rot, v_tildes[0], v_tildes[1]) for rot in all_rotations]

print(sw_d)
