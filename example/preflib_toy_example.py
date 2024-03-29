"""
From PrefLib:

This dataset contains the results of surveying students at AGU University of Science and Technology about their course preferences. Each student provided a rank ordering over all the courses with no missing elements. There are 9 courses to choose from in 2003 and 7 in 2004.

The data on this page has been donated by Piotr Faliszewski.

Selected studies: P. Skowron, P. Faliszewski and A. Slinko. Achieving Fully Proportional Representation is Easy in Practice. Proceedings of AAMAS, 2013. | V. Hashemi and U. Endriss. Measuring Diversity of Preferences in a Group. Proceedings of ECAI, 2014.
"""

import numpy as np

from preflibtools.instances import OrdinalInstance

from socialchoicekit.preflib_utils import preflib_soc_to_profile
from socialchoicekit.data_generation import UniformValuationProfileGenerator
from socialchoicekit.deterministic_scoring import Plurality, SocialWelfare
from socialchoicekit.elicitation_voting import KARV
from socialchoicekit.elicitation_utils import ValuationProfileElicitor, SynchronousStdInElicitor
from socialchoicekit.distortion import distortion

url = 'https://www.preflib.org/static/data/agh/00009-00000001.soc'

# 1.1) Import data
print("----- 1.1) Import data -----")
instance = OrdinalInstance()
instance.parse_url(url)
profile = preflib_soc_to_profile(instance)
print(profile)

# 1.2) Generate (hypothetical) cardinal profile
print("----- 1.2) Generate hypothetical cardinal profile -----")
valuation_profile = UniformValuationProfileGenerator(high=1, low=0, seed=1).generate(profile)
print(valuation_profile)

# 1.3) Compute optimal utility using cardinal information
print("----- 1.3) Compute optimal utility using cardinal information -----")
social_welfare = SocialWelfare().score(valuation_profile)
print(social_welfare)
optimal_alternative = int(np.argmax(social_welfare))
print("Optimal alternative: ", optimal_alternative + 1)
optimal_welfare = np.amax(social_welfare)
print("Optimal welfare: ", optimal_welfare)

# 2) Test baseline: Plurality (pick favorite)
print("----- 2) Test baseline: Plurality -----")
plurality = Plurality()
plurality_winner = plurality.scf(profile)
print(plurality.score(profile))
print("Plurality winner: ", plurality_winner)
print("Distortion: ", distortion(plurality_winner, valuation_profile))

# 3) Elicitation (query)-based voting
print("----- 3) Elicitation-based voting -----")
karv = KARV(k=3)
valuation_profile_elicitor = ValuationProfileElicitor(valuation_profile=valuation_profile, memoize=True)
stdin_elicitor = SynchronousStdInElicitor(memoize=True)
karv_winner = karv.scf(profile, valuation_profile_elicitor)
# karv_winner = karv.scf(profile, stdin_elicitor)
print("KARV winner: ", karv_winner)
print("Distortion: ", distortion(karv_winner, valuation_profile))
