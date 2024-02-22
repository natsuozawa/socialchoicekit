from socialchoicekit.preflib_utils import *
from socialchoicekit.data_generation import UniformValuationProfileGenerator
from socialchoicekit.elicitation_voting import LambdaPRV
from socialchoicekit.elicitation_utils import ValuationProfileElicitor
from socialchoicekit.deterministic_scoring import SocialWelfare

from preflibtools.instances import OrdinalInstance


url = 'https://www.preflib.org/static/data/agh/00009-00000001.soc'
instance = OrdinalInstance()
instance.parse_url(url)
profile: StrictCompleteProfile = preflib_soc_to_profile(instance)

valuation_profile: ValuationProfile = UniformValuationProfileGenerator(0, 1).generate(profile)

voting_rule = LambdaPRV(3)
elicitor = ValuationProfileElicitor(valuation_profile)
winner = voting_rule.scf(profile, elicitor)
social_welfare = SocialWelfare().score(valuation_profile)
distortion = np.max(social_welfare) / social_welfare[winner]
# TODO: make function for distortion calculation given winner (MOST IMPORTANT)
# TODO: make scf for SocialWelfare

from socialchoicekit.deterministic_scoring import *
rules = [Plurality, Borda, Veto, Harmonic]
for r in rules:
  voting_rule = LambdaPRV(3)
  elicitor = ValuationProfileElicitor(valuation_profile)
  winner = voting_rule.scf(profile, elicitor)
  social_welfare = SocialWelfare().score(valuation_profile)
  distortion = np.max(social_welfare) / social_welfare[winner]

# TODO: make routine to generate ordinal data?
# TODO: allow researcher to specify dataset(s), data generation rule(s), voting rule(s) to do this in one function and return distortion

# Goal:
# Both TSM and data generation
# data generation: both with profile specified and not specified.
