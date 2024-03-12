.. _getting_started:

Getting Started
===============

Installation
------------

When this library is published, you will be able to install it using pip:

.. code-block:: bash

    pip install socialchoicekit

Python Version
^^^^^^^^^^^^^^

This library runs with Python 3.8 and above. This library is tested with 3.8.12.

Dependencies
^^^^^^^^^^^^

This library depends on the following packages:

- `numpy <https://numpy.org/>`_
- `scipy <https://scipy.org/>`_: for the maximum weighted matching flow algorithm
- `preflibtools <https://preflib.github.io/preflibtools/>`_: for Preflib integration

Quick Start Example
-------------------

.. code-block:: python

  import numpy as np

  from preflibtools.instances import OrdinalInstance

  from socialchoicekit.preflib_utils import preflib_soc_to_profile
  from socialchoicekit.data_generation import UniformValuationProfileGenerator
  from socialchoicekit.deterministic_scoring import Plurality, SocialWelfare
  from socialchoicekit.elicitation_voting import KARV
  from socialchoicekit.elicitation_utils import ValuationProfileElicitor, SynchronousStdInElicitor

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
  print("Distortion: ", optimal_welfare / social_welfare[plurality_winner - 1])

  # 3) Elicitation (query)-based voting
  print("----- 3) Elicitation-based voting -----")
  karv = KARV(k=3)
  valuation_profile_elicitor = ValuationProfileElicitor(valuation_profile=valuation_profile, memoize=True)
  stdin_elicitor = SynchronousStdInElicitor(memoize=True)
  karv_winner = karv.scf(profile, valuation_profile_elicitor)
  # karv_winner = karv.scf(profile, stdin_elicitor)
  print("KARV winner: ", karv_winner)
  print("Distortion: ", optimal_welfare / social_welfare[karv_winner - 1])

User Guide
----------

To understand how to work with ordinal and cardinal profiles on socialchoicekit, see :ref:`profiles`.
