.. _profiles:

Profiles User Guide
===================

Preference Profiles
-------------------

In socialchoicekit, we express an ordinal profile as a Python numpy array. The :class:`socialchoicekit.profile_utils.Profile` object is a subclass of ``numpy.ndarray``. The following are subclasses of :class:`socialchoicekit.profile_utils.Profile`:

- :class:`socialchoicekit.profile_utils.StrictProfile`: Profile without ties.
- :class:`socialchoicekit.profile_utils.ProfileWithTies`: Profile with ties.
- :class:`socialchoicekit.profile_utils.CompleteProfile`: Profile without missing values.
- :class:`socialchoicekit.profile_utils.IncompleteProfile`: Profile with missing values. Use ``np.nan`` to represent missing values.
- :class:`socialchoicekit.profile_utils.StrictCompleteProfile`: Corresponds to SoC in Preflib.
- :class:`socialchoicekit.profile_utils.StrictIncmpleteProfile`: Corresponds to SoI in Preflib.
- :class:`socialchoicekit.profile_utils.CompleteProfileWithTies`: Corresponds to ToC in Preflib.
- :class:`socialchoicekit.profile_utils.IncompleteProfileWithTies`: Corresponds to ToI in Preflib.

We can use these to express preference profiles for any of the three :ref:`settings` (voting, resource allocation, and stable matching).
The following is an example of initializing a profile without ties and missing values.

Consider the following ordinal profile. For example, agent 1 prefers alternative 4 the most, then alternative 3, then alternative 1, and prefers alternative 2 the least. Then, agent 1 would assign a rank of 1 to alternative 4, a rank of 2 to alternative 3, a rank of 3 to alternative 1, and a rank of 4 to alternative 2. We express this in the numpy array.

+--------+--------------------------------------------------+
| Agents | Ballot                                           |
+========+==================================================+
| 1      | :math:`4 \succsim_1 3 \succsim_1 1 \succsim_1 2` |
+--------+--------------------------------------------------+
| 2      | :math:`1 \succsim_2 2 \succsim_2 3 \succsim_2 4` |
+--------+--------------------------------------------------+
| 3      | :math:`3 \succsim_3 2 \succsim_3 4 \succsim_3 1` |
+--------+--------------------------------------------------+

.. code-block:: python

  from socialchoicekit.profile_utils import StrictCompleteProfile
  StrictCompleteProfile.of(
    np.array([
      [3, 4, 2, 1], # agent 1
      [1, 2, 3, 4], # agent 2
      [4, 2, 1, 3], # agent 3
    ])
  )

We also express cardinal profiles as numpy arrays. The :class:`socialchoicekit.profile_utils.ValuationProfile` object is a subclass of ``numpy.ndarray``. The following are subclasses of :class:`socialchoicekit.profile_utils.ValuationProfile`:

- :class:`socialchoicekit.profile_utils.CompleteValuationProfile`: Valuation profile without missing values.
- :class:`socialchoicekit.profile_utils.InompleteValuationProfile`: Valuation profile with missing values. Use ``np.nan`` to represent missing values.
- :class:`socialchoicekit.profile_utils.IntegerValuationProfile`: Valuation profile with integer utilities. This must be complete because ``np.nan`` cannot be used inside an integer array.

+-----------------------+------+------+------+------+
| Agents \ Alternatives | 1    | 2    | 3    | 4    |
+-----------------------+------+------+------+------+
| 1                     | 0.25 | 0.1  | 0.3  | 0.35 |
+-----------------------+------+------+------+------+
| 2                     | 0.5  | 0.2  | 0.16 | 0.14 |
+-----------------------+------+------+------+------+
| 3                     | 0.9  | 0.05 | 0.03 | 0.02 |
+-----------------------+------+------+------+------+


If in the above example the agents had the above cardinal profile, we can express it in socialchoicekit as follows:

.. code-block:: python

  from socialchoicekit.profile_utils import CompleteValuationProfile
  CompleteValuationProfile.of(
    np.array([
      [0.25, 0.1, 0.3, 0.35], # agent 1
      [0.5, 0.2, 0.16, 0.14], # agent 2
      [0.9, 0.05, 0.03, 0.02], # agent 3
    ])
  )

Preflib Integration
-------------------

Instead of manually creating profiles, we support loading data directly from the `Preflib <https://preflib.github.io/>`_.

- :func:`socialchoicekit.preflib_utils.preflib_soc_to_profile`: Load a Preflib dataset in SoC form to :class:`socialchoicekit.profile_utils.StrictCompleteProfile`.
- :func:`socialchoicekit.preflib_utils.preflib_soi_to_profile`: Load a Preflib dataset in SoI form to :class:`socialchoicekit.profile_utils.StrictIncompleteProfile`.
- :func:`socialchoicekit.preflib_utils.preflib_toc_to_profile`: Load a Preflib dataset in ToC form to :class:`socialchoicekit.profile_utils.CompleteProfileWithTies`.
- :func:`socialchoicekit.preflib_utils.preflib_toi_to_profile`: Load a Preflib dataset in ToI form to :class:`socialchoicekit.profile_utils.IncompleteProfileWithTies`.
- :func:`socialchoicekit.preflib_utils.preflib_categorical_to_profile`: Load a Preflib dataset in categorical form to :class:`socialchoicekit.profile_utils.IncompleteProfileWithTies`.

.. code-block:: python

  from preflibtools.instances import OrdinalInstance
  from socialchoicekit.preflib_utils import preflib_soc_to_profile

  url = 'https://www.preflib.org/static/data/agh/00009-00000001.soc'
  instance = OrdinalInstance()
  instance.parse_url(url)
  profile = preflib_soc_to_profile(instance)

Profile Generation
------------------

We have functions to (trivially) generate or convert profiles.

- :func:`socialchoicekit.profile_utils.compute_ordinal_profile`: Compute the ordinal profile given a cardinal profile.
- :class:`socialchoicekit.data_generation.NormalValuationProfileGenerator`: Generate a cardinal profile that is consistent with a given ordinal profile, using a normal distribution of values.
- :class:`socialchoicekit.data_generation.UniformValuationProfileGenerator`: Generate a cardinal profile that is consistent with a given ordinal profile, using a uniform distribution of values.
- :func:`socialchoicekit.profile_utils.incomplete_profile_to_complete_profile`: Fill in missing values in an ordinal profile with the least preferred rank.
- :func:`socialchoicekit.profile_utils.incomplete_valuation_profile_to_complete_valuation_profile`: Fill in missing values in a cardinal profile with utility of 0.
- :func:`socialchoicekit.profile_utils.profile_with_ties_to_strict_profile`: Break ties in a profile with ties.

.. code-block:: python

  valuation_profile = UniformValuationProfileGenerator(high=1, low=0, seed=1).generate(profile)
