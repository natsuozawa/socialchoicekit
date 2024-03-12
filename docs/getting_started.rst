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

Examples
--------

See the `example` directory for a comprehensive example.

User Guide
----------

Preference Profiles
^^^^^^^^^^^^^^^^^^^

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
