.. _concepts:

Overview and Concepts
=====================

Social Choice Theory
--------------------

**Social choice theory** [S1986]_ is the study of aggregating individual preferences into a collective decision. Social choice theory has been used to formulate, analyze, and evaluate decision making processes in a number of settings.
Social choice theory is an extremely interdisciplinary field, with origins in economics.
**Computational social choice theory** [BCELP2016]_ is an active research area that examines the application of computational techniques and paradigms to social choice theory and the application of social choice theoretical concepts to computational environments.

For a non-computer science introduction to social choice theory, the article by `Stanford Encyclopedia of Philosophy <https://plato.stanford.edu/entries/social-choice/>`_ is a good starting point:

The Handbook of Computational Social Choice is a detailed resource with a good introductory chapter [BCELP2016]_. Access to the PDF is provided for free `here <https://procaccia.info/wp-content/uploads/2020/03/comsoc.pdf>`_.

Settings
--------
In this library, we use the following settings.

- **Voting** (General social choice): Select one alternative that best satisfies individual preferences.
- **Resource allocation** (One-sided matching): Assign agents to items while maximizing satisfaction of agent preferences.
- **Stable matching** (Two-sided matching): Assign agents to other agents while maximally satisfying agent preferences and satisfying the *stability* property.

Voting
^^^^^^

In the voting setting, we have

- :math:`n` **agents** or voters, denoted by the set :math:`N = \{1, 2, \ldots, n\}`.
- :math:`m` **alternatives** or candidates, denoted by the set :math:`A = \{1, 2, \ldots, m\}`.
- A **ballot** for each agent, denoted by a linear ordering :math:`\succsim_i` over the alternatives. A linear ordering is a binary relation that is transitive, complete, and antisymmetric.
- An **ordinal profile**, denoted by the family of ballots :math:`P = \sigma = (\succsim_i)_{i \in N}`. An ordinal profile contains information about how each agent ranks the alternatives, but does not contain any numeric information about the strength of the preferences. In socialchoicekit, we simply refer to these as **profiles**.
- A **cardinal profile**, denoted by the matrix :math:`v = (v_{i,\,j})_{i \in N, j \in M}`. A cardinal profile contains numeric information about the strength of the preferences of each agent for each alternative. This subsumes the ranking information contained in the ordinal profile. In socialchoicekit, we refer to these as **valuation profiles**.

Sometimes, Voting problems are solved by voting rules, the following of which are implemented in socialchoicekit.

- :class:`socialchoicekit.deterministic_scoring.Plurality`
- :class:`socialchoicekit.deterministic_scoring.Borda`
- :class:`socialchoicekit.deterministic_scoring.Harmonic`
- :class:`socialchoicekit.deterministic_scoring.KApproval`
- :class:`socialchoicekit.deterministic_scoring.Veto`
- :class:`socialchoicekit.deterministic_tournament.Copeland`
- :class:`socialchoicekit.deterministic_multiround.SingleTransferableVote`
- More voting rules, including randomized versions of the above

A **social choice function** (scf) for a voting rule is a function that takes an ordinal profile as input and returns one winner (or optionally more, in the case of a tie). A **social welfare function** (swf) for a voting rule is a function that takes a cardinal profile as input and returns a ranking of the alternatives.

We call algorithms that take a cardinal profile isntead of an ordinal profile as input **cardinal algorithms**. :class:`socialchoicekit.deterministic_scoring.SocialWelfare` is a cardinal algorithm for scoring rules. Note that there is a unique optimal solution given a cardinal profile.

Resource Allocation
^^^^^^^^^^^^^^^^^^^

In the resource allocation setting, we have the same setting as voting, but instead:

- :math:`m` **items** instead of alternatives.
- In our implementation, we assume that an algorithm allocates one item to each agent. It is possible that an agent is not allocated any items.

The following ordinal algorithms are implemented in socialchoicekit.

- :class:`socialchoicekit.randomized_allocation.RandomSerialDictatorship`
- :class:`socialchoicekit.randomized_allocation.ProbablisticSerial`
- :class:`socialchoicekit.randomized_allocation.SimultaneousEating`
- More algorithms

:class:`socialchoicekit.deterministic_allocation.MaximumWeightMatching` is a cardinal algorithm for resource allocation. Note that there is a unique optimal solution given a cardinal algorithm.

Stable Matching
^^^^^^^^^^^^^^^

In the stable matching setting, we have the same setting as resource allocation, but instead:

- :math:`n` agents from one group and :math:`m` agents from a second group.
- There are two ordinal profiles, one for each group.
- There are two cardinal profiles, one for each group.
- A **two-sided matching** is a set of pairs of agents, where an agent from the first group is matched to an agent from the second group. Depending on the problem, an agent from one group may be matched to multiple agents (see below). In this case, there would be a pair for each combination.
- A **stable matching** is a matching where there are no pairs :math:`(h, r), (h', r')` such that

  - :math:`h` prefers :math:`r'` to :math:`r` and
  - :math:`r'` prefers :math:`h` to :math:`h'`
  - The intuition for this is that if there was such a pair, then the agents would prefer to be matched to each other instead of their current partners.

This problem was first introduced by Gale and Shapley [GS1962]_ as the hospital resident problem, where the aim was to match hospitals to multiple residents (trainee doctors). socialchoicekit has an implementation of the classical algorithm which takes as input two ordinal profiles (each corresponding) :class:`socialchoicekit.deterministic_matching.GaleShapley`.

A cardinal algorithm to this problem was proposed by Irving [I1987]_ and implemented in :class:`socialchoicekit.deterministic_matching.Irving`.

Distortion
----------

In this library, we especially focus on algorithms that are used in the study of distortion. Distortion [PR2006]_ is the worst case ratio between the optimal utility obtainable from cardinal information and the optimal utility obtainable from an algorithm using limited preference information.

Formally, distortion for voting is defined as

.. math::
  distortion(f) = \sup_{N, A, v} \frac{\max_{j \in A} SW(j|v)}{SW(f(P)|v)}

where :math:`f` is the ordinal algorithm, and `SW` is the cardinal algorithm. We can derive similar definitions for the other two settings.

The best achievable distortion by deterministic voting rules is :math:`\Theta(m^2)` [CP2011]_.
Randomization allows for a significantly lower distortion, with the best possible distortion of :math:`\Theta(\sqrt{m})` [BCHLPS2015]_ [EKPS2022]_.

For a comprehensive survey on distortion on the properties known, see [AFSV2021]_.

Elicitation
-----------

Distortion worst case bounds are high given only the ordinal profile, but it is possible to achieve a much lower distortion given a little more information.

**Elicitation** is a technique where additional queries are made to obtain the cardinal values for a subset of alternatives.
While obtaining a complete cardinal profile is hard, this may still be feasible.

[ABFV2021]_ proposed an algorithm for voting that made :math:`O(k \log{m})` queries per agent to achieve :math:`O(m^{\frac{1}{k+1}})` distortion.
Under this, with :math:`O(\log^2{m})` queries per agent :math:`O(1)` distortion is achieved. This is implemented in :class:`socialchoicekit.elicitation_voting.KARV`.
[ABFV2022]_ proposed an algorithm with the same characteristics that works with resource allocation. This is implemented in :class:`socialchoicekit.elicitation_allocation.LambdaTSF`.
We also propose an algorithm with the same characteristics that works with stable matching, using [I1987]_ under the hood. This is implemented in :class:`socialchoicekit.elicitation_matching.DoubleLambdaTSF`.

[ABFV2022a]_ also showed an algorithm that achieves good distortion with just two queries for voting (under limited circumstances) and resource allocation.
We implemented a version of this algorithm for resource allocation in :class:`socialchoicekit.elicitation_allocation.MatchTwoQueries`.

References
----------

.. [ABFV2021] Georgios Amanatidis, Georgios Birmpas, Aris Filos-Ratsikas, and Alexandros A. Voudouris. Peeking behind the ordinal curtain: Improving distortion via cardinal queries. Artificial Intelligence, 296:103488, 2021.
.. [ABFV2022] Georgios Amanatidis, Georgios Birmpas, Aris Filos-Ratsikas, and Alexandros A. Voudouris. A few queries go a long way: Information-distortion tradeoffs in matching. Journal of Artificial Intelligence Research, 74:226–261, 2022.
.. [ABFV2022a] Georgios Amanatidis, Georgios Birmpas, Aris Filos-Ratsikas, and Alexandros A. Voudouris. Don’t roll the dice, ask twice: The two-query distortion of match- ing problems and beyond. In S. Koyejo, S. Mohamed, A. Agarwal, D. Bel- grave, K. Cho, and A. Oh, editors, Advances in Neural Information Process- ing Systems 35 (NeurIPS 2022), volume 35 of Advances in Neural Information Processing Systems, pages 30665–30677. Curran Associates Inc, 2023. URL https://neurips.cc/Conferences/2022. The 36th Conference on Neural In- formation Processing Systems, 2022, NeurIPS 2022 ; Conference date: 28-11-2022 Through 09-12-2022.
.. [AFSV2021] Elliot Anshelevich, Aris Filos-Ratsikas, Nisarg Shah, and Alexandros A. Voudouris. Distortion in social choice problems: The first 15 years and beyond. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence (IJCAI-21), pages 4294–4301, 2021.
.. [BCELP2016] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jerome Lang, and Ariel D. Procaccia, editors. Handbook of computational social choice. Cambridge University Press, 2016.
.. [BCHLPS2015] Craig Boutilier, Ioannis Caragiannis, Simi Haber, Tyler Lu, Ariel D. Procaccia, and Or Sheffet. Optimal social choice functions: A utilitarian view. Artificial Intelligence, 227:190–213, 2015.
.. [CP2011] Ioannis Caragiannis and Ariel D. Procaccia. Voting almost maximizes social welfare despite limited communication. Artificial Intelligence, 175(9-10):1655–1671, 2011.
.. [EKPS2022] Soroush Ebadian, Anson Kahng, Dominik Peters, and Nisarg Shah. Optimized distortion and proportional fairness in voting. In Proceedings of the 23rd ACM Conference on Economics and Computation (EC ’22), page 38 pages, Boulder, CO, USA, 2022. ACM. July 11-15.
.. [GS1962] David Gale and Lloyd Stowell Shapley. College admissions and the stability of marriage. American Mathematical Monthly, 69:9–15, 1962.
.. [I1987] Robert W. Irving, Paul Leather, and Dan Gusfield. An efficient algorithm for the “optimal” stable marriage. Journal of the Association for Computing Machinery,, 34(3):532–543, 1987.
.. [PR2006] Ariel D. Procaccia and Jeffrey S. Rosenschein. The distortion of cardinal preferences in voting. In International Workshop on Cooperative Information Agents (CIA), pages 317–331, 2006.
.. [S1986] Amartya Sen. Social choice theory. In Handbook of mathematical economics, volume 3, pages 1073–1181. 1986.
