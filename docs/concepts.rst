.. _concepts:

Concepts
========

Social Choice Theory
--------------------

**Social choice theory** [S1986]_ is the study of aggregating individual preferences into a collective decision. Social choice theory has been used to formulate, analyze, and evaluate decision making processes in a number of settings.
Social choice theory is an extremely interdisciplinary field, with origins in economics.
**Computational social choice theory** [BCELP2016]_ is an active research area that examines the application of computational techniques and paradigms to social choice theory and the application of social choice theoretical concepts to computational environments.

For a non-computer science introduction to social choice theory, the article by `Stanford Encyclopedia of Philosophy <https://plato.stanford.edu/entries/social-choice/>`_ is a good starting point:

The Handbook of Computational Social Choice is a detailed resource with a good introductory chapter [BCELP2016]_. Access to the PDF is provided for free `here <https://procaccia.info/wp-content/uploads/2020/03/comsoc.pdf>`_.

Distortion
----------

In this library, we especially focus on algorithms that are used in the study of distortion. Distortion [PR2006]_ is the worst case ratio between the optimal utility obtainable from cardinal information and the optimal utility obtainable from an algorithm using limited preference information.

Settings
--------
In this library, we use the following settings.

- **Voting** (General social choice): Select one alternative that best satisfies individual preferences.
- **Resource allocation** (One-sided matching): Assign agents to items while maximizing satisfaction of agent preferences.
- **Stable matching** (Two-sided matching): Assign agents to other agents while maximally satisfying agent preferences and satisfying the *stability* property.

References
----------

.. [BCELP2016] Felix Brandt, Vincent Conitzer, Ulle Endriss, Jerome Lang, and Ariel D. Procaccia, editors. Handbook of computational social choice. Cambridge University Press, 2016.
.. [PR2006] Ariel D. Procaccia and Jeffrey S. Rosenschein. The distortion of cardinal preferences in voting. In International Workshop on Cooperative Information Agents (CIA), pages 317–331, 2006.
.. [S1986] Amartya Sen. Social choice theory. In Handbook of mathematical economics, volume 3, pages 1073–1181. 1986.
