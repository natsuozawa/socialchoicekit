# socialchoice-kit (provisional name)

socialchoice-kit aims to be a comprehensive implementation of the most important rules in computational social choice theory. It is currently in development by Natsu Ozawa under the supervision of Dr. Aris Filos-Ratsikas at the University of Edinburgh.

This library supports Python 3.7 and above.

# Example Usage

```
from socialchoicekit.deterministic_scoring import Plurality

rule = Plurality()
profile = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
rule.scf(profile)
```
