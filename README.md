# socialchoice-kit (provisional name)

[![Netlify Status](https://api.netlify.com/api/v1/badges/b284a5ad-ff4f-4acd-98f8-7ee0c5ed08fb/deploy-status)](https://app.netlify.com/sites/socialchoice-kit/deploys)

socialchoice-kit aims to be a comprehensive implementation of the most important rules in computational social choice theory. It is currently in development by Natsu Ozawa under the supervision of Dr. Aris Filos-Ratsikas at the University of Edinburgh.

This library supports Python 3.7 and above.

# Example Usage

```
from socialchoicekit.deterministic_scoring import Plurality

rule = Plurality()
profile = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
rule.scf(profile)
```

# Compile documentation
Sphinx with autodoc is used to compile documentation.

(Run this command when a new module is added)
```
sphinx-apidoc -o docs/ socialchoicekit/
```

```
cd docs
make html
```

To locally view the compiled documentation, use
```
cd docs/_build/html
python -m http.server
```
