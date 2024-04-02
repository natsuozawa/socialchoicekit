# socialchoicekit

[![Netlify Status](https://api.netlify.com/api/v1/badges/b284a5ad-ff4f-4acd-98f8-7ee0c5ed08fb/deploy-status)](https://app.netlify.com/sites/socialchoicekit/deploys)

socialchoicekit aims to be a comprehensive implementation of the most important rules in computational social choice theory. It is currently in development by Natsu Ozawa under the supervision of Dr. Aris Filos-Ratsikas at the University of Edinburgh.

This library supports Python 3.8 and above.

Documentation can be found [here](https://socialchoicekit.natsuozawa.com/)

# Concepts

## Social Choice Theory

Social choice theory [Sen, 1986] is the study of aggregating individual preferences into a collective decision. Social choice theory has been used to formulate, analyze, and evaluate decision making processes in a number of settings.

In this library, we use the following settings.

- Voting: The goal is to select a candidate or proposal that best represents individual votes by an electoral process. Also known as the general social choice setting, this setting is the most studied.
- Resource allocation: The goal is to match agents to items while respecting preferences of the agent. In this library, we simply call it allocation. Also known as matching with one sided preferences.
- Matching: The goal is to match agents from one group to agents in another while respecting preferences of all agents. Also known as matching with two sided preferences.

## Distortion

In this library, we especially focus on algorithms that are used in the study of distortion. Distortion is the worst case ratio between the optimal utility obtainable from cardinal information and the optimal utility obtainable from an algorithm using limited preference information.

# Usage

## Example Usage

```
from socialchoicekit.deterministic_scoring import Plurality

rule = Plurality()
profile = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
rule.scf(profile)
```

## Development

Create a virtual environment and install from `requirements.txt`
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Compile documentation
Sphinx with autodoc is used to compile documentation.

(Run this command when a new module is added)
```
sphinx-apidoc -e -o docs/ socialchoicekit/
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

## Build

```
python setup.py sdist bdist_wheel
```

If there is an error: invalid command 'bdist_wheel', type:

```
pip install wheel
```

and try again.
