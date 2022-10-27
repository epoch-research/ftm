TODO:
- Google sheets
- general introduction to testing
- suggested ideas for tests
- how to write modular code
- when to use classes vs functions
- how to compare different versions of the model? (existing approach: diffy.py)


# Dependency management
## Basics
Basic best practices for dependency management include:
- Using virtual environment
- Specifying version numbers

Using a good package manager simplifies these tasks.

Good dependency management is essential:
- If you don't specify dependencies, a new user has to manually track down every missing dependency and install it on their machine before being able to run the code!! If your user (correctly) uses virtual envs, every missing dependency will be _every_ dependency.
- If you don't specify dependencies precisely enough (e.g. a `requirements.txt` without version numbers), multiple people working on the same project might not have the same environment (the "it works on my machine" problem)
- If you don't use virtual environments, your program might randomly break when its dependencies are updated by some other work you're doing on the same computer. If you ever want to deploy your application to a server, you also have to specify dependencies exhaustively.
- etc, etc...

There are many guides to dependency management online. I generally find the content from RealPython to be quite good, and they have this [course](https://realpython.com/courses/managing-python-dependencies/) on dependency management. (Since it's not fee, I haven't been able to read it).

## Choice of package manager
Dependency management in Python has long been widely known to be much messier than other leading languages. With the arrival of more modern tools, Python has caught up, but it's still a worse experience than e.g. Ruby.

`pip` is the traditional package manager / downloader. There are a number of newer tools that are IMO greatly superior, so that one should simply not use `pip` in 2022. The ones I know of are `pipenv`, `conda` and `poetry`. I use [Poetry](https://python-poetry.org/) on all my projects; I weakly think it's the best one. 

## Advanced topic: Python version management
Ideally, you'd also isolate the version of Python that is being used by your project, instead of just using the system version. If you use Poetry, it will let you specify version(s) in the `pyproject.toml` file.

There are several tools for using different version of Python in parallel on your computer. The one I use is [`pyenv`](https://github.com/pyenv/pyenv).

# Subclassing
In [`distributions.py`](/opmodel/distributions.py) it looks like you've copied and pasted the entire `JointDistribution` class, as far as I can tell in order to modify the `.rvs` method to allow for conditional sampling.

Almost all languages, and in particular so-called "object-oriented" languages, support a feature called "subclassing" (see e.g. [Inheritance and Composition: A Python OOP Guide](https://realpython.com/inheritance-composition-python/)). Subclassing lets you take a class and modify just some aspects of its functionality, inheriting the rest from the parent class. The basic syntax is `class ChildClass(BaseClass)`. For example:

```python
class BaseClass:
    def method_A(self):
        return "A"
    def method_B(self):
        return "B"

class ChildClass(BaseClass):
    def method_B(self):
        return "beta"
```

```python-repl
>>> c = ChildClass()
>>> c.method_A()
A
>>> c.method_B()
beta
```

So there is no need to copy and paste entire class definitions. You could simply have written:

```python
from copula_wrapper import JointDistribution

class ConditionalJointDistribution(JointDistribution):
    def rvs(self):
        # Get conditional values
        fixed_values = [marginal.cdf(conditions.get(name, np.nan)) for (name, marginal) in self.marginals.items()]
        
        # ... continuation of your code here
```

Your copy and paste approach also forces you to copy and paste the `get_pearsons_rho` function. If you took `copula_wrapper` as a dependency you wouldn't have to worry about its internals.


# Imports
You may benefit from reading the [Python tutorial on modules](https://docs.python.org/3/tutorial/modules.html)

## Relative imports
Relative imports such as these
```python
from ..core.utils import log, get_parameter_table, get_rank_correlations, get_clipped_ajeya_dist
```
Are generally considered bad practise. They mean that your code will behave differently depending on where it's being imported/called from! And that may not be the place [you think](https://stackoverflow.com/a/65589847/8010877):

> If you write from . import module, opposite to what you think, module will not be imported from current directory, but from the top level of your package! If you run .py file as a script, it simply doesn't know where the top level is and thus refuses to work.

It's generally considered better to do the explicit:

```python
from opmodel.core.utils import log, get_parameter_table, get_rank_correlations, get_clipped_ajeya_dist
```

In `megareport.py` you have exactly this situation:

```python
from . import log
from . import *

from .report import Report
```
```
❯ python opmodel/analysis/megareport.py
Traceback (most recent call last):
  File "/Users/t/repos/opmodel/opmodel/analysis/megareport.py", line 1, in <module>
    from . import log
ImportError: attempted relative import with no known parent package
```

I see in [your README.md](../README.md) that you're using the `-m` option to run it. As you can see by running `python -h`, this option means "run library module as a script". I prefer to keep package modules on the one hand, and scripts on the other, completely non-overlapping. IMO, a package module should not do script-like stuff like generating a report, and any script should be runnable directly (I've never used `python -m` before). 

## Don't import in `__init__.py`
There's ≈never any reason to do this:

```python
# opmodel/analysis/__init__.py

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
```

Import modules where you need them, not in the package initializer. 

The tutorial [explains](https://docs.python.org/3/tutorial/modules.html#packages) what `__init__` files are used for:

> The `__init__.py` files are required to make Python treat directories containing the file as packages. This prevents directories with a common name, such as string, unintentionally hiding valid modules that occur later on the module search path. In the simplest case, `__init__.py` can just be an empty file, but it can also execute initialization code for the package or set the `__all__` variable, described later.

## Don't `import *`

One should never do this:

```python
from opmodel.core.utils import *
```

This is explained well [on StackOverflow](https://stackoverflow.com/questions/2386714/why-is-import-bad) and many other places. Why is `import *` bad?

> - Because it puts a lot of stuff into your namespace (might shadow some other object from previous import and you won't know about it).
> - Because you don't know exactly what is imported and can't easily find from which module a certain thing was imported (readability).
> - Because you can't use cool tools like pyflakes to statically detect errors in your code.

# Conditional distributions
You're using conditional distributions in a few places in [`distributions.py`](/opmodel/distributions.py):

- the `.rvs` method of `JointDistribution`
- the `sample_normal_cond` method of `GaussianCopula`

There was an extended discussion about this [on Slack](https://tadamcz.slack.com/archives/C03GMA1PC03/p1666795355505989).

I originally thought you could achieve what you want without even modifying `JointDistribution`, but this may not be possible (thanks Eduardo!). 

If you _do_ have to access the wrapped `CopulaDistribution`, then you should do this by modifying the functionality of `JointDistribution` (you can subclass, or you can open a PR to `copula_wrapper`), not by accessing `joint_distribution.wrapped` anywhere else (I don't think I've seen you do this anywhere, but just to be clear).

## Meaning of cop_args?
Have you tested that this line:

```python
rvs = self.wrapped.rvs(nobs=nobs, random_state=random_state, cop_args = fixed_values)
```

does what you expect?

The following from the `statsmodels` docs makes me think `cop_args` is for parametrising the distribution, not for evaluating it at particular points. But I don't understand copulas well at all, so I may well be wrong. If you've tested this, then great!

```
cop_args : tuple
    Copula parameters. If None, then the copula parameters will be
    taken from the ``cop_args`` attribute created when initiializing
    the instance.
marg_args : list of tuples
    Parameters for the marginal distributions. It can be None if none
    of the marginal distributions have parameters, otherwise it needs
    to be a list of tuples with the same length has the number of
    marginal distributions. The list can contain empty tuples for
    marginal distributions that do not take parameter arguments.
```

## Two implementations?
I don't understand why you have the modified `JointDistribution` _as well as_ the `class GaussianCopula(sm_api.GaussianCopula)`. Is this duplicate functionality? 

The method `sample_normal_cond` is very hard to follow. It makes me think "a mathematician wrote this code" :).

# The `ParamsDistribution` class
## Constructor
From the `__init__` (or constructor) it looks like it's a method specific to your model (e.g. you're calling `get_parameter_table()`). In that case, I would make that more obvious in the name, like `TakeoffParamsDistr`. `ParamsDistribution` makes it seem like something more generic. If it is something more generic, then you should pass in the "parameter table" etc. as arguments, not have them hard-coded in the `__init__`. 

For readability, it would be good to break up the method a bit. 

You're setting up the three dictionaries `pairwise_rank_corr`, `marginals`, and `directions` in one big block. Instead, you could have the methods `marginals` and `correlations` for example. 

Here's an illustration of `correlations` as a method:

```python
class ParamsDistribution:
	def __init__(self):
		# ... more code here

		self.pairwise_rank_corr = self.correlations(ignore_rank_correlations, marginals, rank_correlations, directions)

	def correlations(self, ignore_rank_correlations, marginals, rank_correlations, directions):
		correlations = {}
		if not ignore_rank_correlations:
			for left in marginals.keys():
				for right in marginals.keys():
					if right not in rank_correlations or left not in rank_correlations:
						continue

					if isinstance(marginals[right], PointDistribution) or isinstance(marginals[left],PointDistribution):
						continue

				r = rank_correlations[right][left]
				if not np.isnan(r) and r != 0:
					correlations[(left, right)] = r * directions[left] * directions[right]
		return correlations
```

Apart from breaking separate functionality into more readable chunks, this has some other advantages. First, you can explicitly see what the correlations depend on simply by looking at the method signature. Also, you can move the statement `if not ignore_rank_correlations` to a place where it has less impact on readability (imo):

```python
class ParamsDistribution:
	def __init__(self):
		# ... more code here
		
		if not ignore_rank_correlations:
			self.pairwise_rank_corr = self.correlations(marginals, rank_correlations, directions)
		else:
			self.pairwise_rank_corr = {}

	def correlations(self, marginals, rank_correlations, directions):
		correlations = {}
		for left in marginals.keys():
			for right in marginals.keys():
				if right not in rank_correlations or left not in rank_correlations:
					continue

				if isinstance(marginals[right], PointDistribution) or isinstance(marginals[left],PointDistribution):
					continue

			r = rank_correlations[right][left]
			if not np.isnan(r) and r != 0:
				correlations[(left, right)] = r * directions[left] * directions[right]
		return correlations
```