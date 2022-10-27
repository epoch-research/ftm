# Dependency management
You have a `requirements.txt` file, but it does not match the actual dependencies of your project. For example, you depend on `scipy` and `statsmodels`, but these are not included.

## Basics
Basic best practices for dependency management include:
- Using a package manager
- Using virtual environment

Good dependency management is essential. Otherwise, multiple people working on the same project might not have the same environment (the "it works on my machine" problem). If you don't use virtual environments, your program might randomly break when its dependencies are updated by some other work you're doing on the same computer. If you ever want to deploy your application to a server, you also have to specify dependencies exhaustively.

There are many guides to dependency management online. I generally find the content from RealPython to be quite good, and they have this [course](https://realpython.com/courses/managing-python-dependencies/) on dependency management. (Since it's not fee, I haven't been able to read it).

## Choice of package manager
Dependency management in Python has long been widely known to be much messier than other leading languages. With the arrival of more modern tools, Python has caught up, but it's still a worse experience than e.g. Ruby.

`pip` is the traditional package manager / downloader. There are a number of newer tools that are IMO greatly superior, so that one should simply not use `pip` in 2022. The ones I know of are `pipenv`, `conda` and `poetry`. I use [Poetry](https://python-poetry.org/) on all my projects; I weakly think it's the best one. 

## Advanced topic: Python version management
Ideally, you'd also isolate the version of Python that is being used by your project, instead of just using the system version. If you use Poetry, it will let you specify version(s) in the `pyproject.toml` file.

There are several tools for using different version of Python in parallel on your computer. The one I use is [`pyenv`](https://github.com/pyenv/pyenv).

# Subclassing
In [`distributions.py`](/opmodel/stats/distributions.py) it looks like you've copied and pasted the entire `JointDistribution` class, as far as I can tell in order to modify the `.rvs` method to allow for conditional sampling.

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

# Conditional distributions
You're using conditional distributions in a few places in [`distributions.py`](/opmodel/stats/distributions.py):

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
