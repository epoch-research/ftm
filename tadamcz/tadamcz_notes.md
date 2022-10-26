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

However, I don't necessarily recommend this, see [other section](#conditional-distributions).

# Conditional distributions
You're using conditional distributions in a few places in [`distributions.py`](/opmodel/stats/distributions.py):

- the `.rvs` method of `JointDistribution`
- the `sample_normal_cond` method of `GaussianCopula`

I may be missing something about your requirements, but I see some problems here.

You're introducing a complicated new functionality to the `rvs` methods, when this is not necessary. The conditional distribution `D|X=x` is of course another distribution, with `X` fixed at a precise value. So you could create another instance of `JointDistribution` with all parameters unchanged except that `X` is a degenerate distribution with the value `x`. Then you'd be dealing with another "vanilla" `JointDistribution` object. You can more easily rely on the behaviour of that object because of the tests in `copula_wrapper` and `stasmodels`. It's easy to make a mistake when overriding the behaviour of an object (especially if you don't have tests), better to use existing objects if you can. (For convenience, I could add a `conditionalize` method to `JointDistribution` that would return the appropriate new `JointDistribution` object.)

For example, in this line:

```python
rvs = self.wrapped.rvs(nobs=nobs, random_state=random_state, cop_args = fixed_values)
```

You have to worry about the interpretation of `cop_args`, which I'm not sure of (e.g. would the values need to be transformed in some way?). It's easier not to have to worry about that.

# Copulas
I don't understand why you're re-implementing `GaussianCopula`. The point of my `copula_wrapper` package was to avoid this.