"""
Utilities.
"""

# Import display in non IPython environments
try:
  from IPython.display import display
except ModuleNotFoundError: 
  # As a fallback, just print
  def display(x):
    print(x)
