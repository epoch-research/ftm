import numpy as np
import inspect
import matplotlib.pyplot as plt
import pandas as pd
import math
import traceback

from . import utils
from .utils import display

class SimulateTakeOff():
  """ Class to run a simulation of how automation and the economy 
      will feed into each other.

      The basic idea:
        - We update the SotA in hardware and software based on last years' rnd production
        - We update production factors given reinvesment: capital, labour and compute 
        - We allocate those between AI training, production, hardware RnD and software RnD
        - We check which tasks can be automated given the training budget
        - We use a CES production function to estimate goods production
        - We use a CES production function to estimate rnd production
  """
  def __init__(self, 

      # Automation thesholds
      full_automation_requirements_training,
      flop_gap_training,
      goods_vs_rnd_requirements_training,

      full_automation_requirements_runtime,
      flop_gap_runtime,
      goods_vs_rnd_requirements_runtime,

      # Production
      labour_substitution_goods,
      labour_substitution_rnd,
      capital_substitution_goods,
      capital_substitution_rnd,

      # R&D parameters
      hardware_returns,
      software_returns,
      hardware_performance_ceiling,
      software_ceiling,
      rnd_parallelization_penalty,
      hardware_delay,

      # Fractional inputs
      frac_capital_hardware_rnd_growth,
      frac_labour_hardware_rnd_growth,
      frac_compute_hardware_rnd_growth,

      frac_labour_software_rnd_growth,
      frac_compute_software_rnd_growth,

      frac_gwp_compute_growth,
      frac_compute_training_growth,

      frac_capital_hardware_rnd_growth_rampup,
      frac_labour_hardware_rnd_growth_rampup,
      frac_compute_hardware_rnd_growth_rampup,
      
      frac_labour_software_rnd_growth_rampup,
      frac_compute_software_rnd_growth_rampup,

      frac_gwp_compute_growth_rampup,
      frac_compute_training_growth_rampup,

      frac_capital_hardware_rnd_ceiling,
      frac_labour_hardware_rnd_ceiling,
      frac_compute_hardware_rnd_ceiling,

      frac_labour_software_rnd_ceiling,
      frac_compute_software_rnd_ceiling,

      frac_gwp_compute_ceiling,
      frac_compute_training_ceiling,

      # Initial values
      initial_frac_capital_hardware_rnd,
      initial_frac_labour_hardware_rnd,
      initial_frac_compute_hardware_rnd,

      initial_frac_labour_software_rnd,
      initial_frac_compute_software_rnd,
      
      initial_biggest_training_run,
      ratio_initial_to_cumulative_input_hardware_rnd,
      ratio_initial_to_cumulative_input_software_rnd,
      initial_hardware_production,
      ratio_hardware_to_initial_hardware_production,
      initial_hardware_performance,
      initial_gwp,
      initial_cognitive_share_goods,
      initial_cognitive_share_rnd,
      initial_compute_share_goods,
      initial_compute_share_rnd,

      # Other
      rampup_trigger,
      initial_capital_growth,
      labour_growth,
      tfp_growth,
      compute_depreciation,

      # Simulation resolution
      t_start = 2022,
      t_end = 2100,
      t_step = 0.1, # Time between steps in years

      n_labour_tasks = 100,
      ):
    
    # Add all inputs to model parameters
    for item in inspect.signature(SimulateTakeOff).parameters:
      setattr(self, item, eval(item))
    
    # Checks
    self.check_input_validity()

    # Process input parameters
    self.process_input_parameters()

    # Allocate memory for the simulation state
    self.create_simulation_state()
  
  def check_input_validity(self):
    
    assert self.flop_gap_training > 1
    assert self.flop_gap_runtime > 1

    assert self.hardware_delay >= 0

    assert self.t_start < self.t_end
    assert self.t_step > 0

    assert type(self.n_labour_tasks) is int, "n_labour_tasks must be an integer"
    assert self.n_labour_tasks > 0, "n_labour_tasks must be positive"
  
  def process_input_parameters(self):
    # Labour tasks
    self.n_labour_tasks_goods = self.n_labour_tasks
    self.n_labour_tasks_rnd = self.n_labour_tasks

    # Process training and runtime flops
    self.process_automation_costs()

    # Define rampup triggers
    self.rampup_trigger = self.rampup_trigger

    # Define initial quantities
    self.initial_software = 1
    self.initial_tfp_goods = 1
    self.initial_tfp_rnd = 1
    self.initial_rnd_input_hardware = 8000000.0
    self.initial_rnd_input_software = 8000000.0
    self.initial_population = 8e9
    self.investment_rate = 0.2
    self.initial_hardware = \
      self.initial_hardware_production \
      * self.ratio_hardware_to_initial_hardware_production
    
    # Economy shares
    self.initial_capital_share_goods = 1 - self.initial_cognitive_share_goods
    self.initial_capital_share_rnd = 1 - self.initial_cognitive_share_rnd

    self.initial_compute_share_goods = \
      self.initial_compute_share_goods \
      * self.initial_cognitive_share_goods # TERRIBLE HACK
    self.initial_compute_share_rnd = \
      self.initial_compute_share_rnd \
      * self.initial_cognitive_share_rnd # TERRIBLE HACK

    self.initial_labour_share_goods = \
      self.initial_cognitive_share_goods - self.initial_compute_share_goods
    self.initial_labour_share_rnd = \
      self.initial_cognitive_share_rnd - self.initial_compute_share_rnd

    
    # Returns to hardware and software need to be adjusted 
    # by the paralellization penalty
    self.hardware_returns = \
      self.hardware_returns / self.rnd_parallelization_penalty
    self.software_returns = \
      self.software_returns / self.rnd_parallelization_penalty
    
    # Hardware delay is adjusted by timestep
    self.hardware_delay_idx = round(self.hardware_delay / self.t_step)

  def process_automation_costs(self):
    """ Initialize the training and runtime flops for goods and rnd
    """

    # Define requirements and gap for goods and R&D
    self.full_automation_training_flops_goods = self.full_automation_requirements_training
    self.full_automation_runtime_flops_goods = self.full_automation_requirements_runtime
    
    self.full_automation_training_flops_rnd = \
      self.full_automation_requirements_training / self.goods_vs_rnd_requirements_training
    self.full_automation_runtime_flops_rnd = \
      self.full_automation_requirements_runtime / self.goods_vs_rnd_requirements_runtime

    self.automation_training_flop_gap_goods = self.flop_gap_training
    self.automation_training_flop_gap_rnd = self.flop_gap_training

    self.automation_runtime_flop_gap_goods = self.flop_gap_runtime
    self.automation_runtime_flop_gap_rnd = self.flop_gap_runtime

    # Define distribution of requirements
    self.automation_training_flops_goods = \
      SimulateTakeOff.quantiles_from_gap(
          self.full_automation_training_flops_goods,
          self.automation_training_flop_gap_goods,
          )
    self.automation_runtime_flops_goods = \
      SimulateTakeOff.quantiles_from_gap(
          self.full_automation_runtime_flops_goods,
          self.automation_runtime_flop_gap_goods,
          )
    self.automation_training_flops_rnd = \
      SimulateTakeOff.quantiles_from_gap(
          self.full_automation_training_flops_rnd,
          self.automation_training_flop_gap_rnd,
          )
    self.automation_runtime_flops_rnd = \
      SimulateTakeOff.quantiles_from_gap(
          self.full_automation_runtime_flops_rnd,
          self.automation_runtime_flop_gap_rnd,
          )

    self.automation_training_flops_goods =\
      SimulateTakeOff.process_quantiles(self.automation_training_flops_goods, 
                                        self.n_labour_tasks_goods)
    
    self.automation_runtime_flops_goods =\
      SimulateTakeOff.process_quantiles(self.automation_runtime_flops_goods, 
                                        self.n_labour_tasks_goods)
    
    self.automation_training_flops_rnd =\
      SimulateTakeOff.process_quantiles(self.automation_training_flops_rnd, 
                                        self.n_labour_tasks_rnd)
    
    self.automation_runtime_flops_rnd =\
      SimulateTakeOff.process_quantiles(self.automation_runtime_flops_rnd, 
                                        self.n_labour_tasks_rnd)
  
    # Check that no task is yet automatable
    # if np.any(self.automation_training_flops_goods < self.initial_biggest_training_run) \
    # or np.any(self.automation_training_flops_rnd < self.initial_biggest_training_run):
    #   print(f"self.automation_training_flops_goods = {self.automation_training_flops_goods}")
    #   print(f"self.automation_training_flops_rnd = {self.automation_training_flops_rnd}")
    #   print(f"self.initial_biggest_training_run = {self.initial_biggest_training_run}")
    #   raise ValueError("Assumption not met: some tasks are automatable from the beginning.")

    # The first task is always automatable
    self.automation_training_flops_goods = \
      np.insert(self.automation_training_flops_goods, 0, 1.0)
    self.automation_runtime_flops_goods = \
      np.insert(self.automation_runtime_flops_goods, 0, 1.0)
    self.automation_training_flops_rnd = \
      np.insert(self.automation_training_flops_rnd, 0, 1.0)
    self.automation_runtime_flops_rnd = \
      np.insert(self.automation_runtime_flops_rnd, 0, 1.0)

    # Check that the automation costs are monotonic
    if np.any(np.diff(self.automation_training_flops_goods) < 0.) \
    or np.any(np.diff(self.automation_runtime_flops_goods) < 0.) \
    or np.any(np.diff(self.automation_training_flops_rnd) < 0.) \
    or np.any(np.diff(self.automation_runtime_flops_rnd) < 0.):
      raise ValueError("Assumption not met: the automation costs must be monotonically increasing.")
  
  ##############################################################################

  def create_simulation_state(self):
    # Create empty vectors of the correct length 
    # to hold the results of the simulation

    self.timesteps = np.arange(self.t_start, self.t_end, self.t_step)
    self.n_timesteps = len(self.timesteps)

    self.create_simulation_state_investment()
    self.create_simulation_state_automation()
    self.create_simulation_state_production()

  def create_simulation_state_investment(self):
    
    self.create_simulation_state_rnd()
    self.create_simulation_state_total_input()
    self.create_simulation_state_fractional_inputs()

  def create_simulation_state_rnd(self):
    # Hardware RnD
    self.cumulative_rnd_input_hardware = np.zeros(self.n_timesteps)
    self.hardware_performance = np.zeros(self.n_timesteps)

    # Software RnD
    self.cumulative_rnd_input_software = np.zeros(self.n_timesteps)
    self.software = np.zeros(self.n_timesteps)
  
  def create_simulation_state_fractional_inputs(self):

    self.rampup = np.zeros(self.n_timesteps, dtype=bool)
    self.rampup_start = None
    self.rampup_mid = None

    # Goods vs compute investment split
    self.frac_gwp_compute = np.zeros(self.n_timesteps)

    # Total R&D fractional inputs
    self.frac_capital_hardware_rnd = np.zeros(self.n_timesteps)
    self.frac_labour_hardware_rnd = np.zeros(self.n_timesteps)
    self.frac_compute_hardware_rnd = np.zeros(self.n_timesteps)

    self.frac_labour_software_rnd = np.zeros(self.n_timesteps)
    self.frac_compute_software_rnd = np.zeros(self.n_timesteps)

    # Training compute
    self.frac_compute_training = np.zeros(self.n_timesteps)

    # Goods production fractional inputs
    self.frac_capital_goods = np.zeros(self.n_timesteps)
    self.frac_labour_goods = np.zeros(self.n_timesteps)
    self.frac_compute_goods = np.zeros(self.n_timesteps)

  def create_simulation_state_total_input(self):
    self.capital = np.zeros(self.n_timesteps)
    self.labour = np.zeros(self.n_timesteps)
    self.compute_investment = np.zeros(self.n_timesteps)
    self.hardware = np.zeros(self.n_timesteps)
    self.compute = np.zeros(self.n_timesteps)

    self.tfp_goods = np.zeros(self.n_timesteps)
    self.tfp_rnd = np.zeros(self.n_timesteps)
  
  def create_simulation_state_automation(self):
    self.biggest_training_run = np.zeros(self.n_timesteps)
    self.automatable_tasks_goods = np.zeros(self.n_timesteps, dtype=int)
    self.automatable_tasks_rnd = np.zeros(self.n_timesteps, dtype=int)
    self.frac_automatable_tasks = np.zeros(self.n_timesteps)
    self.frac_automatable_tasks_goods = np.zeros(self.n_timesteps)
    self.frac_automatable_tasks_rnd = np.zeros(self.n_timesteps)
    self.task_compute_to_labour_ratio_goods = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_goods+1))
    self.task_compute_to_labour_ratio_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd+1))
    self.automation_years_goods = []
    self.automation_years_rnd = []
    self.agi_year = None

  def create_simulation_state_production(self):
    # Goods production
    self.capital_goods = np.zeros(self.n_timesteps)
    self.labour_goods = np.zeros(self.n_timesteps)
    self.compute_goods = np.zeros(self.n_timesteps)

    self.labour_task_input_goods = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_goods+1))
    self.compute_task_input_goods = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_goods+1))
    self.task_input_goods = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_goods+1))

    self.frac_tasks_automated_goods = np.zeros(self.n_timesteps)

    self.automation_multiplier_goods = np.zeros(self.n_timesteps)
    self.gwp = np.zeros(self.n_timesteps)

    self.capital_share_goods = np.zeros(self.n_timesteps)
    self.cognitive_share_goods = np.zeros(self.n_timesteps)
    self.labour_share_goods = np.zeros(self.n_timesteps)
    self.compute_share_goods = np.zeros(self.n_timesteps)

    # Hardware RnD production
    self.capital_hardware_rnd = np.zeros(self.n_timesteps)
    self.labour_hardware_rnd = np.zeros(self.n_timesteps)
    self.compute_hardware_rnd = np.zeros(self.n_timesteps)

    self.labour_task_input_hardware_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))
    self.compute_task_input_hardware_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))
    self.task_input_hardware_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))

    self.frac_tasks_automated_rnd = np.zeros(self.n_timesteps)
    self.rnd_input_hardware = np.zeros(self.n_timesteps)
    self.automation_multiplier_rnd = np.zeros(self.n_timesteps)

    self.capital_share_hardware_rnd = np.zeros(self.n_timesteps)
    self.cognitive_share_hardware_rnd = np.zeros(self.n_timesteps)
    self.labour_share_hardware_rnd = np.zeros(self.n_timesteps)
    self.compute_share_hardware_rnd = np.zeros(self.n_timesteps)

    # Software RnD production
    self.capital_software_rnd = np.zeros(self.n_timesteps)
    self.labour_software_rnd = np.zeros(self.n_timesteps)
    self.compute_software_rnd = np.zeros(self.n_timesteps)

    self.labour_task_input_software_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))
    self.compute_task_input_software_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))
    self.task_input_software_rnd = \
      np.zeros((self.n_timesteps, self.n_labour_tasks_rnd + 1))

    self.rnd_input_software = np.zeros(self.n_timesteps)

  
  ########################################################################

  def run_simulation(self):
    # Treat numpy warnings as exceptions
    # np.seterr(over='raise')
    np.seterr('raise')

    try:
      for t_idx in range(self.n_timesteps):
        self.t_idx = t_idx
        if t_idx == 0:
          self.initialize_inputs()
        else:
          self.reinvest_output_in_inputs(t_idx)
        self.automate_tasks(t_idx)
        self.production(t_idx)
      
    except FloatingPointError as e: 
      print("An overflow has happened and the simulation has stopped.")
      print(e)
      print(traceback.format_exc())
    finally:
      # Set default error handling
      np.seterr()
    
    # Compute takeoff metrics
    self.compute_metrics()
  
  ########################################################################

  # INPUT INITIALIZATION
  def initialize_inputs(self):
    self.initialize_rnd_state()
    self.initialize_total_inputs()
    self.initialize_fractional_inputs()
  
  def initialize_rnd_state(self):

    ## Hardware
    self.hardware_performance[0] = self.initial_hardware_performance

    self.cumulative_rnd_input_hardware[0]  =                \
      self.initial_rnd_input_hardware                       \
      ** self.rnd_parallelization_penalty                   \
      / self.ratio_initial_to_cumulative_input_hardware_rnd \
      / self.rnd_parallelization_penalty

    ## Software
    self.software[0] = self.initial_software
    
    self.cumulative_rnd_input_software[0]  =                \
      self.initial_rnd_input_software                       \
      ** self.rnd_parallelization_penalty                   \
      / self.ratio_initial_to_cumulative_input_software_rnd \
      / self.rnd_parallelization_penalty

  def initialize_total_inputs(self):
    self.labour[0] = self.initial_population

    # Capital is initialized so that its rate of growth in the first time step
    # matches the gwp rate of growth 
    self.capital[0] =\
      self.initial_gwp * self.investment_rate \
      / (np.exp(self.initial_capital_growth)-1)
    
    self.hardware[0] = self.initial_hardware
    
    self.compute_investment[0] = \
      self.initial_hardware_production * self.t_step \
      / self.initial_hardware_performance # Hardware production is 
    
    self.compute[0] = self.hardware[0] * self.initial_software

    self.tfp_goods[0] = self.initial_tfp_goods
    self.tfp_rnd[0] = self.initial_tfp_rnd
  
  def initialize_fractional_inputs(self):

    self.rampup[0] = False

    # Split of gwp between capital and compute
    self.frac_gwp_compute[0] = self.compute_investment[0] / self.initial_gwp / self.t_step

    # R&D fractional inputs
    self.frac_capital_hardware_rnd[0] = self.initial_frac_capital_hardware_rnd
    self.frac_labour_hardware_rnd[0] = self.initial_frac_labour_hardware_rnd
    self.frac_compute_hardware_rnd[0] = self.initial_frac_compute_hardware_rnd

    self.frac_labour_software_rnd[0] = self.initial_frac_labour_software_rnd
    self.frac_compute_software_rnd[0] = self.initial_frac_compute_software_rnd

    # Training compute is initialized to match initial largest training run
    self.frac_compute_training[0] = \
      self.initial_biggest_training_run \
      / (self.hardware[0] * self.initial_software)
    
    ## Initial compute must be greater than initial training run
    if self.initial_biggest_training_run > self.compute[0]:
      raise ValueError("Initial biggest training run is bigger than available compute")

    # Goods production fractional inputs
    self.frac_capital_goods[0] = \
      1 - self.frac_capital_hardware_rnd[0]
    self.frac_labour_goods[0] = \
      1 - self.frac_labour_hardware_rnd[0] - self.frac_labour_software_rnd[0]
    self.frac_compute_goods[0] = \
      1 - self.frac_compute_hardware_rnd[0] - self.frac_compute_software_rnd[0] \
        - self.frac_compute_training[0]

  #############################################################################

  # TASK AUTOMATION
  def automate_tasks(self, t_idx):
    # Compute largest training run
    self.biggest_training_run[t_idx] = \
      self.compute[t_idx] * self.frac_compute_training[t_idx]

    # Update index of automatable tasks
    self.automatable_tasks_goods[t_idx] = \
      np.sum(
          self.automation_training_flops_goods < \
          self.biggest_training_run[t_idx]
      ) 

    self.automatable_tasks_rnd[t_idx] = \
      np.sum(
          self.automation_training_flops_rnd < \
          self.biggest_training_run[t_idx]
      ) 
    
    # Update fraction of automated tasks
    self.frac_automatable_tasks_goods[t_idx] =  \
      (self.automatable_tasks_goods[t_idx] - 1) \
      / self.n_labour_tasks_goods # We dont count the initial compute task

    self.frac_automatable_tasks_rnd[t_idx] =  \
      (self.automatable_tasks_rnd[t_idx] - 1) \
      / self.n_labour_tasks_rnd # We dont count the initial compute task
    
    self.frac_automatable_tasks[t_idx] = \
      (self.automatable_tasks_goods[t_idx] \
      + self.automatable_tasks_rnd[t_idx] - 2)  \
      / (self.n_labour_tasks_goods
         + self.n_labour_tasks_rnd) # We dont count the initial compute task

    # Update compute to labour ratio of automatable tasks
    self.task_compute_to_labour_ratio_goods[t_idx] = \
      1 / self.automation_runtime_flops_goods
    self.task_compute_to_labour_ratio_rnd[t_idx] = \
      1 / self.automation_runtime_flops_rnd

    t_year = self.index_to_time(t_idx)

    # Update list of automation years
    if t_idx > 0 and \
       self.frac_automatable_tasks_goods[t_idx] > \
       self.frac_automatable_tasks_goods[t_idx-1]:
      self.automation_years_goods.append(t_year)
    
    if t_idx > 0 and \
       self.frac_automatable_tasks_rnd[t_idx] > \
       self.frac_automatable_tasks_rnd[t_idx-1]:
      self.automation_years_rnd.append(t_year)
    
    # Note down when AGI happens
    if t_idx > 0 and \
       self.frac_automatable_tasks[t_idx] >= 1. and \
       self.frac_automatable_tasks[t_idx-1] < 1.:
      self.agi_year = t_year
  
  ##############################################################################

  # PRODUCTION
  def production(self, t_idx):
    self.goods_production(t_idx)
    self.hardware_rnd_production(t_idx)
    self.software_rnd_production(t_idx)

  def goods_production(self, t_idx):
    # Compute goods production budgets
    self.capital_goods[t_idx] = self.capital[t_idx] * self.frac_capital_goods[t_idx]
    self.labour_goods[t_idx] = self.labour[t_idx] * self.frac_labour_goods[t_idx]
    self.compute_goods[t_idx] = self.compute[t_idx] * self.frac_compute_goods[t_idx]
    
    # Initialize task weights to match the initial economy share ratio
    if t_idx == 0:
      
      no_automation_labour_task_input_goods = np.zeros(self.n_labour_tasks_goods + 1)
      no_automation_labour_task_input_goods[1:] = self.labour_goods[0] / self.n_labour_tasks_goods
      
      no_automation_compute_task_input_goods = np.zeros(self.n_labour_tasks_goods + 1)
      no_automation_compute_task_input_goods[0] = self.compute_goods[0]

      initial_capital_to_cognitive_share_ratio_goods = \
        self.initial_capital_share_goods / self.initial_cognitive_share_goods
      initial_compute_to_labour_share_ratio_goods = \
        self.initial_compute_share_goods / self.initial_labour_share_goods

      self.capital_task_weights_goods, \
      self.labour_task_weights_goods = \
        SimulateTakeOff.adjust_task_weights(
          self.capital_goods[0],
          no_automation_labour_task_input_goods,
          no_automation_compute_task_input_goods,
          self.task_compute_to_labour_ratio_goods[0,:],
          self.capital_substitution_goods,
          self.labour_substitution_goods,
          initial_capital_to_cognitive_share_ratio_goods,
          initial_compute_to_labour_share_ratio_goods,
        )

    # Compute optimal task allocation
    self.labour_task_input_goods[t_idx, :], \
    self.compute_task_input_goods[t_idx, :] = \
      SimulateTakeOff.solve_allocation(
          self.labour_goods[t_idx], 
          self.compute_goods[t_idx],
          self.labour_task_weights_goods, 
          self.labour_substitution_goods, 
          self.task_compute_to_labour_ratio_goods[t_idx], 
          self.automatable_tasks_goods[t_idx]
          )

    self.task_input_goods[t_idx, :] = \
      self.labour_task_input_goods[t_idx, :] + \
      self.task_compute_to_labour_ratio_goods[t_idx]*self.compute_task_input_goods[t_idx, :]
    
    self.frac_tasks_automated_goods[t_idx] =\
      (np.sum(self.compute_task_input_goods[t_idx, :] > 0) - 1) \
      / self.n_labour_tasks_goods
    ## We substract 1 to account for the initial compute task
    
    # Keep track of economy share ratios
    self.capital_share_goods[t_idx],   \
    self.cognitive_share_goods[t_idx], \
    self.labour_share_goods[t_idx],    \
    self.compute_share_goods[t_idx] =  \
      SimulateTakeOff.compute_shares(
          self.capital_goods[t_idx],
          self.labour_task_input_goods[t_idx],
          self.compute_task_input_goods[t_idx],
          self.capital_task_weights_goods, \
          self.labour_task_weights_goods, 
          self.task_compute_to_labour_ratio_goods[t_idx],
          self.capital_substitution_goods,
          self.labour_substitution_goods,
      )

    # Compute output
    output = \
      SimulateTakeOff.nested_ces_production_function(
          self.capital_goods[t_idx], 
          self.task_input_goods[t_idx],
          self.capital_task_weights_goods, 
          self.labour_task_weights_goods,
          self.capital_substitution_goods,
          self.labour_substitution_goods,
          self.tfp_goods[t_idx],
          )
    
    ## Compute the ratio of output to gwp
    if t_idx == 0:
      self.output_to_gwp_factor = self.initial_gwp / output

    self.gwp[t_idx] = output * self.output_to_gwp_factor

    ## Compute how much worse is the output without automation
    # Compute optimal task allocation
    no_automation_labour_task_input_goods, \
    no_automation_compute_task_input_goods = \
      SimulateTakeOff.solve_allocation(
          self.labour_goods[t_idx], 
          self.compute_goods[t_idx],
          self.labour_task_weights_goods, 
          self.labour_substitution_goods, 
          self.task_compute_to_labour_ratio_goods[t_idx], 
          AT=1 # Only first task is automatable
          )

    no_automation_task_input_goods = \
      no_automation_labour_task_input_goods + \
      self.task_compute_to_labour_ratio_goods[t_idx]*no_automation_compute_task_input_goods
    
    no_automation_output = \
      SimulateTakeOff.nested_ces_production_function(
          self.capital_goods[t_idx], 
          no_automation_task_input_goods,
          self.capital_task_weights_goods, 
          self.labour_task_weights_goods,
          self.capital_substitution_goods,
          self.labour_substitution_goods,
          self.tfp_goods[t_idx],
          )
    
    self.automation_multiplier_goods[t_idx] = output / no_automation_output
  
  def hardware_rnd_production(self, t_idx):
    # Compute hardware rnd production budgets
    self.capital_hardware_rnd[t_idx] = self.capital[t_idx] * self.frac_capital_hardware_rnd[t_idx]
    self.labour_hardware_rnd[t_idx] = self.labour[t_idx] * self.frac_labour_hardware_rnd[t_idx]
    self.compute_hardware_rnd[t_idx] = self.compute[t_idx] * self.frac_compute_hardware_rnd[t_idx]
    
    # Initialize task weights to match the initial economy share ratio
    if t_idx == 0:
      no_automation_labour_task_input_rnd = np.zeros(self.n_labour_tasks_rnd + 1)
      no_automation_labour_task_input_rnd[1:] = self.labour_hardware_rnd[0] / self.n_labour_tasks_rnd
      
      no_automation_compute_task_input_rnd = np.zeros(self.n_labour_tasks_goods + 1)
      no_automation_compute_task_input_rnd[0] = self.compute_hardware_rnd[0]

      initial_capital_to_cognitive_share_ratio_rnd = \
        self.initial_capital_share_rnd / self.initial_cognitive_share_rnd
      initial_compute_to_labour_share_ratio_rnd = \
        self.initial_compute_share_rnd / self.initial_labour_share_rnd

      self.capital_task_weights_rnd, \
      self.labour_task_weights_rnd, =\
        SimulateTakeOff.adjust_task_weights(
          self.capital_hardware_rnd[0],
          no_automation_labour_task_input_rnd,
          no_automation_compute_task_input_rnd,
          self.task_compute_to_labour_ratio_rnd[0,:],
          self.capital_substitution_rnd,
          self.labour_substitution_rnd,
          initial_capital_to_cognitive_share_ratio_rnd,
          initial_compute_to_labour_share_ratio_rnd,
        )

    # Compute optimal task allocation
    self.labour_task_input_hardware_rnd[t_idx, :], \
    self.compute_task_input_hardware_rnd[t_idx, :] = \
      SimulateTakeOff.solve_allocation(
          self.labour_hardware_rnd[t_idx], 
          self.compute_hardware_rnd[t_idx], 
          self.labour_task_weights_rnd, 
          self.labour_substitution_rnd, 
          self.task_compute_to_labour_ratio_rnd[t_idx], 
          self.automatable_tasks_rnd[t_idx]
          )
    
    self.task_input_hardware_rnd[t_idx, :] = \
      self.labour_task_input_hardware_rnd[t_idx, :] + \
      self.task_compute_to_labour_ratio_rnd[t_idx]*self.compute_task_input_hardware_rnd[t_idx, :]
    
    # Note down fraction of tasks automated
    self.frac_tasks_automated_rnd[t_idx] =\
      (np.sum(self.compute_task_input_hardware_rnd[t_idx, :] > 0) - 1) \
      / self.n_labour_tasks_rnd
    ## We substract 1 to account for the initial compute task
    
    # Keep track of economy shares
    self.capital_share_hardware_rnd[t_idx], \
    self.cognitive_share_hardware_rnd[t_idx], \
    self.labour_share_hardware_rnd[t_idx], \
    self.compute_share_hardware_rnd[t_idx] =\
      SimulateTakeOff.compute_shares(
          self.capital_hardware_rnd[t_idx],
          self.labour_task_input_hardware_rnd[t_idx],
          self.compute_task_input_hardware_rnd[t_idx],
          self.capital_task_weights_rnd, \
          self.labour_task_weights_rnd, 
          self.task_compute_to_labour_ratio_rnd[t_idx],
          self.capital_substitution_rnd,
          self.labour_substitution_rnd,
      )

    # Compute output
    
    output_hardware = \
      SimulateTakeOff.nested_ces_production_function(
          self.capital_hardware_rnd[t_idx], 
          self.task_input_hardware_rnd[t_idx],
          self.capital_task_weights_rnd, 
          self.labour_task_weights_rnd,
          self.capital_substitution_rnd,
          self.labour_substitution_rnd,
          self.tfp_rnd[t_idx],
          )
    
    if t_idx == 0:
      self.rnd_input_to_hardware_investment_factor = \
        self.initial_rnd_input_hardware / output_hardware
    
    self.rnd_input_hardware[t_idx] = \
      output_hardware * self.rnd_input_to_hardware_investment_factor
    
    ## Compute how much worse is the hardware output without automation
    # Compute optimal task allocation
    no_automation_labour_task_input_rnd, \
    no_automation_compute_task_input_rnd = \
      SimulateTakeOff.solve_allocation(
          self.labour_hardware_rnd[t_idx], 
          self.compute_hardware_rnd[t_idx],
          self.labour_task_weights_rnd, 
          self.labour_substitution_rnd, 
          self.task_compute_to_labour_ratio_rnd[t_idx], 
          AT=1 # Only first task is automatable
          )

    no_automation_task_input_hardware_rnd = \
      no_automation_labour_task_input_rnd + \
      self.task_compute_to_labour_ratio_rnd[t_idx]*no_automation_compute_task_input_rnd
    
    no_automation_output = \
      SimulateTakeOff.nested_ces_production_function(
          self.capital_hardware_rnd[t_idx], 
          no_automation_task_input_hardware_rnd,
          self.capital_task_weights_rnd, 
          self.labour_task_weights_rnd,
          self.capital_substitution_rnd,
          self.labour_substitution_rnd,
          self.tfp_rnd[t_idx],
          )
    
    self.automation_multiplier_rnd[t_idx] = output_hardware / no_automation_output

  def software_rnd_production(self, t_idx):
    # Compute software rnd production budgets
    self.labour_software_rnd[t_idx] = self.labour[t_idx] * self.frac_labour_software_rnd[t_idx]
    self.compute_software_rnd[t_idx] = self.compute[t_idx] * self.frac_compute_software_rnd[t_idx]

    # Compute optimal task allocation
    self.labour_task_input_software_rnd[t_idx, :], \
    self.compute_task_input_software_rnd[t_idx, :] = \
      SimulateTakeOff.solve_allocation(
          self.labour_software_rnd[t_idx], 
          self.compute_software_rnd[t_idx], 
          self.labour_task_weights_rnd, 
          self.labour_substitution_rnd, 
          self.task_compute_to_labour_ratio_rnd[t_idx], 
          self.automatable_tasks_rnd[t_idx]
          )
    
    self.task_input_software_rnd[t_idx, :] = \
      self.labour_task_input_software_rnd[t_idx, :]  \
      + self.task_compute_to_labour_ratio_rnd[t_idx] \
      * self.compute_task_input_software_rnd[t_idx, :]
    
    # Compute output
    output_software = \
      SimulateTakeOff.ces_production_function(
          self.task_input_software_rnd[t_idx, :], 
          self.labour_task_weights_rnd,
          self.labour_substitution_rnd,
          self.tfp_rnd[t_idx]
          )
    
    if t_idx == 0:
      self.rnd_input_to_software_investment_factor = \
        self.initial_rnd_input_software / output_software

    self.rnd_input_software[t_idx] =\
      output_software * self.rnd_input_to_software_investment_factor
  
  #############################################################################

  ## REINVEST OUTPUT IN INPUTS
  def reinvest_output_in_inputs(self, t_idx):
    self.update_rnd_state(t_idx)
    self.allocate_fractional_inputs(t_idx)
    self.calculate_total_inputs(t_idx)
  
  def update_rnd_state(self, t_idx):
    
    def _update_rnd(
        current_performance, 
        initial_performance,
        research_input, 
        cumulative_adjusted_input, 
        returns,
        performance_ceiling
        ):
      adjusted_input =\
        research_input**self.rnd_parallelization_penalty
      new_cumulative_adjusted_input =\
        cumulative_adjusted_input + adjusted_input*self.t_step
      growth_in_cumulative_inputs =\
        new_cumulative_adjusted_input / \
        cumulative_adjusted_input
      ceiling_penalty = \
        (np.log10(performance_ceiling) - np.log10(current_performance)) /\
        (np.log10(performance_ceiling) - np.log10(initial_performance))
      performance_growth_rate = \
        growth_in_cumulative_inputs**(returns * ceiling_penalty)
      new_performance = \
        min(
            current_performance * performance_growth_rate,
            performance_ceiling
        )

      return new_performance, new_cumulative_adjusted_input

    # Hardware
    self.hardware_performance[t_idx],\
    self.cumulative_rnd_input_hardware[t_idx] = \
      _update_rnd(
        self.hardware_performance[t_idx-1], 
        self.initial_hardware_performance,
        self.rnd_input_hardware[t_idx-1], 
        self.cumulative_rnd_input_hardware[t_idx-1], 
        self.hardware_returns,
        self.hardware_performance_ceiling
        )

    # Software
    self.software[t_idx],\
    self.cumulative_rnd_input_software[t_idx] = \
      _update_rnd(
        self.software[t_idx-1], 
        self.initial_software,
        self.rnd_input_software[t_idx-1], 
        self.cumulative_rnd_input_software[t_idx-1], 
        self.software_returns,
        self.software_ceiling
        )
  
  def allocate_fractional_inputs(self, t_idx):

    # Rampup
    self.rampup[t_idx] = \
      self.frac_tasks_automated_goods[t_idx-1] >= self.rampup_trigger
    
    if self.rampup[t_idx] and not self.rampup[t_idx-1]:
      t_year = self.index_to_time(t_idx)
      self.rampup_start = t_year
    
    if self.rampup[t_idx] and \
    self.frac_tasks_automated_goods[t_idx-1] >= 0.3 and \
    not self.frac_tasks_automated_goods[t_idx-2] >= 0.3:
      self.rampup_mid = self.index_to_time(t_idx)
      

    def update_frac_input(current_frac, growth_rate, growth_rate_rampup, max_frac):
      frac = current_frac\
      * np.exp(self.t_step * 
               (growth_rate_rampup if self.rampup[t_idx] else growth_rate)
               )
      frac = min(frac, max_frac)
      return frac
    
    # Hacky loop
    frac_metrics = [
      'frac_gwp_compute', 
      'frac_capital_hardware_rnd',
      'frac_labour_hardware_rnd',
      'frac_compute_hardware_rnd',
      'frac_labour_software_rnd',
      'frac_compute_software_rnd',
      'frac_compute_training'
      ]
    
    for frac_metric in frac_metrics:
      getattr(self, frac_metric)[t_idx] =\
        update_frac_input(
          getattr(self,frac_metric)[t_idx-1], 
          getattr(self,f'{frac_metric}_growth'),
          getattr(self,f'{frac_metric}_growth_rampup'),
          getattr(self,f'{frac_metric}_ceiling'),
          )

    # Goods production fractional inputs
    self.frac_capital_goods[t_idx] = \
      1 - self.frac_capital_hardware_rnd[t_idx]
    self.frac_labour_goods[t_idx] = \
      1 - self.frac_labour_hardware_rnd[t_idx] - self.frac_labour_software_rnd[t_idx]
    self.frac_compute_goods[t_idx] = \
      1 - self.frac_compute_hardware_rnd[t_idx] - self.frac_compute_software_rnd[t_idx] - self.frac_compute_training[t_idx]
  
  def calculate_total_inputs(self, t_idx):
    
    # Compute  
    self.compute_investment[t_idx] = \
      self.gwp[t_idx-1] * self.frac_gwp_compute[t_idx] * self.t_step
    
    if t_idx-self.hardware_delay_idx >= 0:
      buyable_hardware_performance = \
        self.hardware_performance[t_idx-self.hardware_delay_idx]
    else:
      buyable_hardware_performance = \
        self.hardware_performance[0] * (self.hardware_performance[1] \
        / self.hardware_performance[0])**(t_idx-self.hardware_delay_idx)
    
    new_hardware = self.compute_investment[t_idx] * buyable_hardware_performance

    self.hardware[t_idx] = \
      self.hardware[t_idx-1] \
      * ((1.-self.compute_depreciation)**self.t_step) \
      + new_hardware
    self.compute[t_idx] = self.hardware[t_idx] * self.software[t_idx]
    
    # Non-compute inputs
    capital_investment = \
      self.gwp[t_idx-1] * self.investment_rate * self.t_step
    self.capital[t_idx] =\
      self.capital[t_idx-1] + capital_investment
      
    self.labour[t_idx] =\
      self.labour[t_idx-1] * np.exp(self.labour_growth * self.t_step)
    
    # Total factor production
    self.tfp_goods[t_idx] = self.tfp_goods[t_idx-1] * np.exp(self.tfp_growth * self.t_step)
    self.tfp_rnd[t_idx] = self.tfp_rnd[t_idx-1] * np.exp(self.tfp_growth * self.t_step)

  ###########################################################################

  ## AUXILIARY FUNCTIONS

  @staticmethod
  def quantiles_from_gap(top, gap):
      unit = gap**(1/7)
      quantiles = {
          1.0: top,
          0.5: top/(unit**4),
          0.2: top/(unit**7),
          0.1: top/(unit**8.5),
          0.05: top/(unit**9.5),
          0.0: top/(unit**10.5),
          }
      return quantiles

  @staticmethod
  def process_quantiles(quantile_dict, n_items):
    """ Input is a dictionary of quantiles {q1:v1, ..., qn:vn}
        Returns a numpy array of size n_items whose quantiles match the dictionary
        The rest of entries are geometrically interpolated
    """
    values = []
    
    for q in np.linspace(0, 1, n_items):
      
      prev_quantile = \
        np.max([quantile for quantile in quantile_dict.keys() if q >= quantile])
      next_quantile = \
        np.min([quantile for quantile in quantile_dict.keys() if q <= quantile])
      prev_value = quantile_dict[prev_quantile]
      next_value = quantile_dict[next_quantile]

      if prev_quantile == next_quantile:
        value = prev_value
      else:
        value = prev_value*((next_value/prev_value)**((q-prev_quantile)/(next_quantile-prev_quantile)))

      values.append(value)
    values = np.array(values)

    # Check that the quantiles of the result approximately match the input dict
    for quantile, expected_value in quantile_dict.items():
      actual_value = np.quantile(values, quantile)
      assert np.abs(np.log10(actual_value / expected_value)) < 1
    
    return values
  
  @staticmethod
  def solve_allocation(L, C, β, ρ, η, AT):
    """
    Solve the input allocation problem for
    L = Labour budget
    C = Compute budget
    β = task weights 
    ρ = labour / capital substitution parameter
    η = compute / labour substitution ratio
    AT = number of automatable tasks

    See description of solution at the end of the notebook
    We assume that 
      * η is monotonically decreasing on its index.
      * task 0 is automatable
    """

    # Check assumptions
    assert np.all(np.diff(η) <= 0.)
    assert AT > 0

    # Preprocessing parameters
    N = len(β)
    σ = 1. / (1.-ρ)

    # Iterate over critical indices
    for I in range(AT):
      # Initialize
      labour_input_task = np.zeros(N)
      compute_input_task = np.zeros(N)

      ## Equation 20
      A = η[I]**σ * np.sum(β[I:]**σ)
      B = np.sum(β[:I]**σ * η[:I]**(σ-1))
      compute_input_task[I] =\
        (C*A - L*B) / (A + η[I]*B)
      
      ## Equation 18
      labour_input_task[I] =\
        (L + η[I]*compute_input_task[I]) \
        * (β[I]**σ / np.sum(β[I:]**σ)) \
        - η[I]*compute_input_task[I]
      
      ## Equation 17
      labour_input_task[I+1:] =\
        (L + η[I]*compute_input_task[I]) \
        * (β[I+1:]**σ / np.sum(β[I:]**σ))
      
      ## Equation 14
      Z = np.sum(β[:I+1]**σ * η[:I+1]**(σ-1))
      compute_input_task[:I] =\
        (C + labour_input_task[I]/η[I]) \
        * β[:I]**σ * η[:I]**(σ-1) / Z
      
      if labour_input_task[I] >= 0: break
    
    else:
      # The critical index is the last one
      I = AT-1

      # Initialize
      labour_input_task = np.zeros(N) 
      compute_input_task = np.zeros(N)

      ## Equations 14 & 15
      Z = np.sum(β[:I+1]**σ * η[:I+1]**(σ-1))
      compute_input_task[:I+1] =\
        C * β[:I+1]**σ * η[:I+1]**(σ-1) / Z
      
      ## We assume LI = 0
      labour_input_task[I] = 0

      ## Equation 22
      labour_input_task[I+1:] =\
        L * (β[I+1:]**σ / np.sum(β[I+1:]**σ))
    
    # Fix rounding error
    if np.all(labour_input_task==0): 
      labour_input_task[-1] = L

    return labour_input_task, compute_input_task
  
  @staticmethod
  def odds_to_probs(o):
    """ Stable implementation of conversion between odds and probs
    """
    # For small outputs the odds are approx equal to the probs
    if o < 1e-10:
      p = o
      p_not = 1.-p
    # For big outputs the odds can be approx like this
    elif o > 1e10:
      p = 1 - 1/o
      p_not = 1/o
    else:
      p = 1/(1+1/o)
      p_not = 1.-p
    
    assert 0. <= p <= 1., f"o = {o}, p = {p}"
    assert 0. <= p_not <= 1., f"o = {o}, p_not = {p_not}"
    assert np.abs(p/p_not / o - 1.) < 1e-6
    return p, p_not
  
  @staticmethod
  def adjust_task_weights(
      capital,
      labour_task_input,
      compute_task_input,
      task_compute_to_labour_ratio,
      capital_substitution,
      labour_substitution,
      capital_to_cognitive_share_ratio,
      compute_to_labour_share_ratio,
    ):
    """ Computes the task weights that would result in a 
        target capital_to_labour_share_ratio and compute_to_labour_share_ratio of the economy
    """
    
    # Compute inner task weights

    task_input = \
      labour_task_input + \
      task_compute_to_labour_ratio*compute_task_input

    labour_share = np.sum(
      labour_task_input * \
      task_input**(labour_substitution-1)
    )

    compute_share = np.sum(
      task_compute_to_labour_ratio * \
      compute_task_input * \
      task_input**(labour_substitution-1)
    )

    assert np.all(compute_task_input[1:] == 0.)
    assert labour_task_input[0] == 0.
    compute_to_labour_task_weight_ratio = \
      compute_to_labour_share_ratio * labour_share / compute_share
    
    compute_task_weight, labour_task_weight = \
      SimulateTakeOff.odds_to_probs(compute_to_labour_task_weight_ratio)
    
    n_labour_tasks = len(labour_task_input)-1
    inner_task_weights = \
      np.array([compute_task_weight] + 
               [labour_task_weight for i in range(n_labour_tasks)]
               )

    # Compute outer task weights
    cognitive_input = \
      SimulateTakeOff.ces_production_function(
          task_input,
          inner_task_weights, 
          labour_substitution
        )
    
    capital_share = capital**capital_substitution
    cognitive_share = cognitive_input**capital_substitution

    capital_to_cognitive_task_weight_ratio = \
      capital_to_cognitive_share_ratio * cognitive_share / capital_share

    capital_task_weight, cognitive_task_weight = \
      SimulateTakeOff.odds_to_probs(capital_to_cognitive_task_weight_ratio)
    outer_task_weights = np.array([capital_task_weight, cognitive_task_weight])

    return outer_task_weights, inner_task_weights
  
  @staticmethod
  def compute_shares(
          capital,
          labour_task_input,
          compute_task_input,
          capital_task_weights,
          labour_task_weights, 
          task_compute_to_labour_ratio,
          capital_substitution,
          labour_substitution,
      ):

    # Compute inputs
    task_input = \
      labour_task_input + \
      compute_task_input*task_compute_to_labour_ratio
    
    cognitive_input = \
      SimulateTakeOff.ces_production_function(
          task_input,
          labour_task_weights, 
          labour_substitution
        )

    # Compute capital and cognitive shares
    capital_task_weight = capital_task_weights[0]
    cognitive_task_weight = capital_task_weights[1]

    capital_share = capital_task_weight*capital**capital_substitution
    cognitive_share = cognitive_task_weight*cognitive_input**capital_substitution

    sum = capital_share + cognitive_share
    capital_share /= sum
    cognitive_share /=sum

    # Compute labour and compute shares
    labour_share = np.sum(labour_task_weights \
                   * labour_task_input \
                   * task_input**(labour_substitution-1))
    
    compute_share = np.sum(labour_task_weights \
                    * compute_task_input*task_compute_to_labour_ratio \
                    * task_input**(labour_substitution-1))
    
    sum = labour_share + compute_share
    labour_share /= sum
    compute_share /= sum
    
    labour_share *= cognitive_share
    compute_share *= cognitive_share
    
    return capital_share, cognitive_share, labour_share, compute_share
    
  ces_production_function = lambda inputs, alphas, rho, tfp=1. : \
    tfp*np.sum(alphas*(inputs**rho) / alphas.sum())**(1./rho)
  
  nested_ces_production_function =        \
    lambda capital, cognitive_inputs,     \
           outer_weights, inner_weights,  \
           outer_rho, inner_rho,          \
           tfp=1 :                        \
    tfp*SimulateTakeOff.ces_production_function(
       np.array([capital, 
        SimulateTakeOff.ces_production_function(
            cognitive_inputs,
            inner_weights,
            inner_rho
        )]),
      outer_weights,
      outer_rho
      )
  
  def time_to_index(self, t):
    return round((t - self.t_start) / self.t_step)
  
  def index_to_time(self, idx):
    return self.t_start + idx * self.t_step

  ###########################################################################
  
  def _length_between_thresholds(
        self,
        series1,
        series2,
    ):
    """ Utility function to measure the amount of time between 
        two thresholds being crossed
    """
    if not np.any(series1) or not np.any(series2):
      return self.t_end - self.t_start
    idx1 = np.argmax(series1)
    idx2 = np.argmax(series2)
    return (idx2 - idx1) * self.t_step

  takeoff_metrics = [
    'cog_output_multiplier',
    'billion_agis', 
    'full_automation', 
    'rampup_to_agi',
    'gwp_growth',
    'combined'
    ]

  def compute_metrics(self):
    """ Computes indicator metrics measuring length of AI takeoff 
    """
    # Initialize takeoff metrics dict
    self.takeoff_metrics = {}

    # GWP doubling times
    self.compute_doubling_times()

    # Years from “total cognitive output is 2X human cognitive output” to 
    # “total cognitive output is 10X human cognitive output”
    self.takeoff_metrics["cog_output_multiplier"] = \
      self._length_between_thresholds(
          self.automation_multiplier_rnd > 2,
          self.automation_multiplier_rnd > 10,
      )
    
    # Time from AI that automates 10% of cognitive tasks to when 
    # we have enough compute to run 10 billion AGIs
    
    self.frac_automated_tasks = \
      self.frac_tasks_automated_goods * self.n_labour_tasks_goods \
    + self.frac_tasks_automated_rnd * self.n_labour_tasks_rnd \
    / ( self.n_labour_tasks_goods + self.n_labour_tasks_rnd )
    
    ten_billion_agi_compute = \
      max(np.max(self.automation_runtime_flops_goods), 
          np.max(self.automation_runtime_flops_rnd))*1e10
    full_automation_flops = \
      max(np.max(self.automation_training_flops_goods), 
          np.max(self.automation_training_flops_rnd))
    self.takeoff_metrics["billion_agis"] = \
      self._length_between_thresholds(
          self.frac_automated_tasks > 0.1,
          (self.compute >= ten_billion_agi_compute) &\
          (self.biggest_training_run >= full_automation_flops),
      )

    # Time from AI that can perform 50% of tasks to AI that can perform 100%.

    self.takeoff_metrics["full_automation"] = \
      self._length_between_thresholds(
          self.frac_automatable_tasks > 0.5,
          self.frac_automatable_tasks >= 1.,
      )

    # Time from rampup to full automation
    self.takeoff_metrics["rampup_to_agi"] = \
      self.agi_year - self.rampup_start \
      if self.agi_year and self.rampup_start \
      else self.t_end - self.t_start 
    self.takeoff_metrics["rampup_to_agi"] = max(self.takeoff_metrics["rampup_to_agi"], 0.0)
    ## BUG: IF RAMPUP AND AGI HAPPEN AT THE SAME TIME THEN THIS WOULD BE NEGATIVE!
    
    ## Combined metric
    self.takeoff_metrics["combined"] = \
      np.mean([self.takeoff_metrics[metric] for metric in self.takeoff_metrics])
    
    # Time from 5% GWP growth to 15% GWP growth
    delta = int(1 / self.t_step)
    gwp_growth = np.log(self.gwp[delta:self.t_idx] / self.gwp[:self.t_idx-delta])
    self.takeoff_metrics["gwp_growth"] = \
      self._length_between_thresholds(
          gwp_growth > 0.05,
          gwp_growth > 0.15,
      )

  def compute_doubling_times(self):
    self.doubling_times = [self.t_step / np.log2(self.gwp[1]/self.gwp[0])]
    if self.rampup_start is None: return
    reference_idx = self.time_to_index(self.rampup_start)
    for idx in range(reference_idx, self.n_timesteps):
      if self.gwp[idx] > 2*self.gwp[reference_idx]:
        self.doubling_times.append(self.index_to_time(idx) - self.index_to_time(reference_idx))
        reference_idx = idx
    
    # Round doubling times
    self.doubling_times = [round(dt, 2) for dt in self.doubling_times]
    
    # We are only interested in the first five doubling times
    self.doubling_times = self.doubling_times[:5]

  ###########################################################################

  ## VISUALIZATION ##

  def plot(self, metric, plot_growth = False, new_figure=True, line_color='black', crop_after_agi = True):
    """ Plot a metric over time.
        Eg gwp, compute, capital, labour, hardware_efficiency, software, hardware
    """
    x = self.timesteps
    y = getattr(self, metric)
    
    if crop_after_agi:
      idx_end = min(self.time_to_index(self.agi_year+5), self.t_idx) \
                if self.agi_year is not None and crop_after_agi else self.t_idx
      x = x[:idx_end]
      y = y[:idx_end]
      
    if plot_growth:
      # Plot annual growth
      delta = int(1 / self.t_step)
      x = x[delta:]
      y = np.log(y[delta:] / y[:-delta])
    
    if new_figure:
      plt.figure(figsize=(14, 8), dpi=80);
    #if not plot_growth:
    plt.yscale('log')
    plt.plot(x, y, color = line_color);
    if plot_growth:
      plt.title(f"{metric} growth over time");
    else:
      plt.title(f"{metric} over time");

    self._plot_vlines(line_color = line_color)
  
  def _plot_vlines(self, line_color = 'black'):
    
    if self.rampup_start:
      plt.axvline(self.rampup_start, 
                linestyle='dotted',
                color=line_color,
                label='Start of ramp-up');
                
    if self.rampup_mid:
      plt.axvline(self.rampup_mid, 
                linestyle='-.',
                color=line_color,
                label='Middle of ramp-up');
                
    if self.agi_year:
      plt.axvline(self.agi_year, 
                linestyle='dashed',
                color=line_color,
                label='Full automation');

      
  
  def plot_compute_decomposition(self, new_figure=True, crop_after_agi = True):
    """ Show the growth of the factors that drive compute
    """
    
    if new_figure:
      plt.figure(figsize=(14, 8), dpi=80);

    start_idx = 0
    reference_idx = self.time_to_index(self.rampup_start) if self.rampup_start is not None else 0
    end_idx = min(self.time_to_index(self.agi_year+5), self.t_idx) if self.agi_year is not None and crop_after_agi else self.t_idx
    
    plt.plot(self.timesteps[start_idx:end_idx], self.compute_investment[start_idx:end_idx]/self.compute_investment[reference_idx], label='$ on FLOP globally', color = 'blue')
    plt.plot(self.timesteps[start_idx:end_idx], self.hardware_performance[start_idx:end_idx]/self.hardware_performance[reference_idx], label='Hardware (FLOP/$)', color = 'orange')
    plt.plot(self.timesteps[start_idx:end_idx], self.software[start_idx:end_idx]/self.software[reference_idx], label='Software (2020-FLOP per FLOP)', color = 'green')
    plt.plot(self.timesteps[start_idx:end_idx], self.frac_compute_training[start_idx:end_idx]/self.frac_compute_training[reference_idx], label='Fraction global FLOP on training', color = 'red')
    
    plt.yscale('log')

    self._plot_vlines()

    # Plot horizontal lines showing each order of magnitude
    high = max(
            self.frac_compute_training[end_idx-1]/self.frac_compute_training[reference_idx],
            self.software[end_idx-1]/self.software[reference_idx],
            self.hardware_performance[end_idx-1]/self.hardware_performance[reference_idx],
            self.compute_investment[end_idx-1]/self.compute_investment[reference_idx]
            )
    low = min(
            self.frac_compute_training[start_idx]/self.frac_compute_training[reference_idx],
            self.software[start_idx]/self.software[reference_idx],
            self.hardware_performance[start_idx]/self.hardware_performance[reference_idx],
            self.compute_investment[start_idx]/self.compute_investment[reference_idx]
            )
    for oom in range(math.floor(np.log10(low)), math.ceil(np.log10(high))):
      plt.axhline(10**oom, linestyle='dotted', color='black',)
    
    if new_figure:
      plt.title(f"Compute increase decomposition");
      plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
      plt.tight_layout();

  def plot_compute_decomposition_bioanchors_style(self, new_figure = True):
    """ Show the growth of the factors that drive compute in the style of the Bio Anchors report
    """
    
    if new_figure:
      plt.figure(figsize=(14, 8), dpi=80);

    reference_idx = 0

    training_investment = self.compute_investment * self.frac_compute_training
    
    plt.plot(self.timesteps, training_investment/training_investment[reference_idx], label = 'Training compute investment', color = 'purple')
    plt.plot(self.timesteps, self.hardware_performance/self.hardware_performance[reference_idx], label = 'Hardware performance', color = 'orange')
    plt.plot(self.timesteps, self.software/self.software[reference_idx], label = 'Software', color = 'green')
    
    plt.yscale('log')

    utils.draw_oom_lines()

    if new_figure:
      plt.title(f"Compute increase decomposition")
      plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
      plt.tight_layout()

  def get_summary_table(self):
    summary_table = []
    
    prerampup = np.mean([self.t_start, self.rampup_start]) if self.rampup_start is not None else None
    raw_metrics = ['biggest_training_run', 'frac_tasks_automated_goods', 'frac_tasks_automated_rnd']
    doubling_time_metrics = ['hardware_performance', 'software', 'compute_investment', 'frac_compute_training', 'gwp', 'capital', 'labour', 'tfp_rnd', "rnd_input_software", "cumulative_rnd_input_software"]
    
    for period, t in {'prerampup' : prerampup , 
                      'rampup_start': self.rampup_start, 
                      'mid rampup': self.rampup_mid, 
                      'agi': self.agi_year}.items() :
      
      if t is None:
        summary_row = {
          'period' : period,
          'year' : np.nan,
        }
      
        for raw_metric in raw_metrics:
          summary_row[f"{raw_metric}"] = np.nan
      
        for doubling_time_metric in doubling_time_metrics:
          summary_row[f"{doubling_time_metric} growth rate"] = np.nan
          summary_row[f"{doubling_time_metric} doubling time"] = np.nan
          
        summary_table.append(summary_row)
        continue
        
      idx = self.time_to_index(t)
      t = self.index_to_time(idx)
      t_end = t + 1
      idx_end = self.time_to_index(t_end)
      
      summary_row = {
        'period' : period,
        'year' : t,
      }
      
      # Auxiliary functions
      dt = lambda s : 1 / np.log2(s[idx_end]/s[idx]) \
                      if np.log2(s[idx_end]/s[idx]) != 0 else np.nan
      
      gr = lambda s : np.log(s[idx_end] / s[idx])
      
      for raw_metric in raw_metrics:
        summary_row[f"{raw_metric}"] = getattr(self, raw_metric)[idx]
      
      for doubling_time_metric in doubling_time_metrics:
        summary_row[f"{doubling_time_metric} growth rate"] = gr(getattr(self, doubling_time_metric))
        summary_row[f"{doubling_time_metric} doubling time"] = dt(getattr(self, doubling_time_metric))

      summary_table.append(summary_row)

    summary_table = pd.DataFrame(summary_table)

    return summary_table
  
  def display_summary_table(self):
    display(self.get_summary_table())

  def get_takeoff_metrics(self):
    metrics_df = pd.DataFrame(self.takeoff_metrics, index = [0])
    metrics_df["doubling times"] = repr(self.doubling_times[:4])
    return metrics_df
  
  def display_takeoff_metrics(self):
    display(self.get_takeoff_metrics())

    
if __name__ == "__main__":
  # Retrieve parameter estimates from spreadsheet
  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  # Run model
  model = SimulateTakeOff(**best_guess_parameters, t_step=1)
  model.run_simulation()

  # Plot things
  model.plot('gwp')
  model.plot('gwp', plot_growth=True)
  model.plot_compute_decomposition()
  plt.show()
  model.display_summary_table()
  model.display_takeoff_metrics()
