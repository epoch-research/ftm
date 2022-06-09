#@title Test suite

import unittest

import numpy as np
from opmodel import *

import opmodel

print(opmodel.SimulateTakeOff)

class TestSimulateTakeoff(unittest.TestCase):
  
  def setUp(self):
    pass
  
  def test_values(self):
    pass
  
  def test_basic_checks(self):
    # Create model
    model = SimulateTakeOff()

    # Run simulation
    model.run_simulation()
    
    # Non-negativity of allocations
    self.assertTrue(np.all(model.task_input_goods >= 0.))
    self.assertTrue(np.all(model.labour_task_input_goods >= 0.))
    self.assertTrue(np.all(model.compute_task_input_goods >= 0.))
    
    self.assertTrue(np.all(model.task_input_rnd >= 0.))
    self.assertTrue(np.all(model.labour_task_input_rnd >= 0.))
    self.assertTrue(np.all(model.compute_task_input_rnd >= 0.))

    # Initial gwp matches target
    self.assertAlmostEqual(
        model.gwp[0], 
        model.initial_gwp
    )

    # Initial capital growth matches initial gwp growth
    initial_capital_growth =\
      np.log(model.capital[1]/model.capital[0]) / model.t_step
    self.assertAlmostEqual(initial_capital_growth, model.initial_gwp_growth, places=2)

    # Initial capital to cognitive share matches target
    self.assertAlmostEqual(
      model.capital_share_goods[0] / model.cognitive_share_goods[0],
      model.initial_capital_to_cognitive_share_ratio_goods,
      places=2
    )
    self.assertAlmostEqual(
      model.capital_share_hardware_rnd[0] / model.cognitive_share_hardware_rnd[0],
      model.initial_capital_to_cognitive_share_ratio_hardware_rnd,
      places=2
    )

    # Initial compute to labour share matches target
    self.assertAlmostEqual(
      model.compute_share_goods[0] / model.labour_share_goods[0],
      model.initial_compute_to_labour_share_ratio_goods,
      places=2
    )
    self.assertAlmostEqual(
      model.compute_share_hardware_rnd[0] / model.labour_share_hardware_rnd[0],
      model.initial_compute_to_labour_share_ratio_hardware_rnd,
      places=2
    )

    # Initial biggest training run matches target
    self.assertAlmostEqual(
        model.biggest_training_run[0], 
        model.initial_biggest_training_run
      )

    # Rising inputs
    self.assertTrue(np.all(np.diff(model.capital) >= 0.))
    self.assertTrue(np.all(np.diff(model.labour) >= 0.))
    self.assertTrue(np.all(np.diff(model.compute) >= 0.))

    # Research improving over time
    self.assertTrue(np.all(np.diff(model.hardware_performance) >= 0.))
    self.assertTrue(np.all(np.diff(model.software) >= 0.))

    # Rising outputs
    self.assertTrue(np.all(np.diff(model.gwp) >= 0.))
    self.assertTrue(np.all(np.diff(model.rnd_input_hardware) >= 0.))
    self.assertTrue(np.all(np.diff(model.rnd_input_software) >= 0.))
  
  def test_different_timesteps(self):
    # Create models
    model1 = SimulateTakeOff(t_step=0.5)
    model2 = SimulateTakeOff(t_step=1)

    # Run simulations
    model1.run_simulation()
    model2.run_simulation()

    # Compare some metrics
    self.assertAlmostEqual(
        np.log10(model1.hardware_performance[2]),
        np.log10(model2.hardware_performance[1]),
        places=2
    )

    self.assertAlmostEqual(
        np.log10(model1.hardware[2]),
        np.log10(model2.hardware[1]),
        places=2
    )

    self.assertAlmostEqual(
        model1.software[2],
        model2.software[1],
        places=2,
    )

    self.assertAlmostEqual(
        np.log10(model1.compute[2]),
        np.log10(model2.compute[1]),
        places=2,
    )

    self.assertAlmostEqual(
        np.log10(model1.capital[2]),
        np.log10(model2.capital[1]),
        places=2,
    )

    self.assertAlmostEqual(
        model1.labour[2],
        model2.labour[1],
        places=2,
    )
    
  
  # def test_rnd_progress(self):
  #   model = SimulateTakeOff()

  #   t_idx = 1
    
  #   model.rnd_parallelization_penalty = 0.58
    
  #   model.hardware_returns = 5.1
  #   model.hardware_performance_ceiling = 1e30
  #   model.initial_hardware_performance = 1e19

  #   model.hardware_performance[t_idx-1] = 10**19.08060536
  #   model.rnd_input_hardware[t_idx - 1] = 1.67E+11**(1/model.rnd_parallelization_penalty)
  #   model.cumulative_rnd_input_hardware[t_idx-1] = 4.67E+12

  #   model.software_returns = 3.4
  #   model.software_ceiling = 1e7
  #   model.initial_software = 1

  #   model.software[t_idx-1] = 1.459772
  #   model.rnd_input_software[t_idx - 1] = 1.65E+10**(1/model.rnd_parallelization_penalty)
  #   model.cumulative_rnd_input_software[t_idx-1] = 1.69E+11

  #   model.update_rnd_state(t_idx)

  #   self.assertGreater(model.hardware_performance[t_idx], model.hardware_performance[t_idx-1])
  #   self.assertGreater(model.software[t_idx], model.software[t_idx-1])
    
  #   self.assertAlmostEqual(np.log10(model.hardware_performance[t_idx]), 19.16076397)
  #   self.assertAlmostEqual(model.software[t_idx], 2.07)


class TestAllocationFunction(unittest.TestCase):
  def test_allocation_function_simple_input(self):
    labour = 1.
    compute = 1.
    task_shares = np.array([1.,2.])
    task_shares = task_shares / np.sum(task_shares) # normalize
    substitution = -0.5
    task_compute_to_labour_ratio = np.array([1.,1.])
    automatable_tasks = 1

    labour_task_input, compute_task_input = \
      SimulateTakeOff.solve_allocation(
          labour,
          compute,
          task_shares,
          substitution,
          task_compute_to_labour_ratio,
          automatable_tasks,
          )

    # Non-negativity of solution
    self.assertTrue(np.all(labour_task_input >= 0.))
    self.assertTrue(np.all(compute_task_input >= 0.))

    # Budget constraints met
    self.assertAlmostEqual(np.sum(labour_task_input), labour)
    self.assertAlmostEqual(np.sum(compute_task_input), compute)

    # Automation constraints met
    self.assertTrue(np.all(np.abs(compute_task_input[automatable_tasks:]) < 0.01))

  def test_allocation_function_complex_input(self):
    
    labour = 1.
    compute = 1.
    task_shares = np.array([1.,1.,1.,1.])
    task_shares = task_shares / np.sum(task_shares) # normalize
    substitution = -0.5
    task_compute_to_labour_ratio = np.array([1.,1.,1.,1.])
    automatable_tasks = 2

    labour_task_input, compute_task_input = \
      SimulateTakeOff.solve_allocation(
          labour,
          compute,
          task_shares,
          substitution,
          task_compute_to_labour_ratio,
          automatable_tasks,
          )

    # Non-negativity of solution
    self.assertTrue(np.all(labour_task_input >= 0.))
    self.assertTrue(np.all(compute_task_input >= 0.))

    # Budget constraints met
    self.assertAlmostEqual(np.sum(labour_task_input), labour)
    self.assertAlmostEqual(np.sum(compute_task_input), compute)

    # Automation constraints met
    self.assertTrue(np.all(np.abs(compute_task_input[automatable_tasks:]) < 0.01))

    # Check solution
    expected_compute = np.array([0.5, 0.5, 0., 0.])
    expected_labour = np.array([0., 0., 0.5, 0.5])
    self.assertTrue(np.allclose(labour_task_input, expected_labour), f"received {labour_task_input}, expected {expected_labour}")
    self.assertAlmostEqual(np.max(np.abs(compute_task_input - expected_compute)), 0.)

  def test_ultra_complex_allocation_output(self):

    labour = 5715322978.400196
    compute = 2.1634814945442394e+33 
    task_shares = np.array([0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067,0.02488773520691067])
    substitution = -0.5 
    task_compute_to_labour_ratio = np.array([1.00000000e+00,1.00000000e-07 ,1.00000000e-14 ,3.16227766e-15,1.00000000e-15 ,5.62341325e-16 ,3.16227766e-16 ,1.77827941e-16,1.00000000e-16 ,6.21689065e-17 ,3.86497294e-17 ,2.40281141e-17,1.49380158e-17 ,9.28680109e-18 ,5.77350269e-18 ,3.58932349e-18,2.23144317e-18 ,1.38726382e-18 ,8.62446746e-19 ,5.36173711e-19, 3.33333333e-19 ,2.79727271e-19 ,2.34742038e-19 ,1.96991249e-19, 1.65311473e-19 ,1.38726382e-19 ,1.16416656e-19 ,9.76947407e-20, 8.19836495e-20 ,6.87991876e-20 ,5.77350269e-20 ,4.84501845e-20, 4.06585136e-20 ,3.41198852e-20 ,2.86327871e-20 ,2.40281141e-20, 2.01639564e-20 ,1.69212254e-20 ,1.41999846e-20 ,1.19163688e-20]) 
    automatable_tasks = 20

    labour_task_input, compute_task_input = \
      SimulateTakeOff.solve_allocation(
          labour,
          compute,
          task_shares,
          substitution,
          task_compute_to_labour_ratio,
          automatable_tasks,
          )

    # Non-negativity of solution
    self.assertTrue(np.all(labour_task_input >= 0.))
    self.assertTrue(np.all(compute_task_input >= 0.))

    # Budget constraints met
    self.assertAlmostEqual(np.sum(labour_task_input), labour, places=4)
    self.assertAlmostEqual(np.sum(compute_task_input), compute, delta=0.0001*compute)

    # Automation constraints met
    self.assertTrue(np.all(np.abs(compute_task_input[automatable_tasks:]) < 0.01))
  
  def test_allocation_after_all_automated(self):
    labour = 6008353853.645853 
    compute = 9.364651787582756e+68
    task_shares = np.array(
      [0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876])
    substitution = -0.5 
    task_compute_to_labour_ratio = np.array([1.00000000e+00,1.00000000e-07 ,1.00000000e-14 ,3.16227766e-15,1.00000000e-15 ,5.62341325e-16 ,3.16227766e-16 ,1.77827941e-16,1.00000000e-16 ,6.21689065e-17 ,3.86497294e-17 ,2.40281141e-17,1.49380158e-17 ,9.28680109e-18 ,5.77350269e-18 ,3.58932349e-18,2.23144317e-18 ,1.38726382e-18 ,8.62446746e-19 ,5.36173711e-19, 3.33333333e-19 ,2.79727271e-19 ,2.34742038e-19 ,1.96991249e-19, 1.65311473e-19 ,1.38726382e-19 ,1.16416656e-19 ,9.76947407e-20, 8.19836495e-20 ,6.87991876e-20 ,5.77350269e-20 ,4.84501845e-20, 4.06585136e-20 ,3.41198852e-20 ,2.86327871e-20 ,2.40281141e-20, 2.01639564e-20 ,1.69212254e-20 ,1.41999846e-20 ,1.19163688e-20]) 
    automatable_tasks = 40
    
    labour_task_input, compute_task_input = \
      SimulateTakeOff.solve_allocation(
          labour,
          compute,
          task_shares,
          substitution,
          task_compute_to_labour_ratio,
          automatable_tasks,
          )

    # Non-negativity of solution
    self.assertTrue(np.all(labour_task_input >= 0.))
    self.assertTrue(np.all(compute_task_input >= 0.))

    # Budget constraints met
    ## Labour gets rounded to 0.0 because it is very small compared to compute
    labour_I = (6008353853.645853 + 1.19163688e-20*6.72500025e+67) - 1.19163688e-20*6.72500025e+67
    self.assertAlmostEqual(np.sum(labour_task_input), labour, places=4)
    self.assertAlmostEqual(np.sum(compute_task_input), compute, delta=0.0001*compute)

    # Automation constraints met
    self.assertTrue(np.all(np.abs(compute_task_input[automatable_tasks:]) < 0.01))

    # All tasks done by compute
    self.assertTrue(np.all(compute_task_input > 0.))
  
  def test_allocation_automation_constrained(self):

    labour = 6008353853.645853 
    compute = 9.364651787582756e+68
    task_shares = np.array(
      [0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876,
       0.02484876, 0.02484876, 0.02484876, 0.02484876, 0.02484876])
    substitution = -0.5 
    task_compute_to_labour_ratio = np.array([1.00000000e+00,1.00000000e-07 ,1.00000000e-14 ,3.16227766e-15,1.00000000e-15 ,5.62341325e-16 ,3.16227766e-16 ,1.77827941e-16,1.00000000e-16 ,6.21689065e-17 ,3.86497294e-17 ,2.40281141e-17,1.49380158e-17 ,9.28680109e-18 ,5.77350269e-18 ,3.58932349e-18,2.23144317e-18 ,1.38726382e-18 ,8.62446746e-19 ,5.36173711e-19, 3.33333333e-19 ,2.79727271e-19 ,2.34742038e-19 ,1.96991249e-19, 1.65311473e-19 ,1.38726382e-19 ,1.16416656e-19 ,9.76947407e-20, 8.19836495e-20 ,6.87991876e-20 ,5.77350269e-20 ,4.84501845e-20, 4.06585136e-20 ,3.41198852e-20 ,2.86327871e-20 ,2.40281141e-20, 2.01639564e-20 ,1.69212254e-20 ,1.41999846e-20 ,1.19163688e-20]) 
    automatable_tasks = 38
    
    labour_task_input, compute_task_input = \
      SimulateTakeOff.solve_allocation(
          labour,
          compute,
          task_shares,
          substitution,
          task_compute_to_labour_ratio,
          automatable_tasks,
          )

    # Non-negativity of solution
    self.assertTrue(np.all(labour_task_input >= 0.))
    self.assertTrue(np.all(compute_task_input >= 0.))

    # Budget constraints met
    self.assertAlmostEqual(np.sum(labour_task_input), labour, places=4)
    self.assertAlmostEqual(np.sum(compute_task_input), compute, delta=0.0001*compute)

    # Automation constraints met
    self.assertTrue(np.all(np.abs(compute_task_input[automatable_tasks:]) < 0.01))

class TestTaskWeights(unittest.TestCase):
  def test_weight_estimation(self):
    # Define inputs
    capital = 915374690952521.1
    labour_task_input = np.array(
      [0.00000000e+00, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08,
       2.05894737e+08, 2.05894737e+08, 2.05894737e+08, 2.05894737e+08]
    )
    compute_task_input = np.array(
      [5.64766406e+25, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
    )
    task_compute_to_labour_ratio = np.array(
      [1.00000000e+00, 1.00000000e-07, 1.00000000e-14, 3.16227766e-15,
       1.00000000e-15, 5.62341325e-16, 3.16227766e-16, 1.77827941e-16,
       1.00000000e-16, 6.21689065e-17, 3.86497294e-17, 2.40281141e-17,
       1.49380158e-17, 9.28680109e-18, 5.77350269e-18, 3.58932349e-18,
       2.23144317e-18, 1.38726382e-18, 8.62446746e-19, 5.36173711e-19,
       3.33333333e-19, 2.79727271e-19, 2.34742038e-19, 1.96991249e-19,
       1.65311473e-19, 1.38726382e-19, 1.16416656e-19, 9.76947407e-20,
       8.19836495e-20, 6.87991876e-20, 5.77350269e-20, 4.84501845e-20,
       4.06585136e-20, 3.41198852e-20, 2.86327871e-20, 2.40281141e-20,
       2.01639564e-20, 1.69212254e-20, 1.41999846e-20, 1.19163688e-20]
    )
    capital_substitution = -0.5
    labour_substitution = -0.3
    target_capital_to_cognitive_share_ratio = 40 / 60
    target_compute_to_labour_share_ratio = 0.05 / 60

    # Compute task weights
    capital_task_weights, labour_task_weights =\
      SimulateTakeOff.adjust_task_weights(
        capital,
        labour_task_input,
        compute_task_input,
        task_compute_to_labour_ratio,
        capital_substitution,
        labour_substitution,
        target_capital_to_cognitive_share_ratio,
        target_compute_to_labour_share_ratio,
      ) 

    # Compute resulting shares
    capital_share, cognitive_share, labour_share, compute_share = \
      SimulateTakeOff.compute_shares(
            capital,
            labour_task_input,
            compute_task_input,
            capital_task_weights,
            labour_task_weights,
            task_compute_to_labour_ratio,
            capital_substitution,
            labour_substitution,
        )
    
    # Compute share ratios
    capital_to_cognitive_share_ratio = capital_share / cognitive_share
    compute_to_labour_share_ratio = compute_share / labour_share

    # Check that they match
    self.assertAlmostEqual(
        target_capital_to_cognitive_share_ratio, 
        capital_to_cognitive_share_ratio,
        places=2
        )
    self.assertAlmostEqual(
        target_compute_to_labour_share_ratio, 
        compute_to_labour_share_ratio,
        places=2
        )

#unittest.main(argv=[''], verbosity=2, exit=False);
#unittest.main(argv=['', 'TestAllocationFunction.test_allocation_function_complex_input'], verbosity=2, exit=False);
#unittest.main(argv=['', 'TestTaskWeights.test_weight_estimation'], verbosity=2, exit=False);
