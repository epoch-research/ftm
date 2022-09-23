import unittest
 
import numpy as np
import pandas as pd
from scipy import stats
from opmodel.core.utils import *
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from opmodel.stats.distributions import ParamsDistribution, PointDistribution, AjeyaDistribution

from opmodel.core.opmodel import *

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

  def test_allocation_no_constraints_low_compute(self):
    labour                       = 8059798228.773804
    compute                      = 3.946365405860123e+28
    task_shares                  = np.array([0.9999999999373822, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11, 6.261780453829572e-11])
    substitution                 = -0.5
    task_compute_to_labour_ratio = np.array([1.0, 3.3902416654828264e-17, 2.1291705717481764e-17, 1.3371811719955348e-17, 8.397887470665481e-18, 5.274118081150724e-18, 3.3122998648274283e-18, 2.0802208493864348e-18, 1.3064393197526847e-18, 8.204819679118507e-19, 5.152865881256159e-19, 3.2740035186726697e-19, 2.3097447588832877e-19, 1.6294792662140133e-19, 1.149565409255472e-19, 8.109956705537747e-20, 5.721414130605575e-20, 4.036344562917477e-20, 2.847561294932668e-20, 2.0088982003404088e-20, 1.4172379665759957e-20, 1.0233604690250259e-20, 8.109956705537813e-21, 6.427001995529376e-21, 5.0932891691435174e-21, 4.0363445629174595e-21, 3.198734037975923e-21, 2.5349420214784818e-21, 2.0088982003404005e-21, 1.592017468303725e-21, 1.261646617511409e-21, 9.99833368143855e-22, 7.923508454575994e-22, 6.279244945214136e-22, 4.976194233657511e-22, 3.943548829060913e-22, 3.1251950058544324e-22, 2.4766635961607022e-22, 1.962713544933064e-22, 1.5554169187270422e-22, 1.2326413079015227e-22, 9.768471563165595e-23, 7.741346656865274e-23, 6.134884835795085e-23, 4.861791315738373e-23, 3.852886473088241e-23, 3.053346639221565e-23, 2.4197249943294913e-23, 1.917590676725647e-23, 1.5196578173479864e-23, 1.2043028315981174e-23, 9.768471563165597e-24, 8.109956705537747e-24, 6.733028533727083e-24, 5.589878575434133e-24, 4.640815397049909e-24, 3.8528864730882415e-24, 3.1987340379758704e-24, 2.6556451941093164e-24, 2.2047632948747814e-24, 1.830433220977601e-24, 1.5196578173480116e-24, 1.2616466175113984e-24, 1.0474411866322519e-24, 8.696040747270933e-25, 7.219605801575883e-25, 5.993843571455767e-25, 4.976194233657512e-25, 4.1313238752194716e-25, 3.4298976608502985e-25, 2.847561294932656e-25, 2.3640954133857427e-25, 1.9627135449330482e-25, 1.62947926621402e-25, 1.3528223137176653e-25, 1.123136851409359e-25, 9.324479454564721e-26, 7.741346656865369e-26, 6.427001995529297e-26, 5.335809967107718e-26, 4.4298831749064786e-26, 3.677766836579633e-26, 3.0533466392215516e-26, 2.5349420214784714e-26, 2.104553400427423e-26, 1.7472372061067234e-26, 1.4505870242036393e-26, 1.204302831598132e-26, 9.998333681438588e-27, 8.30079227437588e-27, 6.891463575601101e-27, 5.721414130605551e-27, 4.7500185257873086e-27, 3.943548829060946e-27, 3.274003518672683e-27, 2.718135239327948e-27, 2.2566436282486964e-27, 1.873505185185123e-27, 1.5554169187270548e-27, 1.2913344516968872e-27, 1.0720885481328383e-27])
    automatable_tasks            = 101

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

class TestParamsDistribution(unittest.TestCase):
  """Extremely poor test of the joint parameter distribution"""

  @classmethod
  def setUpClass(self):
    n_samples = 10000

    TestParamsDistribution.params_dist = ParamsDistribution()
    print("Sampling. This could take a while.")
    TestParamsDistribution.samples = TestParamsDistribution.params_dist.rvs(n_samples)

  def setUp(self):
    self.samples = TestParamsDistribution.samples
    self.marginals = TestParamsDistribution.params_dist.get_marginals()
    self.rank_correlations = TestParamsDistribution.params_dist.get_rank_correlations()
    self.parameters = list(self.marginals.keys())

  def test_rank_correlations(self):
    for i in range(len(self.parameters)):
      for j in range(i + 1, len(self.parameters)):
        left = self.parameters[i]
        right = self.parameters[j]

        if isinstance(self.marginals[left], PointDistribution) or isinstance(self.marginals[right], PointDistribution):
          continue

        expected_r = 0
        if (left, right) in self.rank_correlations:
          expected_r = self.rank_correlations[(left, right)]
        elif (right, left) in self.rank_correlations:
          expected_r = self.rank_correlations[(right, left)]

        r = stats.spearmanr(self.samples[left], self.samples[right]).correlation

        self.assertLess(np.abs(r - expected_r), 0.05)

  def test_marginals(self):
    for param, marginal in self.marginals.items():
      param_samples = self.samples[param]

      if isinstance(marginal, PointDistribution):
        self.assertTrue(np.all(param_samples == marginal.get_value()))
      elif isinstance(marginal, AjeyaDistribution):
        # Ajeya's distribution doesn't have a CDF
        p = np.linspace(0, 1, 100)
        max_diff = np.max(np.abs((np.log10(self.eppf(param_samples)(p)) - np.log10(marginal._ppf(p)))/np.log10(marginal._ppf(p))))
        self.assertLess(max_diff, 1)
      else:
        result = stats.kstest(param_samples, marginal.cdf)
        self.assertGreater(result.pvalue, 0.02)

  def ecdf(self, samples):
    result = ECDF(samples)
    result.x[0] = result.x[1] # get rid of -inf
    result.y[0] = 0           # get rid of -inf
    return [result.x, result.y]

  def eppf(self, samples):
    cdf = self.ecdf(samples)
    return interp1d(cdf[1], cdf[0])

class MiscTests(unittest.TestCase):
  def test_process_quantiles(self):
    np.random.seed(0)

    for i in range(1000):
      n_quantiles = 100

      quantiles = np.random.uniform(high = 1, size = n_quantiles - 2)
      quantiles = np.insert(quantiles, 0, 0)
      quantiles = np.append(quantiles, 1)
      quantiles.sort()
      quantiles = quantiles[::-1]

      values = 10**np.random.uniform(high = 40, size = n_quantiles)
      values.sort()
      values = values[::-1]

      quantile_dict = {quantiles[i]: values[i] for i in range(n_quantiles)}

      result = SimulateTakeOff.process_quantiles(quantile_dict, 100)
      expected_result = MiscTests.process_quantiles_old(quantile_dict, 100)

      self.assertTrue(np.all(np.abs(np.log10(result / expected_result)) < 1e-8))

  @staticmethod
  def process_quantiles_old(quantile_dict, n_items):
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

    return values

class TestTradeoff(unittest.TestCase):
  def setUp(self):
    self.parameters = []

    best_guess_parameters = {parameter : row['Best guess'] for parameter, row in get_parameter_table().iterrows()}
    self.parameters.append(best_guess_parameters)

    params_dist = ParamsDistribution()
    for row, sample in params_dist.rvs(10).iterrows():
      parameters = sample.to_dict()
      self.parameters.append(parameters)

    for parameters in self.parameters:
      parameters['runtime_training_tradeoff'] = 1
      parameters['runtime_training_max_tradeoff'] = 1e100

  def test_invariant_automated_tasks(self):
    for parameters in self.parameters:
      parameters = parameters.copy()

      model_1 = SimulateTakeOff(**parameters)
      model_1.run_simulation()

      parameters['full_automation_requirements_training'] /= 2
      parameters['full_automation_requirements_runtime']  *= 2

      model_2 = SimulateTakeOff(**parameters)
      model_2.run_simulation()

      # The number of tasks automated should be the same
      self.assertTrue(np.all(model_1.frac_tasks_automated_goods == model_2.frac_tasks_automated_goods))
      self.assertTrue(np.all(model_1.frac_tasks_automated_rnd == model_2.frac_tasks_automated_rnd))

  def test_invariant_runtime_requirements(self):
    # When biggest_training_run == training requirements, the actual runtime requirements for a task
    # should be the same as if there was no tradeoff

    model = SimulateTakeOff(**self.parameters[0])

    for i in range(100):
      # Pick random values for everything
      runtime_reqs  = np.array([10**np.random.uniform(high = 30)])
      training_reqs = np.array([10**np.random.uniform(high = 40)])
      model.runtime_training_tradeoff = 10**np.random.uniform(low = -3, high = 3)

      biggest_training_run = training_reqs[0]
      actual_reqs = model.compute_runtime_requirements(training_reqs, runtime_reqs, biggest_training_run)

      self.assertTrue(np.abs((actual_reqs[0] - runtime_reqs[0])/runtime_reqs[0]) < 1e-9)
