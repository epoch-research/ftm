let python_to_js_keys = {
  'full_automation_requirements_training':          'training.full_automation_requirements',
  'flop_gap_training':                              'training.flop_gap',
  'goods_vs_rnd_requirements_training':             'training.goods_vs_rnd_requirements',

  'full_automation_requirements_runtime':           'runtime.full_automation_requirements',
  'flop_gap_runtime':                               'runtime.flop_gap',
  'goods_vs_rnd_requirements_runtime':              'runtime.goods_vs_rnd_requirements',

  'labour_substitution_goods':                      'goods.labour_substitution',
  'labour_substitution_rnd':                        'rnd.labour_substitution',
  'capital_substitution_goods':                     'goods.capital_substitution',
  'capital_substitution_rnd':                       'rnd.capital_substitution',

  'hardware_performance_ceiling':                   'hardware_performance.ceiling',
  'software_ceiling':                               'software.ceiling',

  'rnd_parallelization_penalty':                    'rnd.parallelization_penalty',
  'hardware_delay':                                 'hardware_delay',

  'frac_capital_hardware_rnd_growth':               'frac_capital.hardware_rnd.growth',
  'frac_labour_hardware_rnd_growth':                'frac_labour.hardware_rnd.growth',
  'frac_compute_hardware_rnd_growth':               'frac_compute.hardware_rnd.growth',
  'frac_labour_software_rnd_growth':                'frac_labour.software_rnd.growth',
  'frac_compute_software_rnd_growth':               'frac_compute.software_rnd.growth',
  'frac_gwp_compute_growth':                        'frac_gwp.compute.growth',
  'frac_compute_training_growth':                   'frac_compute.training.growth',

  'frac_capital_hardware_rnd_growth_rampup':        'frac_capital.hardware_rnd.growth_rampup',
  'frac_labour_hardware_rnd_growth_rampup':         'frac_labour.hardware_rnd.growth_rampup',
  'frac_compute_hardware_rnd_growth_rampup':        'frac_compute.hardware_rnd.growth_rampup',
  'frac_labour_software_rnd_growth_rampup':         'frac_labour.software_rnd.growth_rampup',
  'frac_compute_software_rnd_growth_rampup':        'frac_compute.software_rnd.growth_rampup',
  'frac_gwp_compute_growth_rampup':                 'frac_gwp.compute.growth_rampup',
  'frac_compute_training_growth_rampup':            'frac_compute.training.growth_rampup',

  'frac_capital_hardware_rnd_ceiling':              'frac_capital.hardware_rnd.ceiling',
  'frac_labour_hardware_rnd_ceiling':               'frac_labour.hardware_rnd.ceiling',
  'frac_compute_hardware_rnd_ceiling':              'frac_compute.hardware_rnd.ceiling',
  'frac_labour_software_rnd_ceiling':               'frac_labour.software_rnd.ceiling',
  'frac_compute_software_rnd_ceiling':              'frac_compute.software_rnd.ceiling',
  'frac_gwp_compute_ceiling':                       'frac_gwp.compute.ceiling',
  'frac_compute_training_ceiling':                  'frac_compute.training.ceiling',

  'initial_frac_capital_hardware_rnd':              'initial.frac_capital.hardware_rnd',
  'initial_frac_labour_hardware_rnd':               'initial.frac_labour.hardware_rnd',
  'initial_frac_compute_hardware_rnd':              'initial.frac_compute.hardware_rnd',
  'initial_frac_labour_software_rnd':               'initial.frac_labour.software_rnd',
  'initial_frac_compute_software_rnd':              'initial.frac_compute.software_rnd',

  'initial_biggest_training_run':                   'initial.biggest_training_run',
  'ratio_initial_to_cumulative_input_hardware_rnd': 'initial.ratio_initial_to_cumulative_input_hardware_rnd',
  'ratio_initial_to_cumulative_input_software_rnd': 'initial.ratio_initial_to_cumulative_input_software_rnd',
  'initial_hardware_production':                    'initial.hardware_production',
  'ratio_hardware_to_initial_hardware_production':  'initial.ratio_hardware_to_initial_hardware_production',
  'initial_buyable_hardware_performance':           'initial.buyable.hardware_performance',
  'initial_gwp':                                    'initial.gwp',
  'initial_cognitive_share_goods':                  'initial.goods.share.cognitive',
  'initial_cognitive_share_rnd':                    'initial.rnd.share.cognitive',
  'initial_compute_share_goods':                    'initial.goods.share.compute',
  'initial_compute_share_rnd':                      'initial.rnd.share.compute',
  'initial_capital_growth':                         'initial.capital_growth',
  'initial_population':                             'initial.population',

  'labour_growth':                                  'labour_growth',
  'tfp_growth':                                     'tfp_growth',
  'compute_depreciation':                           'compute_depreciation',
  't_start':                                        't_start',
  't_end':                                          't_end',
  't_step':                                         't_step',
  'rampup_trigger':                                 'rampup_trigger',
  'hardware_returns':                               'hardware_performance.returns',
  'software_returns':                               'software.returns',
  'runtime_training_tradeoff':                      'runtime_training_tradeoff',
  'runtime_training_max_tradeoff':                  'runtime_training_max_tradeoff',
};

let internal_variables = {
  'frac_gwp_compute':                   'frac_gwp.compute.v',
  'frac_capital_hardware_rnd':          'frac_capital.hardware_rnd.v',
  'frac_labour_hardware_rnd':           'frac_labour.hardware_rnd.v',
  'frac_compute_hardware_rnd':          'frac_compute.hardware_rnd.v',
  'frac_labour_software_rnd':           'frac_labour.software_rnd.v',
  'frac_compute_software_rnd':          'frac_compute.software_rnd.v',
  'frac_compute_training':              'frac_compute.training.v',
  'frac_capital_goods':                 'frac_capital.goods.v',
  'frac_labour_goods':                  'frac_labour.goods.v',
  'frac_compute_goods':                 'frac_compute.goods.v',
  'task_compute_to_labour_ratio_goods': 'task_compute_to_labour_ratio',
  'task_compute_to_labour_ratio_rnd':   'task_compute_to_labour_ratio_rnd',
  'labour_task_input_goods':            'goods.labour_task_input',
  'compute_task_input_goods':           'goods.compute_task_input',
  'task_input_goods':                   'goods.task_input',
  'gwp':                                'gwp',
  'labour_task_input_hardware_rnd':     'hardware_rnd.labour_task_input',
  'compute_task_input_hardware_rnd':    'hardware_rnd.compute_task_input',
  'task_input_hardware_rnd':            'hardware_rnd.task_input',
  'rnd_input_hardware':                 'hardware_performance.rnd_input',
  'labour_task_input_software_rnd':     'software_rnd.labour_task_input',
  'compute_task_input_software_rnd':    'software_rnd.compute_task_input',
  'task_input_software_rnd':            'software_rnd.task_input',
  'rnd_input_software':                 'software.rnd_input',
  'hardware_performance':               'hardware_performance.v',
  'cumulative_rnd_input_hardware':      'hardware_performance.cumulative_rnd_input',
  'software':                           'software.v',
  'cumulative_rnd_input_software':      'software.cumulative_rnd_input',
  'rampup':                             'rampup',
  'compute_investment':                 'compute_investment',
  'hardware':                           'hardware',
  'compute':                            'compute',
  'capital':                            'capital',
  'labour':                             'labour',
  'tfp_goods':                          'goods.tfp',
  'tfp_rnd':                            'rnd.tfp',
  'biggest_training_run':               'biggest_training_run',
  'automatable_tasks_goods':            'goods.automatable_tasks',
  'automatable_tasks_rnd':              'rnd.automatable_tasks',
};

function transform_python_to_js_params(params) {
  let result = {};
  for (let python_key in params) {
    let js_key = python_to_js_keys[python_key];
    let fields = js_key.split('.');

    let r = result;
    for (let i = 0; i < fields.length - 1; i++) {
      let field = fields[i];
      if (!r[field]) r[field] = {};
      r = r[field];
    }

    r[fields[fields.length-1]] = params[python_key];
  }
  return result;
}

function get_internal_variables(sim, step) {
  let state = sim.states[step];
  let variables = {};

  for (let name in internal_variables) {
    let path = internal_variables[name];
    variables[name] = sim.get_by_path(state, path);
  }

  return variables;
}

function get_external_variables(sim, step) {
  let params = sim.params;
  let state = sim.states[step];
  let state0 = sim.states[0];
  let variables = {};

  // Wrapping the variables in lambdas to prevent exceptions if they are missing in this version of the model
  let getters = {
    "rampup":                       () => state.rampup,
    "hardware performance":         () => np.log10(state.hardware_performance.v),
    "frac_gwp_compute":             () => state.frac_gwp.compute.v,
    "hardware":                     () => np.log10(state.hardware),
    "software":                     () => state.software.v,
    "compute":                      () => np.log10(state.compute),
    "labour":                       () => state.labour / state0.labour,
    "capital":                      () => state.capital / state0.capital * 0.025,
    "automatable tasks goods":      () => sim.get_automatable_task_count(state, params, 'goods') + 1,
    "frac automatable tasks goods": () => sim.get_automatable_task_count(state, params, 'goods')/params.goods.n_labour_tasks,
    "automatable tasks rnd":        () => sim.get_automatable_task_count(state, params, 'rnd') + 1,
    "frac automatable tasks rnd":   () => sim.get_automatable_task_count(state, params, 'rnd')/params.rnd.n_labour_tasks,
    "gwp":                          () => state.gwp,
    "frac_capital_rnd":             () => state.frac_capital.hardware_rnd.v,
    "frac_labour_rnd":              () => state.frac_labour.hardware_rnd.v,
    "frac_compute_rnd":             () => state.frac_compute.hardware_rnd.v,
    "rnd input hardware":           () => state.hardware_performance.rnd_input / state0.hardware_performance.rnd_input * 0.003048307243707020,
    "cumulative input hardware":    () => state.hardware_performance.cumulative_rnd_input / state0.hardware_performance.rnd_input * 0.003048307243707020,
    "ratio rnd input hardware":     () => state0.hardware_performance.rnd_input**params.rnd.parallelization_penalty / state0.hardware_performance.cumulative_rnd_input,
    "biggest_training_run":         () => np.log10(state.biggest_training_run),
    "compute share goods":          () => state.goods.compute_share,
  }

  for (let name in getters) {
    variables[name] = getters[name]();
  }

  return variables;
}

function get_takeoff_metrics(sim, step) {
  return sim.get_takeoff_metrics();
}

