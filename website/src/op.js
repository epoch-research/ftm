

function pow(a, b) {
  return Math.pow(a, b);
  //return Math.exp(Math.log(a) * b);
}

let np = {
  nan: NaN,

  min:      (a) => Math.min(...a),
  max:      (a) => Math.max(...a),
  insert:   (a, i, x) => a.splice(i, 0, x),
  round:    (x) => Math.round(x),
  zeros:    (n) => new Array(n).fill(0),
  concatenate: (a, b) => a.concat(b),

  even_round: (x) => {
    // This is Python's way of rounding (see https://docs.python.org/3/library/functions.html#round)
    let n = Math.round(x);
    let r = x % 1;
    if (Math.abs(r) == 0.5 && (n % 2) != 0) {
      n--;
    }
    return n;
  },

  array: (array_or_count) => {
    if (Array.isArray(array_or_count)) return array_or_count;
    return Array(array_or_count);
  },

  clip: (m, a) => np.unaop(a, x => Math.max(x, m)),

  log10:   (a)    => np.unaop(a, Math.log10),
  exp:     (a)    => np.unaop(a, Math.exp),
  maximum: (m, a) => np.unaop(a, x => Math.max(x, m)),
  minimum: (m, a) => np.unaop(a, x => Math.min(x, m)),

  arange: (start, stop, step) => {
    let array = [];
    for (let x = start; x < stop; x += step) {
      array.push(x);
    }
    return array;
  },

  /*
  add:     (a, b) => np.binop(a, b, (x, y) => x + y),
  sub:     (a, b) => np.binop(a, b, (x, y) => x - y),
  mult:    (a, b) => np.binop(a, b, (x, y) => x * y),
  div:     (a, b) => np.binop(a, b, (x, y) => x / y),
  pow:     (a, b) => np.binop(a, b, (x, y) => (x == 0) ? 0 : x ** y),
  lt:      (a, b) => np.binop(a, b, (x, y) => x < y),
  gt:      (a, b) => np.binop(a, b, (x, y) => x > y),
  gte:     (a, b) => np.binop(a, b, (x, y) => x >= y),
  lte:     (a, b) => np.binop(a, b, (x, y) => x <= y),
  and:     (a, b) => np.binop(a, b, (x, y) => x && y),
  */

  add: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] + b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] + b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a + b[i];
    } else {
      c = a + b;
    }

    return c;
  },

  sub: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] - b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] - b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a - b[i];
    } else {
      c = a - b;
    }

    return c;
  },

  mult: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] * b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] * b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a * b[i];
    } else {
      c = a * b;
    }

    return c;
  },

  div: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] / b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] / b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a / b[i];
    } else {
      c = a / b;
    }

    return c;
  },

  pow: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = (a[i] == 0) ? 0 : a[i] ** b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = (a[i] == 0) ? 0 : a[i] ** b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = (a == 0) ? 0 : a ** b[i];
    } else {
      c = (a == 0) ? 0 : a ** b;
    }

    return c;
  },

  lt: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] < b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] < b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a < b[i];
    } else {
      c = a < b;
    }

    return c;
  },

  gt: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] > b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] > b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a > b[i];
    } else {
      c = a > b;
    }

    return c;
  },

  gte: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] >= b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] >= b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a >= b[i];
    } else {
      c = a >= b;
    }

    return c;
  },

  lte: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] <= b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] <= b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a <= b[i];
    } else {
      c = a <= b;
    }

    return c;
  },

  and: (a, b) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] && b[i];
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = a[i] && b;
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = a && b[i];
    } else {
      c = a && b;
    }

    return c;
  },

  count_true: (a) => {
    let count = 0;
    for (let x of a) if (x) count++;
    return count;
  },

  sum: (a) => {
    let acc = 0;
    for (let x of a) acc += x;
    return acc;
  },

  mean: (a) => np.sum(a)/a.length,

  unaop: (a, op) => {
    return Array.isArray(a) ? a.map(x => op(x)) : op(a);
  },

  binop: (a, b, op) => {
    let c;

    if (Array.isArray(a) && Array.isArray(b)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = op(a[i], b[i]);
    } else if (Array.isArray(a)) {
      c = np.array(a.length);
      for (let i = 0; i < c.length; i++) c[i] = op(a[i], b);
    } else if (Array.isArray(b)) {
      c = np.array(b.length);
      for (let i = 0; i < c.length; i++) c[i] = op(a, b[i]);
    } else {
      c = op(a, b);
    }

    return c;
  },

  linspace: (a, b, count) => {
    let array = [];

    let step = (b - a)/(count - 1);
    for (let i = 0; i < count; i++) {
      array.push(a + i*step);
    }

    return array;
  },

  all_equals: (arr, x) => {
    return arr.every(y => y == x);
  },

  any: (arr) => {
    for (let x of arr) if (x) return true;
    return false;
  },

  argmax: (arr) => {
    let max = -Infinity;
    let imax = 0;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > max) {
        imax = i;
        max = arr[i];
      }
    }

    return imax;
  },
};

function override(o1, o2) {
  for (let k in o2) {
    if (typeof o1[k] == 'object') {
      override(o1[k], o2[k]);
    } else {
      o1[k] = o2[k];
    }
  }
}

function run_model(params) {

let blocks = [];
let entry_point = null;

function cheap_deep_copy(d) {
  if (Array.isArray(d)) {
    return [...d];
  } else if (typeof(d) == 'object') {
    let r = {};
    for (let k in d) {
      r[k] = cheap_deep_copy(d[k]);
    }
    return r;
  } else {
    return d;
  }
}

function set_entry_point(b) {
  entry_point = blocks.length;
}

function block(b) {
  blocks.push(b);
  return b;
}

function frac_params(outputs) {
  let object = {};

  for (let out of outputs) {
    object[out] = {
      growth: 0,
      growth_rampup: 0,
      ceiling: 0,
    };
  }

  return object;
}

function production_params(config) {
  return {
    task_compute_to_labour_ratio: 0,
    automatable_tasks: 0,

    labour_task_weights: 0,
    labour_substitution: 0,

    capital_task_weights: 0,
    capital_substitution: 0,

    ...config,
  };
}

function rnd_params(config) {
  return {
    returns: 0,
    ceiling: 0,

    ...config,
  };
}

let input_params = {
  t_start: 2022,
  t_end: 2100,
  t_step: 0.1,

  money_cap_training_before_wakeup: 1e9,

  rampup_trigger: 0.03,
  hardware_delay: 1,
  compute_depreciation: 0.2,
  labour_growth: 0.01,
  tfp_growth: 0.01,

  n_labour_tasks: 100,

  training: {
    full_automation_requirements: 1e36,
    flop_gap: 1e5,
    goods_vs_rnd_requirements: 3,
  },

  runtime: {
    full_automation_requirements: 1e17/6,
    flop_gap: 100,
    goods_vs_rnd_requirements: 10,
  },

  runtime_training_tradeoff: 0,
  runtime_training_max_tradeoff: 1,

  goods: production_params({
    labour_substitution: -0.5,
    capital_substitution: -0.4,
  }),

  rnd: production_params({
    parallelization_penalty: 0.7,
    labour_substitution: -0.5,
    capital_substitution: -0.25,
  }),

  hardware_rnd: {
  },

  software_rnd: {
  },

  hardware_performance: rnd_params({
    returns: 5.1,
    ceiling: 1e30,
  }),

  software: rnd_params({
    returns: 1.25,
    ceiling: 1e7,
  }),

  frac_capital: {
    hardware_rnd: {
      growth: 0.01,
      growth_rampup: 0.14,
      ceiling: 0.03,
    },
  },

  frac_labour: {
    hardware_rnd: {
      growth: 0.01,
      growth_rampup: 0.14,
      ceiling: 0.03,
    },

    software_rnd: {
      growth: 0.18,
      growth_rampup: 0.22,
      ceiling: 0.03,
    },
  },

  frac_compute: {
    hardware_rnd: {
      growth: 0.01,
      growth_rampup: 0.67,
      ceiling: 0.2,
    },

    software_rnd: {
      growth: 0.18,
      growth_rampup: 0.67,
      ceiling: 0.2,
    },

    training: {
      growth: 0.547528364331348,
      growth_rampup: 1.1,
      ceiling: 0.1,
    },
  },

  frac_gwp: {
    compute: {
      growth: 0.19,
      growth_rampup: 0.19,
      ceiling: 0.3,
    },
  },

  initial: {
    frac_capital: { hardware_rnd: 0.002, },
    frac_labour:  { hardware_rnd: 0.002, software_rnd: 0.0002, },
    frac_compute: { hardware_rnd: 0.002, software_rnd: 0.0002, },

    biggest_training_run: 3e24,
    hardware_production: 1e28,
    buyable_hardware_performance: 1.5e17,
    gwp: 8.5e13,

    ratio_initial_to_cumulative_input_hardware_rnd: 0.047,
    ratio_initial_to_cumulative_input_software_rnd: 0.2,
    ratio_hardware_to_initial_hardware_production: 2,

    goods: {
      share: {
        cognitive: 0.5,
        compute: 0.01,
      },
    },

    rnd: {
      share: {
        cognitive: 0.7,
        compute: 0.01,
      },
    },

    capital_growth: 0.0275,
    tfp_growth: 0.01,
    compute_depreciation: 0.2,
  },
};

override(input_params, params);
if (input_params.runtime_training_tradeoff <= 0) {
  input_params.runtime_training_tradeoff = 0;
  input_params.runtime_training_max_tradeoff = 1;
}

function process_params(params) {
  params = cheap_deep_copy(params);

  params.goods.n_labour_tasks = params.n_labour_tasks;
  params.rnd.n_labour_tasks = params.n_labour_tasks;

  params.goods.automation_training_flops = process_automation_costs(
    params.training.full_automation_requirements,
    params.training.flop_gap,
    params.goods.n_labour_tasks
  );

  params.goods.automation_runtime_flops = process_automation_costs(
    params.runtime.full_automation_requirements,
    params.runtime.flop_gap,
    params.goods.n_labour_tasks
  );

  params.rnd.automation_training_flops = process_automation_costs(
    params.training.full_automation_requirements / params.training.goods_vs_rnd_requirements,
    params.training.flop_gap,
    params.rnd.n_labour_tasks
  );

  params.rnd.automation_runtime_flops = process_automation_costs(
    params.runtime.full_automation_requirements / params.runtime.goods_vs_rnd_requirements,
    params.runtime.flop_gap,
    params.rnd.n_labour_tasks
  );

  params.initial.software = 1;
  params.initial.tfp_goods = 1;
  params.initial.tfp_rnd = 1;
  params.initial.rnd_input_hardware = 8000000.0;
  params.initial.rnd_input_software = 8000000.0;
  params.initial.hardware = params.initial.hardware_production * params.initial.ratio_hardware_to_initial_hardware_production;
  params.investment_rate = 0.2;

  let share = params.initial.goods;
  for (let share of [params.initial.goods.share, params.initial.rnd.share]) {
    share.capital = 1 - share.cognitive;

    // Split cognitive share into compute and labour
    share.compute = share.compute * share.cognitive; // TERRIBLE HACK
    share.labour  = share.cognitive - share.compute;
  }

  // Returns to hardware and software need to be adjusted by the paralellization penalty
  params.hardware_performance.returns /= params.rnd.parallelization_penalty;
  params.software.returns /= params.rnd.parallelization_penalty;

  params.hardware_delay_idx = np.even_round(params.hardware_delay/params.t_step);

  return params;
}

function process_automation_costs(full_automation_flops, flop_gap, n_labour_tasks) {
  let automation_flops = quantiles_from_gap(full_automation_flops, flop_gap);
  automation_flops = interpolate_quantiles(automation_flops, n_labour_tasks);
  np.insert(automation_flops, 0, 1.0); // The first task is always automatable

  return automation_flops;
}

class SimulationState {
  t_idx = 0;
  t_year = 0;

  rampup = false;

  automation_training_flops = 0;

  capital = 0;
  labour  = 0;
  compute = 0;

  frac_capital = frac_state(['goods', 'hardware_rnd', 'software_rnd']);
  frac_labour  = frac_state(['goods', 'hardware_rnd', 'software_rnd']);
  frac_compute = frac_state(['goods', 'hardware_rnd', 'software_rnd', 'training']);
  frac_gwp = frac_state(['compute']);

  goods = { tfp: 0, };
  hardware_rnd = {};
  software_rnd = {};

  rnd = {
    tfp: 0,
    automation_runtime_flops: 0,
    automation_training_flops: 0,
    automation_multiplier: 0,
  };

  hardware_performance = {
    v: 0,
    rnd_input: 0,
  };

  software = {
    v: 0,
    rnd_input: 0,
  };

  constructor(params) {
    this.create_task_state(this.goods, params.goods.n_labour_tasks);
    this.goods.task_compute_to_labour_ratio = np.zeros(params.goods.n_labour_tasks + 1);

    this.create_task_state(this.hardware_rnd, params.rnd.n_labour_tasks);
    this.create_task_state(this.software_rnd, params.rnd.n_labour_tasks);
    this.rnd.task_compute_to_labour_ratio = np.zeros(params.rnd.n_labour_tasks + 1);

    this.frac_capital.hardware_rnd = cheap_deep_copy(params.frac_capital.hardware_rnd);

    this.frac_labour.hardware_rnd = cheap_deep_copy(params.frac_labour.hardware_rnd);
    this.frac_labour.software_rnd = cheap_deep_copy(params.frac_labour.software_rnd);

    this.frac_compute.hardware_rnd = cheap_deep_copy(params.frac_compute.hardware_rnd);
    this.frac_compute.software_rnd = cheap_deep_copy(params.frac_compute.software_rnd);
    this.frac_compute.training = cheap_deep_copy(params.frac_compute.training);

    this.frac_gwp.compute = cheap_deep_copy(params.frac_gwp.compute);
  }

  create_task_state(state, n) {
    state.task_input = np.zeros(n + 1);
    state.labour_task_input = np.zeros(n + 1);
    state.compute_task_input = np.zeros(n + 1);
  }
}

function init_input_state(state, params) {
  let initial = params.initial;


  //
  // Initialize R&D state

  // Hardware
  state.hardware_performance.cumulative_rnd_input = np.div(
    np.pow(initial.rnd_input_hardware, params.rnd.parallelization_penalty),
    np.mult(params.initial.ratio_initial_to_cumulative_input_hardware_rnd, params.rnd.parallelization_penalty)
  );

  // Software
  state.software.v = initial.software;

  state.software.cumulative_rnd_input = np.div(
    np.pow(initial.rnd_input_software, params.rnd.parallelization_penalty),
    np.mult(params.initial.ratio_initial_to_cumulative_input_software_rnd, params.rnd.parallelization_penalty)
  );


  //
  // Initialize total inputs

  state.labour = initial.population;

  // Capital is initialized so that its rate of growth in the first time step matches the gwp rate of growth 
  state.capital = initial.gwp * params.investment_rate / (np.exp(initial.capital_growth)-1);
  
  state.hardware = initial.hardware;
  
  state.compute = initial.hardware * initial.software;
  state.compute_investment = initial.hardware_production * params.t_step / initial.buyable_hardware_performance;

  state.goods.tfp = initial.tfp_goods;
  state.rnd.tfp = initial.tfp_rnd;

  state.money_spent_training = state.compute_investment * initial.biggest_training_run / (state.hardware * initial.software) / params.t_step;


  //
  // Initialize fractional inputs

  // Split of gwp between capital and compute
  state.frac_gwp.compute.v = state.compute_investment / initial.gwp / params.t_step;

  // R&D fractional inputs
  state.frac_capital.hardware_rnd.v = initial.frac_capital.hardware_rnd;
  state.frac_capital.goods.v = 1 - state.frac_capital.hardware_rnd.v;

  state.frac_labour.hardware_rnd.v = initial.frac_labour.hardware_rnd;
  state.frac_labour.software_rnd.v = initial.frac_labour.software_rnd;
  state.frac_labour.goods.v = 1 - state.frac_labour.hardware_rnd.v - state.frac_labour.software_rnd.v;

  state.frac_compute.hardware_rnd.v = initial.frac_compute.hardware_rnd;
  state.frac_compute.software_rnd.v = initial.frac_compute.software_rnd;
  state.frac_compute.training.v     = initial.biggest_training_run / state.compute;
  state.frac_compute.goods.v = 1 - state.frac_compute.hardware_rnd.v - state.frac_compute.software_rnd.v - state.frac_compute.training.v;

  // Initial compute must be greater than initial training run
  if (initial.biggest_training_run > state.compute) {
    throw new Error("Initial biggest training run is bigger than available compute");
  }


  //
  // Initialize production inputs

  function initialize_task_weight(state, params, group, item) {
    // Initialize task weights to match the initial economy share ratio

    let capital = state.capital * state.frac_capital[item].v;
    let labour  = state.labour  * state.frac_labour[item].v;
    let compute = state.compute * state.frac_compute[item].v;

    let no_automation_labour_task_input = np.zeros(params[group].n_labour_tasks + 1);
    no_automation_labour_task_input.fill(labour / params[group].n_labour_tasks, 1);
    
    let no_automation_compute_task_input = np.zeros(params[group].n_labour_tasks + 1);
    no_automation_compute_task_input[0] = compute;

    let share = initial[group].share;
    let initial_capital_to_cognitive_share_ratio = share.capital / share.cognitive;
    let initial_compute_to_labour_share_ratio    = share.compute / share.labour;

    let task_weights = adjust_task_weights(
      capital,
      no_automation_labour_task_input,
      no_automation_compute_task_input,
      get_task_compute_to_labour_ratio(params[group].automation_runtime_flops, params[group].automation_training_flops, params.initial.biggest_training_run, params, group),
      params[group].capital_substitution,
      params[group].labour_substitution,
      initial_capital_to_cognitive_share_ratio,
      initial_compute_to_labour_share_ratio,
    );

    params[item].capital_task_weights = task_weights.capital;
    params[item].labour_task_weights = task_weights.labour;
  }

  initialize_task_weight(state, params, 'goods', 'goods');
  initialize_task_weight(state, params, 'rnd', 'hardware_rnd');
  params.software_rnd.capital_task_weights = [0, 1];
  params.software_rnd.labour_task_weights = params.hardware_rnd.labour_task_weights;
}

function frac_state(outputs) {
  let state = frac_params(outputs);

  for (let out of outputs) {
    state[out] = {
      v: 0,
      growth: 0,
      growth_rampup: 0,
      ceiling: 0,
    };
  }

  return state;
}

function quantiles_from_gap(top, gap) {
  let unit = gap**(1/7);

  let quantiles = new Map();
  quantiles.set(1.0, top);
  quantiles.set(0.5, top/(unit**4));
  quantiles.set(0.2, top/(unit**7));
  quantiles.set(0.1, top/(unit**8.5));
  quantiles.set(0.05, top/(unit**9.5));
  quantiles.set(0.0, top/(unit**10.5));

  return quantiles;
}

function interpolate_quantiles(quantiles, n_items) {
  let values = [];

  let quantiles_keys = [...quantiles.keys()];

  for (let q of np.linspace(0, 1, n_items)) {
    let prev_quantile = np.max(quantiles_keys.filter(x => x <= q))
    let next_quantile = np.min(quantiles_keys.filter(x => q <= x));

    let prev_value = quantiles.get(prev_quantile);
    let next_value = quantiles.get(next_quantile);

    let value;
    if (prev_quantile == next_quantile) {
      value = prev_value;
    } else {
      value = prev_value*((next_value/prev_value)**((q-prev_quantile)/(next_quantile-prev_quantile)));
    }

    values.push(value);
  }

  values = np.array(values);

  return values;
}

function adjust_task_weights(
  capital,
  labour_task_input,
  compute_task_input,
  task_compute_to_labour_ratio,
  capital_substitution,
  labour_substitution,
  capital_to_cognitive_share_ratio,
  compute_to_labour_share_ratio,
  ) {
  /*
    Computes the task weights that would result in a target capital_to_labour_share_ratio
    and compute_to_labour_share_ratio of the economy
  */

  // Compute inner task weights

  let task_input = np.add(labour_task_input, np.mult(task_compute_to_labour_ratio, compute_task_input));

  let labour_share = np.sum(np.mult(labour_task_input, np.pow(task_input, labour_substitution-1)));
  let compute_share = np.sum(np.mult(task_compute_to_labour_ratio, np.mult(compute_task_input, np.pow(task_input, labour_substitution-1))));

  let compute_to_labour_task_weight_ratio = compute_to_labour_share_ratio * labour_share / compute_share;
  
  let [compute_task_weight, labour_task_weight] = odds_to_probs(compute_to_labour_task_weight_ratio);
  
  let n_labour_tasks = labour_task_input.length-1;
  let inner_task_weights = np.array([compute_task_weight].concat(Array(n_labour_tasks).fill(labour_task_weight)));

  // Compute outer task weights

  let cognitive_input = ces_production_function(task_input, inner_task_weights, labour_substitution);
  
  let capital_share = capital**capital_substitution;
  let cognitive_share = cognitive_input**capital_substitution;

  let capital_to_cognitive_task_weight_ratio = capital_to_cognitive_share_ratio * cognitive_share / capital_share;

  let [capital_task_weight, cognitive_task_weight] = odds_to_probs(capital_to_cognitive_task_weight_ratio);


  let outer_task_weights = np.array([capital_task_weight, cognitive_task_weight])

  return {capital: outer_task_weights, labour: inner_task_weights};
}

function odds_to_probs(o) {
  // Stable implementation of conversion between odds and probs

  let p;

  if (o < 1e-10) {
    // For small outputs the odds are approx equal to the probs
    p = o;
    p_not = 1.-p;
  } else if (o > 1e10) {
    // For big outputs the odds can be approx like this
    p = 1 - 1/o;
    p_not = 1/o;
  } else {
    p = 1/(1+1/o);
    p_not = 1.-p;
  }
  
  return [p, p_not];
}

function ces_production_function(inputs, alphas, rho, tfp=1) {
  let result = tfp*np.sum(np.div(np.mult(alphas, np.pow(inputs, rho)), np.sum(alphas)))**(1./rho);
  return result;

}

function nested_ces_production_function(
   capital, cognitive_inputs,
   outer_weights, inner_weights,
   outer_rho, inner_rho,
   tfp = 1)
{
  let result = tfp * ces_production_function(
    np.array([capital, ces_production_function(cognitive_inputs, inner_weights, inner_rho)]),
    outer_weights,
    outer_rho
  );

  return result;
}

function solve_allocation(L, C, beta, rho, eta, AT) {
  /*
    Solve the input allocation problem for
    L = Labour budget
    C = Compute budget
    beta = task weights 
    rho = labour / capital substitution parameter
    eta = compute / labour substitution ratio
    AT = number of automatable tasks

    See description of solution at the end of the notebook [TODO]
    We assume that 
      * eta is monotonically decreasing on its index.
      * task 0 is automatable
  */

  // Preprocessing parameters
  let N = beta.length;
  let sigma = 1. / (1.-rho);

  // Precompute partial sums

  let sums_beta = np.zeros(N + 1); // sums_beta[i] = sum(beta[i:]**sigma)
  {
    let sum = 0;
    for (let i = N; i >= 0; i--) {
      if (i < N) sum += beta[i]**sigma;
      sums_beta[i] = sum;
    }
  }

  sums_beta_eta = np.zeros(N + 1); // sums_beta_eta[i] = sum(beta[:i]**sigma * eta[:i]**(sigma-1))
  {
    let sum = 0;
    for (let i = 0; i <= N; i++) {
      sums_beta_eta[i] = sum;
      if (i < N) sum += ((beta[i]*eta[i])**sigma)/eta[i];
    }
  }

  let labour_input_task = np.zeros(N);
  let compute_input_task = np.zeros(N);

  let found_critical_index = false;

  // Iterate over critical indices
  for (let I = 0; I < AT; I++) {
    // Initialize
    labour_input_task.fill(0);
    compute_input_task.fill(0);

    // Equation 20
    let A = eta[I]**sigma * sums_beta[I];
    let B = sums_beta_eta[I];

    compute_input_task[I] = (C*A - L*B) / (A + eta[I]*B);
    
    // Equation 18
    labour_input_task[I] =
      (L + eta[I]*compute_input_task[I])
      * (beta[I]**sigma / sums_beta[I])
      - eta[I]*compute_input_task[I]
    ;
    
    if (labour_input_task[I] >= 0) {
      found_critical_index = true;

      // Equation 17
      for (let i = I+1; i < N; i++) {
        labour_input_task[i] = (L + eta[I]*compute_input_task[I]) * (beta[i]**sigma / sums_beta[I]);
      }
      
      // Equation 14
      let Z = sums_beta_eta[I+1];
      for (let i = 0; i < I; i++) {
        compute_input_task[i] = (C + labour_input_task[I]/eta[I]) * beta[i]**sigma * eta[i]**(sigma-1) / Z;
      }

      break;
    }
  }

  if (!found_critical_index) {
    // The critical index is the last one
    let I = AT-1;

    // Initialize
    labour_input_task.fill(0);
    compute_input_task.fill(0);

    // Equations 14 & 15
    let Z = sums_beta_eta[I+1];
    for (let i = 0; i < I+1; i++) {
      compute_input_task[i] = C * beta[i]**sigma * eta[i]**(sigma-1) / Z;
    }
    
    // We assume LI = 0
    labour_input_task[I] = 0;

    // Equation 22
    for (let i = I+1; i < N; i++) {
      labour_input_task[i] = L * (beta[i]**sigma / sums_beta[I+1]);
    }
  }
  
  // Fix rounding error
  if (np.all_equals(labour_input_task, 0)) {
    labour_input_task[labour_input_task.length-1] = L;
  }

  return {labour: labour_input_task, compute: compute_input_task};
}

function compute_shares(
  capital,
  labour_task_input,
  compute_task_input,
  capital_task_weights,
  labour_task_weights, 
  task_compute_to_labour_ratio,
  capital_substitution,
  labour_substitution,
) {

  // Compute inputs
  let task_input = np.add(labour_task_input, np.mult(compute_task_input, task_compute_to_labour_ratio));
  
  let cognitive_input = ces_production_function(
    task_input,
    labour_task_weights, 
    labour_substitution
  );

  // Compute capital and cognitive shares
  let capital_task_weight = capital_task_weights[0];
  let cognitive_task_weight = capital_task_weights[1];

  let capital_share = capital_task_weight*capital**capital_substitution;
  let cognitive_share = cognitive_task_weight*cognitive_input**capital_substitution;

  let sum = capital_share + cognitive_share;
  capital_share /= sum;
  cognitive_share /= sum;

  // Compute labour and compute shares
  let labour_share = np.sum(np.mult(labour_task_weights, np.mult(labour_task_input, np.pow(task_input, labour_substitution-1))));
  
  let compute_share = np.sum(np.mult(labour_task_weights, np.mult(np.mult(compute_task_input, task_compute_to_labour_ratio), np.pow(task_input, labour_substitution-1))));
  
  let cognitive_sum = labour_share + compute_share;
  labour_share /= cognitive_sum
  compute_share /= cognitive_sum
  
  labour_share *= cognitive_share
  compute_share *= cognitive_share
  
  return {capital_share, cognitive_share, labour_share, compute_share};
}

function get_automated_tasks_count(statex) {
  let count = np.count_true(np.gt(
    np.mult(statex.task_compute_to_labour_ratio, statex.compute_task_input),
    np.mult(10, statex.labour_task_input)
  )) - 1; // -1 to account for the initial task
  return count;
}

function get_frac_tasks_automated(statex, paramsx) {
  return get_automated_tasks_count(statex) / paramsx.n_labour_tasks;
}

function get_automatable_task_count(state, params, group) {
  // -1 to account for the initial task
  return np.count_true(np.lt(params[group].automation_training_flops, state.biggest_training_run * params.runtime_training_max_tradeoff)) - 1;
}

//
// Define the model
//

// Reinvest outputs in inputs

function _update_rnd(
    current_performance, 
    initial_performance,
    research_input, 
    cumulative_adjusted_input, 
    returns,
    performance_ceiling,
    params,
  ) {
  let adjusted_input = research_input ** params.rnd.parallelization_penalty;
  let new_cumulative_adjusted_input = cumulative_adjusted_input + adjusted_input*params.t_step;
  let growth_in_cumulative_inputs = new_cumulative_adjusted_input / cumulative_adjusted_input;
  let ceiling_penalty =
    (np.log10(performance_ceiling) - np.log10(current_performance)) /
    (np.log10(performance_ceiling) - np.log10(initial_performance));
  let performance_growth_rate = growth_in_cumulative_inputs**(returns * ceiling_penalty);
  let new_performance = Math.min(current_performance * performance_growth_rate, performance_ceiling);

  return [new_performance, new_cumulative_adjusted_input];
}

update_rnd_state = block({
  map: (state, params, states) => {
    // In the first time step, we move forward the buyable hardware performance to adjust for the delay in hardware performance
    if (state.t_idx == 1) {
      let result = _update_rnd(
        params.initial.buyable_hardware_performance, 
        params.initial.buyable_hardware_performance,
        state.hardware_performance.rnd_input,
        state.hardware_performance.cumulative_rnd_input,
        params.hardware_performance.returns,
        params.hardware_performance.ceiling,
        params,
      );

      let improved_hardware_performance = result[0];

      let initial_hardware_improvement_rate = improved_hardware_performance / params.initial.buyable_hardware_performance;

      params.initial.hardware_performance = params.initial.buyable_hardware_performance * initial_hardware_improvement_rate**params.hardware_delay_idx;

      state.hardware_performance.v = params.initial.hardware_performance;
      states[0].hardware_performance.v = params.initial.hardware_performance;
    }

    let hardware_result = _update_rnd(
      state.hardware_performance.v,
      states[0].hardware_performance.v,
      state.hardware_performance.rnd_input,
      state.hardware_performance.cumulative_rnd_input,
      params.hardware_performance.returns,
      params.hardware_performance.ceiling,
      params,
    );
    state.hardware_performance.v = hardware_result[0];
    state.hardware_performance.cumulative_rnd_input = hardware_result[1];

    let software_result = _update_rnd(
      state.software.v,
      states[0].software.v,
      state.software.rnd_input,
      state.software.cumulative_rnd_input,
      params.software.returns,
      params.software.ceiling,
      params,
    );
    state.software.v = software_result[0];
    state.software.cumulative_rnd_input = software_result[1];
  }
});

// Rampup

rampup = block({
  map: (state, params) => {
    state.rampup = get_frac_tasks_automated(state.goods, params.goods) >= params.rampup_trigger;
  }
});

// Calculate fractional inputs

function update_frac_input(x, state, params) {
  let frac = x.v * np.exp(params.t_step * (state.rampup ? x.growth_rampup : x.growth));
  frac = Math.min(frac, x.ceiling);
  x.v = frac;
}

allocate_fractional_inputs_compute = block({
  map: (state, params) => {
    update_frac_input(state.frac_gwp.compute, state, params);
  }
});

allocate_fractional_inputs_training = block({
  map: (state, params) => {
    update_frac_input(state.frac_compute.training, state, params);
  }
});

allocate_fractional_inputs_hardware_rnd = block({
  map: (state, params) => {
    update_frac_input(state.frac_capital.hardware_rnd, state, params);
    update_frac_input(state.frac_labour.hardware_rnd, state, params);
    update_frac_input(state.frac_compute.hardware_rnd, state, params);
  }
});

allocate_fractional_inputs_software_rnd = block({
  map: (state, params) => {
    update_frac_input(state.frac_capital.software_rnd, state, params);
    update_frac_input(state.frac_labour.software_rnd, state, params);
    update_frac_input(state.frac_compute.software_rnd, state, params);
  }
});

allocate_fractional_inputs_goods = block({
  map: (state, params) => {
    state.frac_capital.goods.v = 1 - state.frac_capital.hardware_rnd.v - state.frac_capital.software_rnd.v; 
    state.frac_labour.goods.v  = 1 - state.frac_labour.hardware_rnd.v  - state.frac_labour.software_rnd.v; 
    state.frac_compute.goods.v = 1 - state.frac_compute.hardware_rnd.v - state.frac_compute.software_rnd.v - state.frac_compute.training.v;
  }
});

cap_training = block({
  map: (state, params, states) => {
    // Cap the growth of the fraction of FLOP before rampup
    if (state.money_spent_training > params.money_cap_training_before_wakeup && !states[states.length-2].rampup) {
      state.frac_compute.training.v = states[states.length-2].frac_compute.training.v;
    }
  }
});

// Calculate total inputs

calculate_total_inputs_compute = block({
  map: (state, params, states) => {
    state.compute_investment = state.gwp * state.frac_gwp.compute.v * params.t_step;

    let buyable_hardware_performance;
    let t_diff = state.t_idx - params.hardware_delay_idx;
    if (t_diff >= 0) {
      buyable_hardware_performance = states[t_diff].hardware_performance.v;
    } else {
      buyable_hardware_performance = states[0].hardware_performance.v * (states[1].hardware_performance.v/states[0].hardware_performance.v)**t_diff;
    }

    let new_hardware = state.compute_investment * buyable_hardware_performance;

    state.hardware = state.hardware * ((1.-params.compute_depreciation)**params.t_step) + new_hardware;
    state.compute = state.hardware * state.software.v;
  }
});

calculate_total_inputs_non_compute = block({
  map: (state, params) => {
    let capital_investment = state.gwp * params.investment_rate * params.t_step
    state.capital = state.capital + capital_investment;

    state.labour = state.labour * np.exp(params.labour_growth * params.t_step)
  }
});

calculate_total_factor_production = block({
  map: (state, params) => {
    state.goods.tfp *= np.exp(params.tfp_growth * params.t_step)
    state.rnd.tfp   *= np.exp(params.tfp_growth * params.t_step)
  }
});


track_money_spent_training = block({
  map: (state, params) => {
    state.money_spent_training = state.compute_investment * state.frac_compute.training.v / params.t_step;
  }
});

// Automate tasks

set_entry_point();

let runtime_requirements_cache = {
  'goods': null,
  'rnd': null,
};

function get_task_compute_to_labour_ratio(automation_runtime_flops, automation_training_flops, biggest_training_run, params, group) {
  let runtime_requirements = np.mult(automation_runtime_flops, np.pow(np.div(automation_training_flops, biggest_training_run), params.runtime_training_tradeoff));
  runtime_requirements = np.maximum(1, runtime_requirements);
  let task_compute_to_labour_ratio = np.div(1, runtime_requirements);

  /*
  let runtime_requirements = runtime_requirements_cache[group];
  if (!runtime_requirements) {
    runtime_requirements = np.mult(automation_runtime_flops, np.pow(automation_training_flops, params.runtime_training_tradeoff));
    runtime_requirements_cache[group] = runtime_requirements;
  }

  runtime_requirements = np.div(runtime_requirements, biggest_training_run**params.runtime_training_tradeoff);
  runtime_requirements = np.maximum(1, runtime_requirements);
  let task_compute_to_labour_ratio = np.div(1, runtime_requirements);
  */

  return task_compute_to_labour_ratio;
}

automate_tasks = block({
  map: (state, params) => {
    state.biggest_training_run = state.compute * state.frac_compute.training.v;

    function update_automatable_tasks(s, p, group) {
      s.task_compute_to_labour_ratio = get_task_compute_to_labour_ratio(p.automation_runtime_flops, p.automation_training_flops, state.biggest_training_run, params, group);
      s.automatable_tasks = np.count_true(np.lt(p.automation_training_flops, np.mult(state.biggest_training_run, params.runtime_training_max_tradeoff)));
    }

    update_automatable_tasks(state.goods, params.goods, 'goods');
    update_automatable_tasks(state.rnd, params.rnd, 'rnd');
    state.hardware_rnd.task_compute_to_labour_ratio = state.rnd.task_compute_to_labour_ratio; // TODO hack
    state.software_rnd.task_compute_to_labour_ratio = state.rnd.task_compute_to_labour_ratio; // TODO hack
  },
});

// Produce outputs

production_blocks = {}
for (let [group, item] of [['goods', 'goods'], ['rnd', 'hardware_rnd'], ['rnd', 'software_rnd']]) {
  production_blocks[item] = block({
    map: (state, params) => {
      state[item].capital = state.capital * state.frac_capital[item].v;
      state[item].labour  = state.labour  * state.frac_labour[item].v;
      state[item].compute = state.compute * state.frac_compute[item].v;

      let r = solve_allocation(
        state[item].labour,
        state[item].compute,
        params[item].labour_task_weights,
        params[group].labour_substitution,
        state[group].task_compute_to_labour_ratio,
        get_automatable_task_count(state, params, group) + 1,
      );
      state[item].labour_task_input = r.labour;
      state[item].compute_task_input = r.compute;

      state[item].task_input = np.add(state[item].labour_task_input, np.mult(state[group].task_compute_to_labour_ratio, state[item].compute_task_input));
      state[item].output = nested_ces_production_function(
        state[item].capital, state[item].task_input,
        params[item].capital_task_weights, params[item].labour_task_weights,
        params[group].capital_substitution, params[group].labour_substitution,
        state[group].tfp
      );
    },
  });
}

post_production = block({
  map: (state, params, states) => {
    //
    // GWP
    //
    let output_to_gwp_factor = params.initial.gwp / states[0].goods.output;
    state.gwp = state.goods.output * output_to_gwp_factor;

    //
    // Compute selected shares
    //
    state.goods.compute_share = compute_shares(
      state.goods.capital,
      state.goods.labour_task_input,
      state.goods.compute_task_input,
      params.goods.capital_task_weights,
      params.goods.labour_task_weights, 
      state.goods.task_compute_to_labour_ratio,
      params.goods.capital_substitution,
      params.goods.labour_substitution,
    ).compute_share;

    //
    // Automation multiplier
    //

    // Compute how much worse is the hardware output without automation

    let {labour: no_automation_labour_task_input_rnd, compute: no_automation_compute_task_input_rnd} = solve_allocation(
      state.hardware_rnd.labour,
      state.hardware_rnd.compute,
      params.hardware_rnd.labour_task_weights,
      params.rnd.labour_substitution,
      state.rnd.task_compute_to_labour_ratio,
      1 // Only first task is automatable
    );

    let no_automation_task_input_hardware_rnd =
      np.add(no_automation_labour_task_input_rnd, np.mult(state.rnd.task_compute_to_labour_ratio, no_automation_compute_task_input_rnd));

    let no_automation_output = nested_ces_production_function(
      state.hardware_rnd.capital, no_automation_task_input_hardware_rnd,
      params.hardware_rnd.capital_task_weights, params.hardware_rnd.labour_task_weights,
      params.rnd.capital_substitution, params.rnd.labour_substitution,
      state.rnd.tfp
    );

    state.hardware_rnd.automation_multiplier = state.hardware_rnd.output / no_automation_output;
  }
});

reinvest_outputs_in_hardware_performance_rnd_inputs = block({
  map: (state, params, states) => {
    let rnd_input_to_hardware_performance_investment_factor = params.initial.rnd_input_hardware / states[0].hardware_rnd.output;
    state.hardware_performance.rnd_input = state.hardware_rnd.output * rnd_input_to_hardware_performance_investment_factor;
  }
});

reinvest_outputs_in_software_rnd_inputs = block({
  map: (state, params, states) => {
    let rnd_input_to_software_investment_factor = params.initial.rnd_input_software / states[0].software_rnd.output;
    state.software.rnd_input = state.software_rnd.output * rnd_input_to_software_investment_factor;
  }
});


//
// Utilities
//

function time_to_index(t) {
  return np.even_round((t - params.t_start) / params.t_step);
}

function index_to_time(i) {
  return params.t_start + i * params.t_step;
}

//
// Run the simulation
//

params = process_params(input_params);

states = [];
state = new SimulationState(params);
init_input_state(state, params);

if (typeof performance != 'undefined') {
  t0 = performance.now();
}

let timesteps = [];

let iter_count = Math.ceil((params.t_end - params.t_start)/params.t_step);
for (let i = 0; i < iter_count; i++) {
  let t = params.t_start + i * params.t_step;
  timesteps.push(t);

  state = cheap_deep_copy(state);
  states.push(state);
  state.t_idx = i;
  state.t_year = t;
  for (let j = 0; j < blocks.length; j++) {
    if (i == 0 && j < entry_point) {
      continue; // HACK TODO
    }
    let block = blocks[j];
    block.map(state, params, states);
  }
}

let rampup_start = null;
let rampup_mid = null;
let agi_year = null;

for (let i = 0; i < states.length; i++) {
  let state = states[i];
  let prev_state = (i < 1) ? null : states[i-1];

  let year = state.t_year - params.t_step;
  if (rampup_start == null && state.rampup) {
    rampup_start = year;
  }

  if (rampup_mid == null && prev_state && get_frac_tasks_automated(prev_state.goods, params.goods) >= 0.3) {
    rampup_mid = year;
  }

  if (agi_year == null && prev_state && get_frac_tasks_automated(prev_state.goods, params.goods) >= 1) {
    agi_year = year;
  }
}

// Compute doubling times
let doubling_times = [params.t_step / Math.log2(states[1].gwp/states[0].gwp)]
if (rampup_start != null) {
  let reference_idx = time_to_index(rampup_start);
  for (let i = reference_idx; i < states.length; i++) {
    if (states[i].gwp > 2*states[reference_idx].gwp) {
      doubling_times.push(index_to_time(i) - index_to_time(reference_idx))
      reference_idx = i;
    }
  }
}

// Round doubling times
for (let i = 0; i < doubling_times.length; i++) {
  doubling_times[i] = +doubling_times[i].toFixed(2);
}

// We are only interested in the first five doubling times
doubling_times = doubling_times.slice(0, 5);

if (typeof performance != 'undefined') {
  t1 = performance.now();
  console.log('Time:', t1 - t0);
}

function get_thread(path) {
  let thread = [];
  for (let state of states) {
    thread.push(get_by_path(state, path));
  }
  return thread;
}

function get_by_path(state, path) {
  let fields = path.split('.');
  let x = state;
  for (let f of fields) {
    x = x[f];
  }
  return x;
}

function get_growth(path) {
  let thread = get_thread(path);

  let steps_per_year = Math.floor(1 / params.t_step);
  let growth = np.array(states.length - steps_per_year);
  let t = np.array(states.length - steps_per_year);
  for (let i = steps_per_year; i < states.length; i++) {
    t[i-steps_per_year] = states[i].t_year;
    growth[i-steps_per_year] = Math.log(thread[i]/thread[i-steps_per_year]);
  }

  return {t, growth};
}

function length_between_thresholds(series1, series2) {
  // Utility function to measure the amount of time between two thresholds being crossed
  if (!np.any(series1) || !np.any(series2)) {
    return params.t_end - params.t_start;
  }
  let idx1 = np.argmax(series1);
  let idx2 = np.argmax(series2);
  return (idx2 - idx1) * params.t_step;
}

// Compute takeoff metrics
function get_takeoff_metrics() {
  let takeoff_metrics = {};

  let frac_automated_tasks = np.array(states.length);
  for (let i = 0; i < states.length; i++) {
    frac_automated_tasks[i] =
      (get_automated_tasks_count(states[i].goods) + get_automated_tasks_count(states[i].hardware_rnd)) / (params.goods.n_labour_tasks + params.rnd.n_labour_tasks);
  }

  let frac_automatable_tasks = np.array(states.length);
  for (let i = 0; i < states.length; i++) {
    frac_automatable_tasks[i] =
      (get_automatable_task_count(states[i], params, 'goods') + get_automatable_task_count(states[i], params, 'rnd')) / (params.goods.n_labour_tasks + params.rnd.n_labour_tasks);
  }

  let frac_tasks_automated_goods = np.array(states.length);
  for (let i = 0; i < states.length; i++) {
    frac_tasks_automated_goods[i] = get_frac_tasks_automated(states[i].goods, params.goods);
  }

  // Years from "total cognitive output is 2X human cognitive output" to "total cognitive output is 10X human cognitive output"
  let rnd_automation_multiplier = get_thread('hardware_rnd.automation_multiplier');
  takeoff_metrics["cog_output_multiplier"] = length_between_thresholds(np.gt(rnd_automation_multiplier, 2), np.gt(rnd_automation_multiplier, 10));

  // Time from AI that automates 10% of cognitive tasks to when we have enough compute to run 10 billion AGIs

  let ten_billion_agi_compute = Math.max(np.max(params.goods.automation_runtime_flops), np.max(params.rnd.automation_runtime_flops)) * 1e10;
  let full_automation_flops = Math.max(np.max(params.goods.automation_training_flops), np.max(params.rnd.automation_training_flops));
  takeoff_metrics["billion_agis"] = length_between_thresholds(
    np.gt(frac_automated_tasks, 0.1),
    np.and(np.gte(get_thread('compute'), ten_billion_agi_compute),
           np.gte(get_thread('biggest_training_run'), full_automation_flops)
    ),
  );

  // Time from AI that can perform 50% of tasks to AI that can perform 100%.
  takeoff_metrics["full_automation"] = length_between_thresholds(np.gt(frac_automatable_tasks, 0.5), np.gte(frac_automatable_tasks, 1));

  // Time from rampup to full automation

  takeoff_metrics["rampup_to_agi"] = length_between_thresholds(np.gt(frac_tasks_automated_goods, 0.03), np.gte(frac_tasks_automated_goods, 1));

  // Combined metric
  takeoff_metrics["combined"] = np.mean(Object.values(takeoff_metrics));

  // Time from 5% GWP growth to 15% GWP growth
  let steps_per_year = Math.floor(1 / params.t_step);
  let gwp_growth = np.array(states.length - steps_per_year);
  for (let i = 0; i < states.length - steps_per_year; i++) {
    gwp_growth[i] = Math.log(states[i+steps_per_year].gwp/states[i].gwp);
  }
  takeoff_metrics["gwp_growth"] = length_between_thresholds(np.gt(gwp_growth, 0.05), np.gt(gwp_growth, 0.15));

  takeoff_metrics['doubling_times'] = doubling_times;

  for (let k in takeoff_metrics) {
    takeoff_metrics[k] = [takeoff_metrics[k]];
  }

  return takeoff_metrics;
}

function get_summary_table() {
  let rows = [];
  
  let prerampup = (rampup_start == null) ? null : (params.t_start + rampup_start)/2;

  let frac_tasks_automated_goods = np.array(states.length);
  for (let i = 0; i < states.length; i++) {
    frac_tasks_automated_goods[i] = get_frac_tasks_automated(states[i].goods, params.goods);
  }

  let frac_tasks_automated_rnd = np.array(states.length);
  for (let i = 0; i < states.length; i++) {
    frac_tasks_automated_rnd[i] = get_frac_tasks_automated(states[i].hardware_rnd, params.rnd);
  }

  let raw_metrics = {
    'biggest_training_run':       get_thread('biggest_training_run'),
    'frac_tasks_automated_goods': frac_tasks_automated_goods,
    'frac_tasks_automated_rnd':   frac_tasks_automated_rnd,
  };

  let doubling_time_metrics = {
    'hardware_performance':          get_thread('hardware_performance.v'),
    'software':                      get_thread('software.v'),
    'compute_investment':            get_thread('compute_investment'),
    'frac_compute_training':         get_thread('frac_compute.training.v'),
    'gwp':                           get_thread('gwp'),
    'capital':                       get_thread('capital'),
    'labour':                        get_thread('labour'),
    'tfp_rnd':                       get_thread('rnd.tfp'),
    'rnd_input_software':            get_thread('software.rnd_input'),
    'cumulative_rnd_input_software': get_thread('software.cumulative_rnd_input'),
  };

  for (let [period, t] of Object.entries({'prerampup': prerampup, 
                    'rampup_start': rampup_start,
                    'mid rampup': rampup_mid, 
                    'agi': agi_year})) {

    if (t == null) {
      let summary_row = {
        'period' : period,
        'year' : np.nan,
      };
    
      for (let raw_metric in raw_metrics) {
        summary_row[raw_metric] = np.nan;
      }
    
      for (let doubling_time_metric in doubling_time_metrics) {
        summary_row[`${doubling_time_metric} growth rate`] = np.nan;
        summary_row[`${doubling_time_metric} doubling time`] = np.nan;
      }
        
      rows.push(summary_row);
      continue;
    }
      
    let idx = time_to_index(t);
    t = index_to_time(idx)
    let t_end = t + 1;
    let idx_end = time_to_index(t_end);

    if (idx_end >= states.length) {
      let diff = idx_end - (states.length - 1);
      idx -= diff;
      idx_end -= diff;
    }
    
    let summary_row = {
      'period' : period,
      'year' : t,
    }
    
    // Auxiliary functions
    dt = s => (Math.log2(s[idx_end]/s[idx]) != 0) ? 1 / Math.log2(s[idx_end]/s[idx]) : np.nan;
    gr = s => Math.log(s[idx_end] / s[idx]);
    
    for (let raw_metric in raw_metrics) {
      summary_row[raw_metric] = raw_metrics[raw_metric][idx];
    }
    
    for (let doubling_time_metric in doubling_time_metrics) {
      summary_row[`${doubling_time_metric} growth rate`]   = gr(doubling_time_metrics[doubling_time_metric]);
      summary_row[`${doubling_time_metric} doubling time`] = dt(doubling_time_metrics[doubling_time_metric]);
    }

    rows.push(summary_row)
  }

  summary_table = {};
  for (let row of rows) {
    for (let col in row) {
      if (!(col in summary_table)) summary_table[col] = [];
      summary_table[col].push(row[col]);
    }
  }

  return summary_table;
}

return {
  get_thread: get_thread,
  get_by_path: get_by_path,
  get_automatable_task_count: get_automatable_task_count,
  get_takeoff_metrics: get_takeoff_metrics,
  takeoff_metrics: get_takeoff_metrics(),
  get_growth: get_growth,
  get_summary_table: get_summary_table,
  time_to_index: time_to_index,
  states: states,
  params: params,
  rampup_start: rampup_start,
  rampup_mid: rampup_mid,
  agi_year: agi_year,
  doubling_times: doubling_times,
  timesteps: timesteps,
}
}

function run_bioanchors_model({
    t_start                                = 2022,
    t_end                                  = 2100,
    t_step                                 = 0.1,   // Step duration, in years

    t_initial                              = 2022,  // All "initial" values refer to this year

    initial_hardware                       = 1,
    initial_software                       = 1,
    hardware_doubling_time                 = 2.5,   // In years

    software_doubling_start                = 2025,  // When the software starts doubling
    software_doubling_time                 = 2.5,   // In years
    software_ceiling                       = 1e3,   // Max software performance from t_initial levels

    initial_training_investment            = 1,
    fast_training_investment_duration      = 23,    // In years from t_initial
    fast_training_investment_doubling_time = 2.5,   // In years
    slow_training_investment_growth        = 1.03,  // Per year
  }) {

  let hardware_growth = np.pow(2, 1/hardware_doubling_time);
  let software_growth = np.pow(2, 1/software_doubling_time);
  let fast_training_investment_growth = np.pow(2, 1/fast_training_investment_doubling_time);

  let timesteps = np.arange(t_start, t_end, t_step);

  let slow_training_investment_start_index = np.round((t_initial + fast_training_investment_duration)/t_step - t_start);

  let fast_training_investment = np.mult(
      initial_training_investment,
      np.pow(fast_training_investment_growth, np.sub(timesteps.slice(0, slow_training_investment_start_index), t_initial))
  );
  let slow_training_investment = np.mult(
      fast_training_investment[fast_training_investment.length-1],
      np.pow(slow_training_investment_growth, np.sub(timesteps.slice(slow_training_investment_start_index), timesteps[slow_training_investment_start_index-1]))
  );
  let training_investment = np.concatenate(fast_training_investment, slow_training_investment)

  let hardware = np.mult(initial_hardware, np.pow(hardware_growth, np.sub(timesteps, t_initial)));

  let software_timesteps = np.arange(software_doubling_start, t_end, t_step);
  let software = np.mult(
     initial_software,
     np.pow(software_growth, np.sub(software_timesteps, software_doubling_start))
  );
  software = np.minimum(software_ceiling, software);

  return {timesteps, training_investment, hardware, software_timesteps, software};
}

