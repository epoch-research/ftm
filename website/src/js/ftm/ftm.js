/*******************************************************************************

Full Takeoff Model

Output:
  ftm: {
    run_simulation,
    run_bioanchors_model,
    Model,
  }

Dependencies:
  * ./nj.js

TODO:
  * oh, please, do something about the v. annoying .v field

*******************************************************************************/

"use strict";

let ftm = {};

{
  function run_simulation(input_params) {
    let model = new Model(input_params);
    model.run_simulation();
    return model;
  }

  let default_input_params = {
    t_start: 2022,
    t_end: 2100,
    t_step: 0.1,

    money_cap_training_before_wakeup: 1e9,
    research_to_compute_experiments_task_weight_ratio: 1.857142857,

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
      steepness: 0,
    },

    runtime: {
      full_automation_requirements: 1e17/6,
      flop_gap: 100,
      goods_vs_rnd_requirements: 10,
      steepness: 0,
    },

    runtime_training_tradeoff: 0,
    runtime_training_max_tradeoff: 1,
    bright_line_growth_rate: Math.infinity,

    goods: create_production_params({
      labour_substitution: -0.5,
      capital_substitution: -0.4,
    }),

    rnd: create_production_params({
      parallelization_penalty: 0.7,
      labour_substitution: -0.5,
      capital_substitution: -0.25,
    }),

    hardware_rnd: {
    },

    software_rnd: {
      experiments_substitution: -0.1,
    },

    hardware_performance: create_rnd_params({
      returns: 5.1,
      ceiling: 1e30,
    }),

    software: create_rnd_params({
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
      frac_compute: { hardware_rnd: 0.002, software_rnd: 0.0002},

      biggest_training_run: 3e24,
      hardware_production: 1e28,
      buyable_hardware_performance: 1.5e17,
      population: 8e9,
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

      hardware_rnd: {
        share: {
          cognitive: 0.7,
        },
      },

      software_rnd: {
        share: {
          experiments: 0.7,
        },
      },

      rnd: {
        share: {
          compute: 0.01,
        },
      },

      capital_growth: 0.0275,
      tfp_growth: 0.01,
      compute_depreciation: 0.2,
    },
  };

  function create_production_params(config) {
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

  function create_rnd_params(config) {
    return {
      returns: 0,
      ceiling: 0,

      ...config,
    };
  }

  class Model {
    constructor(input_params) {
      this.consts = Model.process_params(input_params);
      this.initial_state = new SimulationState(this.consts);
      Model.init_state_and_consts(this.initial_state, this.consts);
    }

    run_simulation() {
      let state = this.initial_state;

      let states = [];
      let timesteps = [];
      let t = this.consts.t_start;
      let t_idx = 0;

      while (t < this.consts.t_end) {
        state = cheap_deep_copy(state);
        states.push(state);

        timesteps.push(t);

        state.t_idx = t_idx;
        state.t_year = t;

        if (t_idx > 0) {
          Model.reinvest_output_in_inputs(state, this.consts, states)
        }
        Model.automate_tasks(state, this.consts, states);
        Model.production(state, this.consts, states);

        t_idx++;
        t += this.consts.t_step;
      }

      this.states = states;
      this.timesteps = timesteps;

      this.post_simulation(this.states, this.consts);

      this.takeoff_metrics = this.get_takeoff_metrics();
      this.timeline_metrics = this.get_timeline_metrics();
    }

    // ------------------------------------------------------------------------
    // Initialization functions
    // ------------------------------------------------------------------------

    static process_params(input_params) {
      input_params = input_params || {};

      let consts = cheap_deep_copy(default_input_params);

      consts.t_end = input_params.t_end;

      override(consts, input_params);
      if (consts.runtime_training_tradeoff <= 0) {
        consts.runtime_training_tradeoff = 0;
        consts.runtime_training_max_tradeoff = 1;
      }

      consts.goods.n_labour_tasks = consts.n_labour_tasks;
      consts.rnd.n_labour_tasks = consts.n_labour_tasks;

      let seconds_per_year = 60*60*24*365;

      consts.goods.automation_training_flops = this.process_automation_costs(
        consts.training.full_automation_requirements,
        consts.training.flop_gap,
        consts.goods.n_labour_tasks,
        consts.training.steepness,
      );

      consts.goods.automation_runtime_flops = this.process_automation_costs(
        consts.runtime.full_automation_requirements * seconds_per_year,
        consts.runtime.flop_gap,
        consts.goods.n_labour_tasks,
        consts.runtime.steepness,
      );

      consts.rnd.automation_training_flops = this.process_automation_costs(
        consts.training.full_automation_requirements / consts.training.goods_vs_rnd_requirements,
        consts.training.flop_gap,
        consts.rnd.n_labour_tasks,
        consts.training.steepness,
      );

      consts.rnd.automation_runtime_flops = this.process_automation_costs(
        consts.runtime.full_automation_requirements * seconds_per_year / consts.runtime.goods_vs_rnd_requirements,
        consts.runtime.flop_gap,
        consts.rnd.n_labour_tasks,
        consts.runtime.steepness,
      );

      consts.initial.software = 1;
      consts.initial.tfp_goods = 1;
      consts.initial.tfp_rnd = 1;
      consts.initial.hardware = consts.initial.hardware_production * consts.initial.ratio_hardware_to_initial_hardware_production;
      consts.investment_rate = 0.2;

      consts.software_rnd.experiments_task_weights = odds_to_probs(input_params.research_to_compute_experiments_task_weight_ratio);

      let frac_gwp_hardware_rnd_2020 = 0.2e-2;
      let frac_gwp_software_rnd_2020 = 0.02e-2;
      consts.initial.rnd_input_hardware = consts.initial.gwp * frac_gwp_hardware_rnd_2020
      consts.initial.rnd_input_software = consts.initial.gwp * frac_gwp_software_rnd_2020

      consts.initial.goods.share.capital = 1 - consts.initial.goods.share.cognitive;
      consts.initial.hardware_rnd.share.capital = 1 - consts.initial.hardware_rnd.share.cognitive;
      consts.initial.software_rnd.share.cognitive = 1 - consts.initial.software_rnd.share.experiments;

      consts.initial.hardware_rnd.share.compute = consts.initial.rnd.share.compute;
      consts.initial.software_rnd.share.compute = consts.initial.rnd.share.compute;

      let share = consts.initial.goods;
      for (let share of [consts.initial.goods.share, consts.initial.hardware_rnd.share, consts.initial.software_rnd.share]) {
        // Split cognitive share into compute and labour
        share.compute = share.compute * share.cognitive; // TERRIBLE HACK
        share.labour  = share.cognitive - share.compute;
      }

      // Returns to hardware and software need to be adjusted by the paralellization penalty
      consts.hardware_performance.returns /= consts.rnd.parallelization_penalty;
      consts.software.returns /= consts.rnd.parallelization_penalty;

      // Also, the capital substitution parameter for R&D
      consts.rnd.capital_substitution *= consts.rnd.parallelization_penalty;

      consts.hardware_rnd.capital_substitution = consts.rnd.capital_substitution;
      consts.hardware_rnd.labour_substitution = consts.rnd.labour_substitution;

      consts.software_rnd.capital_substitution = consts.software_rnd.experiments_substitution;
      consts.software_rnd.labour_substitution = consts.rnd.labour_substitution;

      consts.hardware_delay_idx = nj.even_round(consts.hardware_delay/consts.t_step);

      return consts;
    }

    static init_state_and_consts(state, consts) {
      let initial = consts.initial;

      //
      // Initialize R&D state

      // Hardware
      state.hardware_performance.cumulative_rnd_input = nj.div(
        nj.pow(initial.rnd_input_hardware, consts.rnd.parallelization_penalty),
        nj.mult(consts.initial.ratio_initial_to_cumulative_input_hardware_rnd, consts.rnd.parallelization_penalty)
      );

      state.buyable_hardware_performance = initial.buyable_hardware_performance;

      // Software
      state.software.v = initial.software;

      state.software.cumulative_rnd_input = nj.div(
        nj.pow(initial.rnd_input_software, consts.rnd.parallelization_penalty),
        nj.mult(consts.initial.ratio_initial_to_cumulative_input_software_rnd, consts.rnd.parallelization_penalty)
      );

      //
      // Initialize total inputs

      state.labour = initial.population;

      // Capital is initialized so that its rate of growth in the first time step matches the gwp rate of growth
      state.capital = initial.gwp * consts.investment_rate / (nj.exp(initial.capital_growth)-1);

      state.hardware = initial.hardware;

      state.compute = initial.hardware * initial.software;
      state.compute_investment = initial.hardware_production * consts.t_step / initial.buyable_hardware_performance;

      state.goods.tfp = initial.tfp_goods;
      state.rnd.tfp = initial.tfp_rnd;

      state.money_spent_training = initial.biggest_training_run / (initial.software * initial.buyable_hardware_performance);


      //
      // Initialize fractional inputs

      // Split of gwp between capital and compute
      state.frac_gwp.compute.v = state.compute_investment / initial.gwp / consts.t_step;

      // R&D fractional inputs
      state.frac_capital.hardware_rnd.v = initial.frac_capital.hardware_rnd;
      state.frac_capital.goods.v = 1 - state.frac_capital.hardware_rnd.v;

      state.frac_labour.hardware_rnd.v = initial.frac_labour.hardware_rnd;
      state.frac_labour.software_rnd.v = initial.frac_labour.software_rnd;
      state.frac_labour.goods.v = 1 - state.frac_labour.hardware_rnd.v - state.frac_labour.software_rnd.v;

      state.frac_compute.hardware_rnd.v = initial.frac_compute.hardware_rnd;
      state.frac_compute.software_rnd.v = initial.frac_compute.software_rnd;
      state.frac_compute.training.v = initial.biggest_training_run / state.compute;
      state.frac_compute.goods.v =
        1 - state.frac_compute.hardware_rnd.v
          - state.frac_compute.software_rnd.v
          - state.frac_compute.training.v;

      // Initial compute must be greater than initial training run
      if (initial.biggest_training_run > state.compute) {
        throw new Error("Initial biggest training run is bigger than available compute");
      }


      //
      // Initialize production inputs

      let _initialize_task_weight = (state, consts, category, item, capital=null, labour=null, compute=null) => {
        // Initialize task weights to match the initial economy share ratio

        if (capital == null) capital = state.capital * state.frac_capital[item].v;
        if (labour == null)  labour  = state.labour  * state.frac_labour[item].v;
        if (compute == null) compute = state.compute * state.frac_compute[item].v;

        let no_automation_labour_task_input = nj.zeros(consts[category].n_labour_tasks + 1);
        no_automation_labour_task_input.fill(labour / consts[category].n_labour_tasks, 1);

        let no_automation_compute_task_input = nj.zeros(consts[category].n_labour_tasks + 1);
        no_automation_compute_task_input[0] = compute;

        let share = initial[item].share;
        let initial_capital_to_cognitive_share_ratio = share.capital / share.cognitive;
        let initial_compute_to_labour_share_ratio    = share.compute / share.labour;

        let task_weights = this.adjust_task_weights(
          capital,
          no_automation_labour_task_input,
          no_automation_compute_task_input,
          this.get_task_compute_to_labour_ratio(consts[category].automation_runtime_flops, consts[category].automation_training_flops, consts.initial.biggest_training_run, consts, category),
          consts[item].capital_substitution,
          consts[category].labour_substitution,
          initial_capital_to_cognitive_share_ratio,
          initial_compute_to_labour_share_ratio,
        );

        consts[item].capital_task_weights = task_weights.capital;
        consts[item].labour_task_weights = task_weights.labour;
      }

      _initialize_task_weight(state, consts, 'goods', 'goods');
      _initialize_task_weight(state, consts, 'rnd', 'hardware_rnd');

      let experiments_compute = state.hardware ** consts.software_rnd.experiments_efficiency;
      consts.initial.software_rnd.share.capital = consts.initial.software_rnd.share.experiments; 
      _initialize_task_weight(state, consts, 'rnd', 'software_rnd', experiments_compute);

      state.bright_line = consts.initial.biggest_training_run;
    }

    static process_automation_costs(full_automation_flops, flop_gap, n_labour_tasks, steepness) {
      let automation_flops = this.quantiles_from_gap(full_automation_flops, flop_gap);
      automation_flops = this.interpolate_quantiles(automation_flops, n_labour_tasks);
      automation_flops = this.add_steepness(full_automation_flops, flop_gap, automation_flops, steepness);
      nj.insert(automation_flops, 0, 1.0); // The first task is always automatable

      return automation_flops;
    }

    static quantiles_from_gap(top, gap) {
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

    static interpolate_quantiles(quantiles, n_items) {
      // Interpolate a set of quantiles over linspace(0, 1, n_items)
      let values = [];

      let quantiles_keys = [...quantiles.keys()];

      for (let q of nj.linspace(0, 1, n_items)) {
        let prev_quantile = nj.max(quantiles_keys.filter(x => x <= q))
        let next_quantile = nj.min(quantiles_keys.filter(x => q <= x));

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

      values = nj.array(values);

      return values;
    }

    static add_steepness(full_requirements, gap, requirements, steepness) {
      /*
        Takes an array of requirements and converts it into a sum of units
        steps separated by `steepness` OOMs, but maintaining both ends of the
        FLOP gap
      */

      if (steepness == 0) return requirements;

      let gap_low = full_requirements/gap;
      let gap_high = full_requirements;

      let result = nj.zeros(requirements.length);
      for (let i = 0; i < result.length; i++) {
        let r = 10**(nj.log10(gap_low) + Math.ceil((nj.log10(requirements[i]) - nj.log10(gap_low))/steepness) * steepness);
        if (r > gap_high) r = gap_high;
        result[i] = r;
      }

      return result
    }

    static adjust_task_weights(
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

      let task_input = nj.add(labour_task_input, nj.mult(task_compute_to_labour_ratio, compute_task_input));

      let labour_share = nj.sum(nj.mult(labour_task_input, nj.pow(task_input, labour_substitution-1)));
      let compute_share = nj.sum(nj.mult(task_compute_to_labour_ratio, nj.mult(compute_task_input, nj.pow(task_input, labour_substitution-1))));

      let compute_to_labour_task_weight_ratio = compute_to_labour_share_ratio * labour_share / compute_share;

      let [compute_task_weight, labour_task_weight] = odds_to_probs(compute_to_labour_task_weight_ratio);

      let n_labour_tasks = labour_task_input.length-1;
      let inner_task_weights = nj.array([compute_task_weight].concat(Array(n_labour_tasks).fill(labour_task_weight)));

      // Compute outer task weights

      let cognitive_input = this.ces_production_function(task_input, inner_task_weights, labour_substitution);

      let capital_share = capital**capital_substitution;
      let cognitive_share = cognitive_input**capital_substitution;

      let capital_to_cognitive_task_weight_ratio = capital_to_cognitive_share_ratio * cognitive_share / capital_share;

      let [capital_task_weight, cognitive_task_weight] = odds_to_probs(capital_to_cognitive_task_weight_ratio);


      let outer_task_weights = nj.array([capital_task_weight, cognitive_task_weight])

      return {capital: outer_task_weights, labour: inner_task_weights};
    }

    // ------------------------------------------------------------------------
    // Simulation functions
    // ------------------------------------------------------------------------

    // ------------------------------------------------------------------------
    // Reinvest outputs in inputs
    //

    static reinvest_output_in_inputs(state, consts, states) {
      this.update_rnd_state(state, consts, states);
      this.update_rampup(state, consts, states);
      this.allocate_frac_inputs(state, consts, states);
      this.calculate_total_inputs(state, consts, states);
      this.limit_frac_inputs(state, consts, states);
      this.calculate_tfps(state, consts, states);
      this.calculate_money_spent_training(state, consts, states);
    }

    static update_rnd_state(state, consts, states) {
      function _update_rnd(
          current_performance,
          initial_performance,
          research_input,
          cumulative_adjusted_input,
          returns,
          performance_ceiling,
          consts,
        ) {
        let adjusted_input = research_input ** consts.rnd.parallelization_penalty;
        let new_cumulative_adjusted_input = cumulative_adjusted_input + adjusted_input*consts.t_step;
        let growth_in_cumulative_inputs = new_cumulative_adjusted_input / cumulative_adjusted_input;
        let ceiling_penalty =
          (nj.log10(performance_ceiling) - nj.log10(current_performance)) /
          (nj.log10(performance_ceiling) - nj.log10(initial_performance));
        let performance_growth_rate = growth_in_cumulative_inputs**(returns * ceiling_penalty);
        let new_performance = Math.min(current_performance * performance_growth_rate, performance_ceiling);

        return {new_performance, new_cumulative_adjusted_input, ceiling_penalty};
      }

      // In the first time step, we move forward the buyable hardware performance to adjust for the delay in hardware performance
      if (state.t_idx == 1) {
        let result = _update_rnd(
          consts.initial.buyable_hardware_performance,
          consts.initial.buyable_hardware_performance,
          state.hardware_performance.rnd_input,
          state.hardware_performance.cumulative_rnd_input,
          consts.hardware_performance.returns,
          consts.hardware_performance.ceiling,
          consts,
        );

        let improved_hardware_performance = result.new_performance;

        let initial_hardware_improvement_rate = improved_hardware_performance / consts.initial.buyable_hardware_performance;

        consts.initial.hardware_performance = consts.initial.buyable_hardware_performance * initial_hardware_improvement_rate**consts.hardware_delay_idx;

        state.hardware_performance.v = consts.initial.hardware_performance;
        states[0].hardware_performance.v = consts.initial.hardware_performance;

        states[0].hardware_rnd.ceiling_penalty = 1;
        states[0].software_rnd.ceiling_penalty = 1;
      }

      let hardware_result = _update_rnd(
        state.hardware_performance.v,
        states[0].hardware_performance.v,
        state.hardware_performance.rnd_input,
        state.hardware_performance.cumulative_rnd_input,
        consts.hardware_performance.returns,
        consts.hardware_performance.ceiling,
        consts,
      );
      state.hardware_performance.v = hardware_result.new_performance;
      state.hardware_performance.cumulative_rnd_input = hardware_result.new_cumulative_adjusted_input;
      state.hardware_rnd.ceiling_penalty = hardware_result.ceiling_penalty;

      let software_result = _update_rnd(
        state.software.v,
        states[0].software.v,
        state.software.rnd_input,
        state.software.cumulative_rnd_input,
        consts.software.returns,
        consts.software.ceiling,
        consts,
      );
      state.software.v = software_result.new_performance;
      state.software.cumulative_rnd_input = software_result.new_cumulative_adjusted_input;
      state.software_rnd.ceiling_penalty = software_result.ceiling_penalty;
    }

    static update_rampup(state, consts) {
      state.rampup = this.get_frac_tasks_automated(state.goods, consts.goods) >= consts.rampup_trigger;
    }

    static allocate_frac_inputs(state, consts, states) {
      function _update_frac_input(x, state, consts) {
        let frac = x.v * nj.exp(consts.t_step * (state.rampup ? x.growth_rampup : x.growth));
        frac = Math.min(frac, x.ceiling);
        x.v = frac;
      }

      _update_frac_input(state.frac_gwp.compute, state, consts);

      _update_frac_input(state.frac_compute.training, state, consts);

      _update_frac_input(state.frac_capital.hardware_rnd, state, consts);
      _update_frac_input(state.frac_labour.hardware_rnd, state, consts);
      _update_frac_input(state.frac_compute.hardware_rnd, state, consts);

      _update_frac_input(state.frac_capital.software_rnd, state, consts);
      _update_frac_input(state.frac_labour.software_rnd, state, consts);
      _update_frac_input(state.frac_compute.software_rnd, state, consts);

      state.frac_capital.goods.v = 1 - state.frac_capital.hardware_rnd.v - state.frac_capital.software_rnd.v;
      state.frac_labour.goods.v  = 1 - state.frac_labour.hardware_rnd.v  - state.frac_labour.software_rnd.v;
      state.frac_compute.goods.v =
        1 - state.frac_compute.hardware_rnd.v
          - state.frac_compute.software_rnd.v
          - state.frac_compute.training.v;
    }

    static calculate_total_inputs(state, consts, states) {
      // Compute inputs
      state.compute_investment = state.gwp * state.frac_gwp.compute.v * consts.t_step;

      let t_diff = state.t_idx - consts.hardware_delay_idx;
      if (t_diff >= 0) {
        state.buyable_hardware_performance = states[t_diff].hardware_performance.v;
      } else {
        state.buyable_hardware_performance = states[0].hardware_performance.v * (states[1].hardware_performance.v/states[0].hardware_performance.v)**t_diff;
      }

      let new_hardware = state.compute_investment * state.buyable_hardware_performance;

      state.hardware = state.hardware * ((1.-consts.compute_depreciation)**consts.t_step) + new_hardware;
      state.compute = state.hardware * state.software.v;

      // Non compute inputs
      let capital_investment = state.gwp * consts.investment_rate * consts.t_step
      state.capital = state.capital + capital_investment;

      state.labour = state.labour * nj.exp(consts.labour_growth * consts.t_step)
    }

    static limit_frac_inputs(state, consts, states) {
      // Cap the growth of the biggest training run
      state.bright_line = consts.initial.biggest_training_run * 10**(consts.bright_line_growth_rate * (state.t_year - consts.t_start));
      state.frac_compute.training.v = Math.min(state.frac_compute.training.v, state.bright_line / state.compute);

      // Cap the growth of the fraction of FLOP before rampup
      if (state.money_spent_training > consts.money_cap_training_before_wakeup && !states[state.t_idx-1].rampup) {
        state.frac_compute.training.v = states[state.t_idx-1].frac_compute.training.v;
      }

      state.frac_compute.goods.v =
        1 - state.frac_compute.hardware_rnd.v
          - state.frac_compute.software_rnd.v
          - state.frac_compute.training.v;
    }

    static calculate_tfps(state, consts) {
      state.goods.tfp *= nj.exp(consts.tfp_growth * consts.t_step)
      state.rnd.tfp   *= nj.exp(consts.tfp_growth * consts.t_step)
    }

    static calculate_money_spent_training(state, consts) {
      state.money_spent_training = state.compute * state.frac_compute.training.v / (state.software.v * state.buyable_hardware_performance);
    }

    // ------------------------------------------------------------------------
    // Task automation
    //

    static automate_tasks(state, consts) {
      state.biggest_training_run = state.compute * state.frac_compute.training.v;

      if (state.biggest_training_run > 1.00001 * state.bright_line) {
        console.error('Bright line violated', state.biggest_training_run, state.bright_line);
      }

      let _update_automatable_tasks = (s, p, category) => {
        s.task_compute_to_labour_ratio = this.get_task_compute_to_labour_ratio(p.automation_runtime_flops, p.automation_training_flops, state.biggest_training_run, consts, category);
        s.automatable_tasks = nj.count_true(nj.lt(p.automation_training_flops, nj.mult(state.biggest_training_run, consts.runtime_training_max_tradeoff)));
      }

      _update_automatable_tasks(state.goods, consts.goods, 'goods');
      _update_automatable_tasks(state.rnd, consts.rnd, 'rnd');
      state.hardware_rnd.task_compute_to_labour_ratio = state.rnd.task_compute_to_labour_ratio; // TODO hack
      state.software_rnd.task_compute_to_labour_ratio = state.rnd.task_compute_to_labour_ratio; // TODO hack
    }

    static get_task_compute_to_labour_ratio(automation_runtime_flops, automation_training_flops, biggest_training_run, consts, category) {
      let runtime_requirements = nj.mult(automation_runtime_flops, nj.pow(nj.div(automation_training_flops, biggest_training_run), consts.runtime_training_tradeoff));
      runtime_requirements = nj.maximum(1, runtime_requirements);
      let task_compute_to_labour_ratio = nj.div(1, runtime_requirements);

      return task_compute_to_labour_ratio;
    }

    // ------------------------------------------------------------------------
    // Production
    //

    static production(state, consts, states) {
      this.produce(state, consts, 'goods', 'goods');
      this.produce(state, consts, 'rnd', 'hardware_rnd');

      state.software_rnd.experiments_compute = state.hardware ** consts.software_rnd.experiments_efficiency;
      this.produce(state, consts, 'rnd', 'software_rnd', state.software_rnd.experiments_compute, true);

      this.post_production(state, consts, states);
    }

    static produce(state, consts, category, item, capital=null, apply_tfp_to_cognitive_outputs=false) {
      if (capital == null) capital = state.capital * state.frac_capital[item].v;
      state[item].capital = capital;
      state[item].labour  = state.labour  * state.frac_labour[item].v;
      state[item].compute = state.compute * state.frac_compute[item].v;

      state[category].at = this.get_automatable_task_count(state, consts, category) + 1;

      let r = this.solve_allocation(
        state[item].labour,
        state[item].compute,
        consts[item].labour_task_weights,
        consts[category].labour_substitution,
        state[category].task_compute_to_labour_ratio,
        state[category].at,
      );
      state[item].labour_task_input = r.labour;
      state[item].compute_task_input = r.compute;

      state[item].task_input = nj.add(state[item].labour_task_input, nj.mult(state[category].task_compute_to_labour_ratio, state[item].compute_task_input));

      // Cognitive outputs
      let cognitive_output = this.ces_production_function(
        state[item].task_input,
        consts[item].labour_task_weights,
        consts[category].labour_substitution,
      );

      state[item].cognitive_output = cognitive_output;

      if (apply_tfp_to_cognitive_outputs) {
        cognitive_output *= state[category].tfp;
      }

      // Final outputs
      state[item].output = this.ces_production_function(
        [state[item].capital, cognitive_output],
        consts[item].capital_task_weights,
        consts[item].capital_substitution,
      );

      if (!apply_tfp_to_cognitive_outputs) {
        state[item].output *= state[category].tfp;
      }
    }

    static post_production(state, consts, states) {
      // GWP
      let output_to_gwp_factor = consts.initial.gwp / states[0].goods.output;
      state.gwp = state.goods.output * output_to_gwp_factor;

      // Compute selected shares
      state.goods.compute_share = this.compute_shares(
        state.goods.capital,
        state.goods.labour_task_input,
        state.goods.compute_task_input,
        consts.goods.capital_task_weights,
        consts.goods.labour_task_weights,
        state.goods.task_compute_to_labour_ratio,
        consts.goods.capital_substitution,
        consts.goods.labour_substitution,
      ).compute_share;

      // Automation multiplier (compute how much worse is the hardware output without automation)

      let {labour: no_automation_labour_task_input_rnd, compute: no_automation_compute_task_input_rnd} = this.solve_allocation(
        state.hardware_rnd.labour,
        state.hardware_rnd.compute,
        consts.hardware_rnd.labour_task_weights,
        consts.rnd.labour_substitution,
        state.rnd.task_compute_to_labour_ratio,
        1 // Only first task is automatable
      );

      let no_automation_task_input_hardware_rnd =
        nj.add(no_automation_labour_task_input_rnd, nj.mult(state.rnd.task_compute_to_labour_ratio, no_automation_compute_task_input_rnd));

      let no_automation_output = this.nested_ces_production_function(
        state.hardware_rnd.capital, no_automation_task_input_hardware_rnd,
        consts.hardware_rnd.capital_task_weights, consts.hardware_rnd.labour_task_weights,
        consts.rnd.capital_substitution, consts.rnd.labour_substitution,
        state.rnd.tfp
      );

      state.hardware_rnd.automation_multiplier = state.hardware_rnd.output / no_automation_output;

      // Compute inputs to hardware R&D
      let rnd_input_to_hardware_performance_investment_factor = consts.initial.rnd_input_hardware / states[0].hardware_rnd.output;
      state.hardware_performance.rnd_input = state.hardware_rnd.output * rnd_input_to_hardware_performance_investment_factor;

      // Compute inputs to sotware R&D
      let rnd_input_to_software_investment_factor = consts.initial.rnd_input_software / states[0].software_rnd.output;
      state.software.rnd_input = state.software_rnd.output * rnd_input_to_software_investment_factor;
    };

    // ------------------------------------------------------------------------
    // Simulation utilities
    //

    static ces_production_function(inputs, alphas, rho, tfp=1) {
      let result = tfp*nj.sum(nj.div(nj.mult(alphas, nj.pow(inputs, rho)), nj.sum(alphas)))**(1./rho);
      return result;
    }

    static nested_ces_production_function(
       capital, cognitive_inputs,
       outer_weights, inner_weights,
       outer_rho, inner_rho,
       tfp = 1)
    {
      let result = tfp * this.ces_production_function(
        nj.array([capital, this.ces_production_function(cognitive_inputs, inner_weights, inner_rho)]),
        outer_weights,
        outer_rho
      );

      return result;
    }

    static solve_allocation(L, C, beta, rho, eta, AT) {
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

      let sums_beta = nj.zeros(N + 1); // sums_beta[i] = sum(beta[i:]**sigma)
      {
        let sum = 0;
        for (let i = N; i >= 0; i--) {
          if (i < N) sum += beta[i]**sigma;
          sums_beta[i] = sum;
        }
      }

      let sums_beta_eta = nj.zeros(N + 1); // sums_beta_eta[i] = sum(beta[:i]**sigma * eta[:i]**(sigma-1))
      {
        let sum = 0;
        for (let i = 0; i <= N; i++) {
          sums_beta_eta[i] = sum;
          if (i < N) sum += ((beta[i]*eta[i])**sigma)/eta[i];
        }
      }

      let labour_input_task = nj.zeros(N);
      let compute_input_task = nj.zeros(N);

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

          if (I > 0 && compute_input_task[I] < 0) {
            for (let i = I; i < N; i++) {
              compute_input_task[i] = 0;
            }

            for (let i = 0; i < I; i++) {
              compute_input_task[i] = C * beta[i]**sigma * eta[i]**(sigma-1) / sums_beta_eta[I];
            }

            for (let i = 0; i < I; i++) {
              labour_input_task[i] = 0;
            }

            for (let i = I; i < N; i++) {
              labour_input_task[i] = L * (beta[i]**sigma / sums_beta[I]);
            }
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
      if (nj.all_equals(labour_input_task, 0)) {
        labour_input_task[labour_input_task.length-1] = L;
      }

      return {labour: labour_input_task, compute: compute_input_task};
    }

    static compute_shares(
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
      let task_input = nj.add(labour_task_input, nj.mult(compute_task_input, task_compute_to_labour_ratio));

      let cognitive_input = this.ces_production_function(
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
      let labour_share = nj.sum(nj.mult(labour_task_weights, nj.mult(labour_task_input, nj.pow(task_input, labour_substitution-1))));

      let compute_share = nj.sum(nj.mult(labour_task_weights, nj.mult(nj.mult(compute_task_input, task_compute_to_labour_ratio), nj.pow(task_input, labour_substitution-1))));

      let cognitive_sum = labour_share + compute_share;
      labour_share /= cognitive_sum
      compute_share /= cognitive_sum

      labour_share *= cognitive_share
      compute_share *= cognitive_share

      return {capital_share, cognitive_share, labour_share, compute_share};
    }

    static get_automated_tasks_count(statex) {
      let count = nj.count_true(nj.gt(
        nj.mult(statex.task_compute_to_labour_ratio, statex.compute_task_input),
        nj.mult(10, statex.labour_task_input)
      )) - 1; // -1 to account for the initial task
      return count;
    }

    static get_frac_tasks_automated(statex, constsx) {
      return this.get_automated_tasks_count(statex) / constsx.n_labour_tasks;
    }

    static get_frac_automatable_tasks(state, consts, category, with_tradeoff=true) {
      return this.get_automatable_task_count(state, consts, category, with_tradeoff) / consts[category].n_labour_tasks;
    }

    static get_automatable_task_count(state, consts, category, with_tradeoff=true) {
      // -1 to account for the initial task
      return nj.count_true(
        nj.lt(consts[category].automation_training_flops, state.biggest_training_run * (with_tradeoff ? consts.runtime_training_max_tradeoff : 1))
      ) - 1;
    }

    // -------------------------------------------------------------------------
    // Misc
    // -------------------------------------------------------------------------

    time_to_index(t) {
      return nj.even_round((t - this.consts.t_start) / this.consts.t_step);
    }

    index_to_time(i) {
      return this.consts.t_start + i * this.consts.t_step;
    }

    post_simulation() {
      let rampup_start = null;
      let rampup_mid = null;
      let agi_year = null;
      let sub_agi_year = null;

      for (let i = 0; i < this.states.length; i++) {
        let state = this.states[i];
        let prev_state = (i < 1) ? null : this.states[i-1];

        let year = state.t_year - this.consts.t_step;
        if (rampup_start == null && state.rampup) {
          rampup_start = year;
        }

        if (rampup_mid == null && prev_state && Model.get_frac_tasks_automated(prev_state.goods, this.consts.goods) >= 0.2) {
          rampup_mid = year;
        }

        if (sub_agi_year == null && prev_state && Model.get_frac_automatable_tasks(prev_state, this.consts, 'goods', false) >= 0.2) {
          sub_agi_year = year;
        }

        if (agi_year == null && prev_state && Model.get_frac_automatable_tasks(prev_state, this.consts, 'goods', false) >= 1) {
          agi_year = year;
        }
      }

      // Compute doubling times
      let doubling_times = [this.consts.t_step / Math.log2(this.states[1].gwp/this.states[0].gwp)]
      if (rampup_start != null) {
        let reference_idx = this.time_to_index(rampup_start);
        for (let i = reference_idx; i < this.states.length; i++) {
          if (this.states[i].gwp > 2*this.states[reference_idx].gwp) {
            doubling_times.push(this.index_to_time(i) - this.index_to_time(reference_idx))
            reference_idx = i;
          }
        }
      }

      // Round doubling times
      for (let i = 0; i < doubling_times.length; i++) {
        doubling_times[i] = +doubling_times[i].toFixed(2);
      }

      // Fraction of tasks automated
      let frac_tasks_automated_goods = nj.array(this.states.length);
      for (let i = 0; i < this.states.length; i++) {
        frac_tasks_automated_goods[i] = Model.get_frac_tasks_automated(this.states[i].goods, this.consts.goods);
      }

      let frac_tasks_automated_rnd = nj.array(this.states.length);
      for (let i = 0; i < this.states.length; i++) {
        frac_tasks_automated_rnd[i] = Model.get_frac_tasks_automated(this.states[i].hardware_rnd, this.consts.rnd);
      }

      this.doubling_times = doubling_times.slice(0, 5); // We are only interested in the first five doubling times
      this.rampup_start = rampup_start;
      this.rampup_mid   = rampup_mid;
      this.agi_year     = agi_year;
      this.sub_agi_year = sub_agi_year;

      this.frac_tasks_automated_goods = frac_tasks_automated_goods;
      this.frac_tasks_automated_rnd = frac_tasks_automated_rnd;
    }

    // -------------------------------------------------------------------------
    // Functions for inspecting the simulation results
    // -------------------------------------------------------------------------

    get_thread(path) {
      let thread = [];
      for (let state of this.states) {
        thread.push(this.get_by_path(state, path));
      }
      return thread;
    }

    get_by_path(state, path) {
      let fields = path.split('.');
      let x = state;
      for (let f of fields) {
        x = x[f];
      }
      return x;
    }

    get_growth(path_or_thread, growth_type='log') {
      let thread = (typeof path_or_thread == 'string') ? this.get_thread(path_or_thread) : path_or_thread;

      let steps_per_year = Math.floor(1 / this.consts.t_step);
      let growth = nj.array(this.states.length - steps_per_year);
      let t = nj.array(this.states.length - steps_per_year);
      for (let i = steps_per_year; i < this.states.length; i++) {
        t[i-steps_per_year] = this.states[i].t_year;

        let g;
        if (growth_type == 'linear') g = thread[i] - thread[i-steps_per_year];
        if (growth_type == 'log')    g = Math.log(thread[i]/thread[i-steps_per_year]);
        if (growth_type == 'log10')  g = Math.log10(thread[i]/thread[i-steps_per_year]);

        growth[i-steps_per_year] = g;
      }

      return {t, growth};
    }

    length_between_thresholds(series1, series2) {
      // Utility function to measure the amount of time between two thresholds being crossed
      if (!nj.any(series1) || !nj.any(series2)) {
        return nj.nan;
      }
      let idx1 = nj.argmax(series1);
      let idx2 = nj.argmax(series2);
      return (idx2 - idx1) * this.consts.t_step;
    }

    time_when(series) {
      if (!nj.any(series)) {
        return nj.nan;
      }
      let idx = nj.argmax(series);
      return this.index_to_time(idx);
    }

    // Compute timeline metrics
    get_timeline_metrics() {
      let timeline_metrics = {};

      timeline_metrics['automation_gns_20%'] = this.time_when(nj.gte(this.frac_tasks_automated_goods, 0.2));
      timeline_metrics['automation_gns_100%'] = this.time_when(nj.gte(this.frac_tasks_automated_goods, 1.0));

      timeline_metrics['sub_agi_year'] = this.sub_agi_year;
      timeline_metrics['agi_year']     = this.agi_year;

      timeline_metrics['automation_rnd_20%'] = this.time_when(nj.gte(this.frac_tasks_automated_rnd, 0.2));
      timeline_metrics['automation_rnd_100%'] = this.time_when(nj.gte(this.frac_tasks_automated_rnd, 1.0));

      timeline_metrics['rampup_start'] = this.rampup_start;

      return timeline_metrics;
    }

    // Compute takeoff metrics
    get_takeoff_metrics() {
      let takeoff_metrics = {};

      // Time from AI that can perform 20% of tasks to AI that can perform 100%.

      takeoff_metrics["full_automation_gns"] = this.length_between_thresholds(
          nj.gt(this.frac_tasks_automated_goods, 0.2),
          nj.gte(this.frac_tasks_automated_goods, 1.),
      );

      takeoff_metrics["full_automation_rnd"] = this.length_between_thresholds(
          nj.gt(this.frac_tasks_automated_rnd, 0.2),
          nj.gte(this.frac_tasks_automated_rnd, 1.),
      );

      // Years from "total cognitive output is 2X human cognitive output" to "total cognitive output is 10X human cognitive output"
      let rnd_automation_multiplier = this.get_thread('hardware_rnd.automation_multiplier');
      takeoff_metrics["cog_output_multiplier"] = this.length_between_thresholds(nj.gt(rnd_automation_multiplier, 2), nj.gt(rnd_automation_multiplier, 10));

      // Time from powerful sub-AGI to AGI
      takeoff_metrics['sub_agi_to_agi'] = (this.agi_year != null) ? this.agi_year - this.sub_agi_year : nj.nan;

      // Time from 5% GWP growth to 20% GWP growth
      let steps_per_year = Math.floor(1 / this.consts.t_step);
      let gwp_growth = nj.array(this.states.length - steps_per_year);
      for (let i = 0; i < this.states.length - steps_per_year; i++) {
        gwp_growth[i] = Math.log(this.states[i+steps_per_year].gwp/this.states[i].gwp);
      }
      takeoff_metrics["gwp_growth"] = this.length_between_thresholds(nj.gt(gwp_growth, 0.05), nj.gt(gwp_growth, 0.20));

      takeoff_metrics['doubling_times'] = this.doubling_times;

      for (let k in takeoff_metrics) {
        takeoff_metrics[k] = [takeoff_metrics[k]];
      }

      return takeoff_metrics;
    }

    get_summary_table() {
      let rows = [];

      let prerampup = (this.rampup_start == null) ? null : (this.consts.t_start + this.rampup_start)/2;

      let frac_tasks_automated_goods = nj.array(this.states.length);
      for (let i = 0; i < this.states.length; i++) {
        frac_tasks_automated_goods[i] = Model.get_frac_tasks_automated(this.states[i].goods, this.consts.goods);
      }

      let frac_tasks_automated_rnd = nj.array(this.states.length);
      for (let i = 0; i < this.states.length; i++) {
        frac_tasks_automated_rnd[i] = Model.get_frac_tasks_automated(this.states[i].hardware_rnd, this.consts.rnd);
      }

      let raw_metrics = {
        'biggest_training_run':       this.get_thread('biggest_training_run'),
        'frac_tasks_automated_goods': frac_tasks_automated_goods,
        'frac_tasks_automated_rnd':   frac_tasks_automated_rnd,
      };

      let doubling_time_metrics = {
        'hardware_performance':          this.get_thread('hardware_performance.v'),
        'software':                      this.get_thread('software.v'),
        'compute_investment':            this.get_thread('compute_investment'),
        'frac_compute_training':         this.get_thread('frac_compute.training.v'),
        'gwp':                           this.get_thread('gwp'),
        'capital':                       this.get_thread('capital'),
        'labour':                        this.get_thread('labour'),
        'tfp_rnd':                       this.get_thread('rnd.tfp'),
        'rnd_input_software':            this.get_thread('software.rnd_input'),
        'cumulative_rnd_input_software': this.get_thread('software.cumulative_rnd_input'),
      };

      for (let [period, t] of Object.entries({'Pre wake-up': prerampup,
                        'Wake-up': this.rampup_start,
                        'Mid rampup': this.rampup_mid,
                        'Full economic automation': this.timeline_metrics['automation_gns_100%']})) {

        if (t == null) {
          let summary_row = {
            'period' : period,
            'year' : nj.nan,
          };

          for (let raw_metric in raw_metrics) {
            summary_row[raw_metric] = nj.nan;
          }

          for (let doubling_time_metric in doubling_time_metrics) {
            summary_row[`${doubling_time_metric} growth rate`] = nj.nan;
            summary_row[`${doubling_time_metric} doubling time`] = nj.nan;
          }

          rows.push(summary_row);
          continue;
        }

        let idx = this.time_to_index(t);
        t = this.index_to_time(idx)
        let t_end = t + 1;
        let idx_end = this.time_to_index(t_end);

        if (idx_end >= this.states.length) {
          let diff = idx_end - (this.states.length - 1);
          idx -= diff;
          idx_end -= diff;
        }

        let summary_row = {
          'period' : period,
          'year' : t,
        }

        // Auxiliary functions
        let dt = s => (Math.log2(s[idx_end]/s[idx]) != 0) ? 1 / Math.log2(s[idx_end]/s[idx]) : nj.nan;
        let gr = s => Math.log(s[idx_end] / s[idx]);

        for (let raw_metric in raw_metrics) {
          summary_row[raw_metric] = raw_metrics[raw_metric][idx];
        }

        for (let doubling_time_metric in doubling_time_metrics) {
          summary_row[`${doubling_time_metric} growth rate`]   = gr(doubling_time_metrics[doubling_time_metric]);
          summary_row[`${doubling_time_metric} doubling time`] = dt(doubling_time_metrics[doubling_time_metric]);
        }

        rows.push(summary_row)
      }

      let summary_table = {};
      for (let row of rows) {
        for (let col in row) {
          if (!(col in summary_table)) summary_table[col] = [];
          summary_table[col].push(row[col]);
        }
      }

      return summary_table;
    }
  }

  class SimulationState {
    constructor(consts) {
      this.t_idx = 0;
      this.t_year = 0;

      this.rampup = false;

      this.automation_training_flops = 0;

      this.capital = 0;
      this.labour  = 0;
      this.compute = 0;

      this.frac_capital = SimulationState.create_frac_state(['goods', 'hardware_rnd', 'software_rnd']);
      this.frac_labour  = SimulationState.create_frac_state(['goods', 'hardware_rnd', 'software_rnd']);
      this.frac_compute = SimulationState.create_frac_state(['goods', 'hardware_rnd', 'software_rnd', 'training',]);
      this.frac_gwp = SimulationState.create_frac_state(['compute']);

      this.goods = { tfp: 0, };
      this.hardware_rnd = {};
      this.software_rnd = {};

      this.rnd = {
        tfp: 0,
        automation_runtime_flops: 0,
        automation_training_flops: 0,
        automation_multiplier: 0,
      };

      this.hardware_performance = {
        v: 0,
        rnd_input: 0,
      };

      this.software = {
        v: 0,
        rnd_input: 0,
      };

      this.init_task_state(this.goods, consts.goods.n_labour_tasks);
      this.goods.task_compute_to_labour_ratio = nj.zeros(consts.goods.n_labour_tasks + 1);

      this.init_task_state(this.hardware_rnd, consts.rnd.n_labour_tasks);
      this.init_task_state(this.software_rnd, consts.rnd.n_labour_tasks);
      this.rnd.task_compute_to_labour_ratio = nj.zeros(consts.rnd.n_labour_tasks + 1);

      this.frac_capital.hardware_rnd = cheap_deep_copy(consts.frac_capital.hardware_rnd);

      this.frac_labour.hardware_rnd = cheap_deep_copy(consts.frac_labour.hardware_rnd);
      this.frac_labour.software_rnd = cheap_deep_copy(consts.frac_labour.software_rnd);

      this.frac_compute.hardware_rnd = cheap_deep_copy(consts.frac_compute.hardware_rnd);
      this.frac_compute.software_rnd = cheap_deep_copy(consts.frac_compute.software_rnd);
      this.frac_compute.training = cheap_deep_copy(consts.frac_compute.training);

      this.frac_gwp.compute = cheap_deep_copy(consts.frac_gwp.compute);
    }

    init_task_state(statex, n) {
      statex.task_input = nj.zeros(n + 1);
      statex.labour_task_input = nj.zeros(n + 1);
      statex.compute_task_input = nj.zeros(n + 1);
    }

    static create_frac_state(outputs) {
      let frac_state = {};

      for (let out of outputs) {
        frac_state[out] = {
          v: 0,
          growth: 0,
          growth_rampup: 0,
          ceiling: 0,
        };
      }

      return frac_state;
    }
  }

  // ---------------------------------------------------------------------------
  // Bioanchors (for comparison)
  // ---------------------------------------------------------------------------

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

    let hardware_growth = nj.pow(2, 1/hardware_doubling_time);
    let software_growth = nj.pow(2, 1/software_doubling_time);
    let fast_training_investment_growth = nj.pow(2, 1/fast_training_investment_doubling_time);

    let timesteps = nj.arange(t_start, t_end, t_step);

    let slow_training_investment_start_index = nj.round((t_initial + fast_training_investment_duration)/t_step - t_start);

    let fast_training_investment = nj.mult(
        initial_training_investment,
        nj.pow(fast_training_investment_growth, nj.sub(timesteps.slice(0, slow_training_investment_start_index), t_initial))
    );
    let slow_training_investment = nj.mult(
        fast_training_investment[fast_training_investment.length-1],
        nj.pow(slow_training_investment_growth, nj.sub(timesteps.slice(slow_training_investment_start_index), timesteps[slow_training_investment_start_index-1]))
    );
    let training_investment = nj.concatenate(fast_training_investment, slow_training_investment)

    let hardware = nj.mult(initial_hardware, nj.pow(hardware_growth, nj.sub(timesteps, t_initial)));

    let software_timesteps = nj.arange(software_doubling_start, t_end, t_step);
    let software = nj.mult(
       initial_software,
       nj.pow(software_growth, nj.sub(software_timesteps, software_doubling_start))
    );
    software = nj.minimum(software_ceiling, software);

    return {timesteps, training_investment, hardware, software_timesteps, software};
  }

  // ---------------------------------------------------------------------------
  // Utilities
  // ---------------------------------------------------------------------------

  function odds_to_probs(o) {
    // Stable implementation of conversion between odds and probs

    let p;
    let p_not;

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

  // ---------------------------------------------------------------------------
  // Exports
  // ---------------------------------------------------------------------------

  ftm.run_simulation = run_simulation;
  ftm.run_bioanchors_model = run_bioanchors_model;
  ftm.Model = Model;
}
