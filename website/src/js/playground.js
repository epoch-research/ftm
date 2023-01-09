/*******************************************************************************

Dependencies:
  * ./nj.js
  * ./ftm.js
  * d3.js
  * tippy.js

*******************************************************************************/

let best_guess_sim = null;
let sim;

let ui_state = {
  // Graphs
  show_bioanchors: false,
  show_normalized_decomposition: false,

  decomposition_graph_transform: null,
  metrics_graph_transform: null,

  // Tables
  metrics_to_show: 'important-metrics',

  takeoff_table_scroll_x: 0,
  summary_table_scroll_x: 0,

  // TODO Generalize these
  metrics_graph_top_selection: 'raw metric',
  metrics_graph_side_selection: 'GWP',
};

let metrics_to_show = document.getElementById('metrics-to-show');
metrics_to_show.addEventListener('change',  () => {
  ui_state.metrics_to_show = metrics_to_show.value;

  let all_metrics_table = {
    ...sim.get_timeline_metrics(),
    ...sim.get_takeoff_metrics(),
  };

  // Round the metrics
  for (let metric in all_metrics_table) {
    let v = all_metrics_table[metric];
    if (typeof v == 'number') {
      all_metrics_table[metric] = v.toFixed(1);
    } else if (Array.isArray(v)) {
      let new_v = [];
      for (let x of v) {
        if (typeof x == 'number') x = x.toFixed(1);
        new_v.push(x);
      }
      all_metrics_table[metric] = new_v;
    }
  }

  let metrics_table;

  if (ui_state.metrics_to_show == 'important-metrics') {
    let important_metrics = ['automation_gns_100%', 'full_automation_gns', 'full_automation_rnd'];

    metrics_table = {};
    for (let key of important_metrics) metrics_table[key] = all_metrics_table[key];
  } else {
    metrics_table = all_metrics_table;
  }

  clear_tables('#takeoff-metrics-table-container');
  let table_wrapper = add_table('#takeoff-metrics-table-container', metrics_table);
  table_wrapper.scrollLeft = ui_state.takeoff_table_scroll_x;
  table_wrapper.addEventListener('scroll', () => {
    ui_state.takeoff_table_scroll_x = table_wrapper.scrollLeft;
  });

  injectMeaningTooltips();
});

function run_simulation(immediate, callback) {
  let params = get_parameters();
  if (params) {
    if (!params['runtime_training_tradeoff_enabled']) {
      params['runtime_training_tradeoff'] = 0;
    }
    delete params['runtime_training_tradeoff_enabled'];

    document.body.classList.add('running');
    cancelBackgroundProcesses();
    dispatchBackgroundProcess(() => {
      let js_params = transform_python_to_js_params(params);
      sim = ftm.run_simulation(js_params);

      if (callback) {
        callback(sim);
      }

      let t = sim.get_thread('t_year');
      let indices = sim.get_thread('t_idx');
      //let b = sim.get_thread('biggest_training_run');
      let b = sim.get_thread('compute');

      metrics_to_show.value = ui_state.metrics_to_show;
      metrics_to_show.dispatchEvent(new Event('change'));

      clear_tables('#summary-table-container');
      clear_tables('#year-by-year-table-container');

      let summary_table_wrapper = add_table('#summary-table-container', sim.get_summary_table());
      summary_table_wrapper.scrollLeft = ui_state.summary_table_scroll_x;
      summary_table_wrapper.addEventListener('scroll', () => {
        ui_state.summary_table_scroll_x = summary_table_wrapper.scrollLeft;
      });

      let detailed_table = {
        'Year':                 t,
        'Rampup':               sim.get_thread('rampup'),
        'Hardware performance': sim.get_thread('hardware_performance.v'),
        'Compute investment':   sim.get_thread('frac_gwp.compute.v'),
        'Hardware':             sim.get_thread('hardware'),
        'Software':             sim.get_thread('software.v'),
        'Compute':              sim.get_thread('compute'),
        'Labour':               sim.get_thread('labour'),
        'Capital':              sim.get_thread('capital'),
        'GWP':                  sim.get_thread('gwp'),
        'Biggest training run': sim.get_thread('biggest_training_run'),
      };

      let yearly_table = {};

      let prev_year = null;
      for (let i of indices) {
        let year = Math.floor(t[i]);
        if (year != prev_year) {
          for (let v in detailed_table) {
            if (!(v in yearly_table)) yearly_table[v] = [];
            yearly_table[v].push((v == 'Year') ? year : detailed_table[v][i]);
          }
          prev_year = year;
        }
      }

      add_table('#year-by-year-table-container', yearly_table);

      injectMeaningTooltips();

      plt.clear('#metrics-graph-container');
      plt.clear('#compute-decomposition-graph-container');

      plt.set_defaults({
        yscale: 'log',
      });

      plot_compute_decomposition(sim, '#compute-decomposition-graph-container');

      let frac_automated_tasks = nj.array(sim.states.length);
      for (let i = 0; i < sim.states.length; i++) {
        frac_automated_tasks[i] =
          (ftm.Model.get_automated_tasks_count(sim.states[i].goods) + ftm.Model.get_automated_tasks_count(sim.states[i].hardware_rnd))
            / (sim.consts.goods.n_labour_tasks + sim.consts.rnd.n_labour_tasks);
      }

      add_multigraph(sim, [
        { label: 'GWP',                     var: 'gwp',                    yscale: 'log'},
        { label: 'Software',                var: 'software.v',             yscale: 'log'},
        { label: 'Hardware',                var: 'hardware',               yscale: 'log'},
        { label: 'Hardware efficiency',     var: 'hardware_performance.v', yscale: 'log'},
        { label: 'Labour',                  var: 'labour',                 yscale: 'log'},
        { label: 'Capital',                 var: 'capital',                yscale: 'log'},
        { label: 'Compute',                 var: 'compute',                yscale: 'log'},
        { label: 'Biggest training run',    var: 'biggest_training_run',   yscale: 'log'},
        { label: 'Money spent on training', var: 'money_spent_training',   yscale: 'log'},

        { label: 'Fraction of GWP spent on training', var: nj.div(sim.get_thread('money_spent_training'), sim.get_thread('gwp')), yscale: 'log'},
        { label: 'Fraction of compute invested in training', var: 'frac_compute.training.v', yscale: 'log'},

        { label: 'Fraction of cognitive tasks automated', var: frac_automated_tasks, yscale: 'linear'},
      ], '#metrics-graph-container');

      document.body.classList.remove('running');
    }, immediate ? 20 : 500);
  }
}

let tradeoff_enabled = document.querySelectorAll('.runtime_training_tradeoff_enabled');

function update_tradeoff_disabled(enable) {
  let tradeoff = document.getElementById('runtime_training_tradeoff');
  let max_tradeoff = document.getElementById('runtime_training_max_tradeoff');

  if (enable) {
    tradeoff.parentElement.classList.remove('disabled');
    max_tradeoff.parentElement.classList.remove('disabled');
  } else {
    tradeoff.parentElement.classList.add('disabled');
    max_tradeoff.parentElement.classList.add('disabled');
  }

  for (let checkbox of tradeoff_enabled) {
    checkbox.checked = enable;
  }
}

for (let checkbox of tradeoff_enabled) {
  checkbox.addEventListener('change', () => {
    update_tradeoff_disabled(checkbox.checked);
    run_simulation(false);
  });
}

document.getElementById('simulate-button').addEventListener('click', () => run_simulation(true));
for (let input of document.querySelectorAll('.input-parameter input')) {
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      run_simulation(true);
    }
  });
}

function get_parameters() {
  let params = {};
  for (let input of document.querySelectorAll('.input-parameter .input')) {
    let v = (input.type == 'checkbox') ? input.checked : +input.value;
    if (Number.isNaN(v)) {
      // Invalid input
      return null;
    }
    params[input.id] = v;
  }

  let runtime_training_tradeoff_enabled = document.querySelector('.runtime_training_tradeoff_enabled').checked;

  params['runtime_training_tradeoff_enabled'] = runtime_training_tradeoff_enabled;

  return params;
}

// Internal, for debugging
function export_scenario() {
  let json = JSON.stringify(get_parameters(), null, 2);
  download(json, 'takeoff-scenario.json', 'text/plain');
}

function import_scenario(params) {
  for (let param in params) {
    let input = document.getElementById(param);
    if (!input) {
      continue;
    }

    if (param == 'runtime_training_tradeoff_enabled') {
      update_tradeoff_disabled(params[param]);
    } else {
      input.value = standard_format(parseFloat(params[param]));
    }
  }

  run_simulation(false);
}

function download(content, filename, type) {
  let link = document.createElement('a');
  let file = new Blob([content], {type: type});
  link.href = URL.createObjectURL(file);
  link.target = '_blank';
  link.download = filename;
  link.dispatchEvent(new MouseEvent('click'));
}

document.getElementById('export-button').addEventListener('click', () => export_scenario());

document.getElementById('import-button').addEventListener('change', function() {
  console.log(this.files);
  if (this.files.length == 0) {
    return;
  }

  let reader = new FileReader();
  reader.addEventListener('load', () => {
    let json = reader.result;
    import_scenario(JSON.parse(json));
  });

  reader.readAsText(this.files[0]);
});

// ------------------------------------------------------------------------
// Plotting stuff
// ------------------------------------------------------------------------

let plt = new Plotter();

function nodify(node_or_query) {
  if (typeof(node_or_query) == 'string') {
    return document.querySelector(node_or_query);
  }
  return node_or_query;
}

function plot_compute_decomposition(sim, container, crop_after_agi = true) {
  let t = sim.timesteps;

  let median_tai_arrival_bio = 2052; // https://www.cold-takes.com/forecasting-transformative-ai-the-biological-anchors-method-in-a-nutshell/

  let bioanchors = run_bioanchors_model({
    t_start: sim.t_start,
    t_end: Math.max(sim.timesteps[sim.timesteps.length-1], median_tai_arrival_bio + 5),

    t_step: best_guess_sim.consts.t_step,
    initial_software: best_guess_sim.states[0].software.v,
    initial_hardware: best_guess_sim.states[0].hardware_performance.v,
    initial_training_investment: 50 * best_guess_sim.states[0].compute_investment * best_guess_sim.states[0].frac_compute.training.v
  });

  let graph = plt.graph;

  let header = graph.add_header();
  header.innerHTML =
    '<input type="checkbox" id="bioanchors-button"><label for="bioanchors-button">Compare with bioanchors</label>' +
    '<br>' +
    '<input type="checkbox" id="normalize-button"><label for="normalize-button">Normalize at wake-up year</label>';

  let show_bioanchors_button = header.querySelector('#bioanchors-button');
  let show_normalized_button = header.querySelector('#normalize-button');

  if (ui_state.show_bioanchors) {
    show_bioanchors_button.checked = true;
  }

  if (ui_state.show_normalized_decomposition) {
    show_normalized_button.checked = true;
  }

  function add_data(t, v, line_options) {
    let crop_year = Infinity;
    if (crop_after_agi && sim.agi_year != null) crop_year = sim.agi_year + 5;
    if (ui_state.show_bioanchors)               crop_year = Math.max(crop_year, median_tai_arrival_bio + 5);

    let start_idx = 0;
    let reference_idx = (sim.rampup_start != null) ? nj.argmax(nj.gte(t, sim.rampup_start)) : 0;
    let end_idx = (sim.timesteps[sim.timesteps.length-1] >= crop_year) ? nj.argmax(nj.gte(t, crop_year)) : t.length;

    t = t.slice(start_idx, end_idx);
    v = v.slice(start_idx, end_idx);
    if (ui_state.show_normalized_decomposition) {
      v = nj.div(v, v[reference_idx]);
    }

    graph.add_data_xy(t, v, line_options);
  }

  function update_graph() {
    graph.clear_dataset();
    graph.clear_axvlines();

    let section = ui_state.show_bioanchors ? 'Our model' : null;

    if (!ui_state.show_bioanchors) {
      add_data(t, sim.get_thread('compute_investment'), {label: '$ on FLOP globally', color: 'blue', section});
    } else {
      add_data(
        bioanchors.timesteps,
        nj.mult(sim.get_thread('compute_investment'), sim.get_thread('frac_compute.training.v')),
        {label: 'Training compute investment ($)', color: 'purple', section}
      );
    }

    add_data(t, sim.get_thread('hardware_performance.v'), {label: 'Hardware (FLOP/$)', color: 'orange', section});
    add_data(t, sim.get_thread('software.v'), {label: 'Software (2022-FLOP per FLOP)', color: 'green', section});

    if (!ui_state.show_bioanchors) {
      add_data(t, sim.get_thread('frac_compute.training.v'), {label: 'Fraction global FLOP on training', color: 'red', section});
    }

    if (ui_state.show_bioanchors) {
      let linestyle = 'dashed';
      let section = 'Bioanchors';

      // Every bioanchor line has the same slope. We have to displace them
      // when showing the normalized decomposition.
      let displacement = ui_state.show_normalized_decomposition ? 5 : 0;

      add_data(bioanchors.timesteps, bioanchors.training_investment, {displacement: +displacement, label: 'Training compute investment ($)', color: 'purple', linestyle: linestyle, section});
      add_data(bioanchors.timesteps, bioanchors.hardware,            {displacement: 0, label: 'Hardware (FLOP/$)', color: 'orange', linestyle: linestyle, section});
      add_data(bioanchors.software_timesteps, bioanchors.software,   {displacement: -displacement, label: 'Software (2022-FLOP per FLOP)', color: 'green', linestyle: linestyle, section});

      graph.axvline(median_tai_arrival_bio, {
        linestyle: 'dashed',
        color: 'blue',
        label: 'Median year of TAI arrival',
        section: 'Bioanchors',
      });
    }

    plot_vlines(sim, 'black', graph)

    graph.update();
  }

  show_bioanchors_button.addEventListener('click', () => {
    ui_state.show_bioanchors = show_bioanchors_button.checked;
    update_graph();
  });

  show_normalized_button.addEventListener('click', () => {
    ui_state.show_normalized_decomposition = show_normalized_button.checked;
    update_graph();
  });

  update_graph();

  plt.set_title('Compute increase decomposition');

  plt.set_transform(ui_state.decomposition_graph_transform)
  plt.set_transform_callback((transform) => {ui_state.decomposition_graph_transform = transform});
  plt.set_legend_placement('outside');

  plot_oom_lines();

  plt.show(container);
}

function add_multigraph(sim, variables, container, crop_after_agi = true) {
  let label_to_var = {};
  let label_to_yscale = {};
  let labels = [];
  for (let o of variables) {
    labels.push(o.label);
    label_to_var[o.label] = o.var;
    label_to_yscale[o.label] = o.yscale;
  }

  plt.set_transform(ui_state.metrics_graph_transform)
  plt.set_transform_callback((transform) => {ui_state.metrics_graph_transform = transform});

  plt.set_top_selections(['raw metric', 'growth per year'], ui_state.metrics_graph_top_selection);
  plt.set_side_selections(labels, ui_state.metrics_graph_side_selection);
  plt.on_select((top, side, graph) => {
    graph.clear_dataset();

    let x;
    let y;

    ui_state.metrics_graph_top_selection = top;
    ui_state.metrics_graph_side_selection = side;

    let show_growth = (top == 'growth per year');

    if (show_growth) {
      if (label_to_yscale[side] == 'log') {
        let r = sim.get_growth(label_to_var[side]);
        x = r.t;
        y = r.growth;
        graph.yscale('linear');
      } else {
        x = null;
        y = null;
      }
    } else {
      x = sim.timesteps;
      y = (typeof label_to_var[side] == 'string') ? sim.get_thread(label_to_var[side]) : label_to_var[side];
      graph.yscale(label_to_yscale[side]);
    }

    if (x != null || y != null) {
      let xlims = [sim.timesteps[0], sim.timesteps[sim.states.length-1]];

      if (crop_after_agi) {
        let end_idx = (crop_after_agi && sim.agi_year != null) ? Math.min(sim.time_to_index(sim.agi_year + 5), sim.states.length) : sim.states.length;
        xlims[1] = sim.timesteps[end_idx];
        x = x.slice(0, end_idx);
        y = y.slice(0, end_idx);
      }

      graph.xlims(xlims);

      let label = side;
      if (show_growth) {
        label += ' growth';
      }
      graph.add_data_xy(x, y, {label: label});
    }
  });
  plot_vlines(sim);

  plt.show(container);
}

function plot_variable(sim, var_name, title, {plot_growth = false, crop_after_agi = true} = {}) {
  let x;
  let y;

  if (plot_growth) {
    let r = sim.get_growth(var_name);
    x = r.t;
    y = r.growth;
  } else {
    x = sim.timesteps;
    y = sim.get_thread(var_name);
  }


  if (crop_after_agi) {
    let end_idx = (crop_after_agi && sim.agi_year != null) ? Math.min(sim.time_to_index(sim.agi_year + 5), sim.states.length) : sim.states.length;
    x = x.slice(0, end_idx);
    y = y.slice(0, end_idx);
  }
    
  plt.plot(x, y);
  plot_vlines(sim);
  plt.set_title(title);
  plt.show();
}

function plot_oom_lines({line_style = 'dotted', color = 'grey'} = {}) {
  let dataset = plt.get_dataset();
  let high = -Infinity;
  let low = +Infinity;

  for (let data of dataset) {
    low = Math.min(nj.min(data.y), low);
    high = Math.max(nj.max(data.y), high);
  }

  low = Math.floor(Math.log10(low));
  high = Math.ceil(Math.log10(high));

  for (let oom = low; oom < high; oom++) {
    plt.axhline(10**oom, {
      linestyle: line_style,
      color: color,
    });
  }
}

function plot_vlines(sim, line_color = 'black', graph = null) {
  graph ||= plt;

  if (sim.rampup_start) {
    graph.axvline(sim.rampup_start, {
      linestyle: 'dotted',
      color: line_color,
      label: 'Wake-up',
    });
  }
              
  if (sim.rampup_mid) {
    graph.axvline(sim.rampup_mid, {
      linestyle: '-.',
      color: line_color,
      label: '20% automation',
    });
  }
              
  if (sim.timeline_metrics['automation_gns_100%']) {
    graph.axvline(sim.timeline_metrics['automation_gns_100%'], {
      linestyle: 'dashed',
      color: line_color,
      label: '100% automation',
    });
  }
}


// ------------------------------------------------------------------------
// Tabling stuff
// ------------------------------------------------------------------------

function clear_tables(container) {
  container = nodify(container);
  container.innerHTML = '';
}

function add_table(container, table, scroll_x) {
  container = nodify(container);

  let str = [];
  str.push('<table class="dataframe">');

  str.push('<thead>');
  str.push('<tr>');
  for (let id in table) {
    let name = id;
    if (id in humanNames) {
      name = humanNames[id];
    }
    str.push(`<th data-param-metric-id="${id}">${name}</th>`);
  }
  str.push('</tr>');
  str.push('</thead>');

  let num_rows = 0;
  for (let v of Object.values(table)) {
    num_rows = Math.max(num_rows, Array.isArray(v) ? v.length : 1);
  }

  let formatted_columns = {};
  for (let col_name in table) {
    let col = table[col_name];
    if (!Array.isArray(col)) col = [col];
    formatted_columns[col_name] = formatColumn(col);
  }

  for (let i = 0; i < num_rows; i++) {
    str.push('<tr>');
    for (let a in table) {
      let cell = formatted_columns[a][i];
      str.push(`<td>${cell}</td>`);
    }
    str.push('</tr>');
  }
  str.push('</table>');

  let wrapper = document.createElement('div');
  wrapper.classList.add('table-wrapper');
  wrapper.innerHTML = str.join('');
  container.appendChild(wrapper);

  return wrapper;
}

function format_value_for_cell(v) {
  if (Array.isArray(v)) {
    return '[' + v.join(', ') + ']';
  }
  return v;
}

function getFormatInformation(x) {
  let str;
  if (Math.abs(x) > 1e4) {
    // Force exponential
    str = x.toExponential();
  } else {
    str = x.toString();
  }

  let re = /^-?([0-9]*)(\.?[0-9]*)(e[+-][0-9]*)?$/i;

  let m = str.match(re);

  let groups = m.slice(1);

  let intDigits = (typeof groups[0] === 'undefined') ? 0 : groups[0].length;
  let fracDigits = (typeof groups[1] === 'undefined') ? 0 : groups[1].length - 1;
  let expDigits = (typeof groups[2] === 'undefined') ? 0 : groups[2].length - 2;
  let isExponential = (typeof groups[2] !== 'undefined');

  let expFracDigits;
  if (isExponential) {
    expFracDigits = fracDigits;
  } else {
    let strExp = x.toExponential();
    let mExp = strExp.match(re);
    let groupsExp = mExp.slice(1);
    expFracDigits = (typeof groupsExp[1] === 'undefined') ? 0 : groupsExp[1].length - 1;
    expDigits = (typeof groupsExp[2] === 'undefined') ? 0 : groupsExp[2].length - 2;
  }

  let info = {
    intDigits: intDigits,
    fracDigits: fracDigits,
    expDigits: expDigits,
    isExponential: isExponential,
    expFracDigits: expFracDigits,
  }
  return info;
}

function formatColumn(col) {
  let maxFracDigits = 0;
  let maxExpFracDigits = 0;
  let maxExpDigits = 0;
  let someExponential = false;

  for (let x of col) {
    if (!Array.isArray(x) && typeof x != 'string' && typeof x != 'boolean' && !Number.isNaN(x) && x != null) {
      let formatInfo = getFormatInformation(x);
      maxFracDigits = Math.max(maxFracDigits, formatInfo.fracDigits);
      maxExpFracDigits = Math.max(maxExpFracDigits, formatInfo.expFracDigits);
      maxExpDigits = Math.max(maxExpDigits, formatInfo.expDigits);
      someExponential |= formatInfo.isExponential;
    }
  }

  const FRAC_DIGIT_CAP = 6;
  maxFracDigits = Math.min(maxFracDigits, FRAC_DIGIT_CAP);
  maxExpFracDigits = Math.min(maxExpFracDigits, FRAC_DIGIT_CAP);

  let formatted = [];
  for (let i = 0; i < col.length; i++) {
    let x = col[i];
    if (Array.isArray(x)) {
      formatted.push('[' + x.join(', ') + ']');
    } else if (typeof x == 'boolean') {
      formatted.push(x ? 'Yes' : 'No');
    } else if (typeof x == 'string' || Number.isNaN(x)) {
      formatted.push(x);
    } else if (x == null) {
      formatted.push('NaN');
    } else {
      if (someExponential) {
        let f = col[i].toExponential(maxExpFracDigits);
        let [coefficient, exponent] = f.split('e');
        let sign = exponent.charAt(0);
        formatted.push(`${coefficient}e${sign}${exponent.substring(1).padStart(maxExpDigits, '0')}`);
      } else {
        formatted.push(col[i].toFixed(maxFracDigits));
      }
    }
  }

  return formatted;
}



// ------------------------------------------------------------------------
// Tooltips
// ------------------------------------------------------------------------

let parameter_names = {"full_automation_requirements_training": "AGI training requirements (FLOP with 2022 algorithms)", "flop_gap_training": "Effective FLOP gap (training)", "goods_vs_rnd_requirements_training": "Training goods vs R&D", "full_automation_requirements_runtime": "AGI runtime requirements", "flop_gap_runtime": "Effective FLOP gap (runtime)", "goods_vs_rnd_requirements_runtime": "Runtime goods vs R&D", "runtime_training_tradeoff": "Trade-off efficiency", "runtime_training_max_tradeoff": "Maximum trade-off", "labour_substitution_goods": "Labour substitution goods", "labour_substitution_rnd": "Labour substitution R&D", "capital_substitution_goods": "Substitution between capital and cognitive tasks for goods and services", "capital_substitution_rnd": "Substitution between capital and cognitive tasks for hardware R&D", "research_experiments_substitution_software": "Research experiments substitution software", "compute_software_rnd_experiments_efficiency": "Efficiency of experiments for software R&D", "hardware_returns": "Returns to hardware", "software_returns": "Returns to software", "hardware_performance_ceiling": "Maximum hardware performance", "software_ceiling": "Maximum software performance", "rnd_parallelization_penalty": "R&D parallelization penalty", "hardware_delay": "Hardware adoption delay", "frac_capital_hardware_rnd_growth": "Growth rate fraction capital hardware R&D", "frac_labour_hardware_rnd_growth": "Growth rate fraction labour hardware R&D", "frac_compute_hardware_rnd_growth": "Growth rate fraction compute hardware R&D", "frac_labour_software_rnd_growth": "Growth rate fraction labour software R&D", "frac_compute_software_rnd_growth": "Growth rate fraction compute software R&D", "frac_gwp_compute_growth": "Growth rate fraction GWP compute", "frac_compute_training_growth": "Growth rate fraction compute training", "frac_capital_hardware_rnd_growth_rampup": "Wake-up growth rate fraction capital hardware R&D", "frac_labour_hardware_rnd_growth_rampup": "Wake-up growth rate fraction labour hardware R&D", "frac_compute_hardware_rnd_growth_rampup": "Wake-up growth rate fraction compute hardware R&D", "frac_labour_software_rnd_growth_rampup": "Wake-up growth rate fraction labour software R&D", "frac_compute_software_rnd_growth_rampup": "Wake-up growth rate fraction of compute software R&D", "frac_gwp_compute_growth_rampup": "Wake-up growth rate fraction of GWP buying compute", "frac_compute_training_growth_rampup": "Wake-up growth rate fraction compute training AI models", "frac_capital_hardware_rnd_ceiling": "Max fraction capital hardware R&D", "frac_labour_hardware_rnd_ceiling": "Max fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_ceiling": "Max fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_ceiling": "Max fraction labour software R&D", "frac_compute_software_rnd_ceiling": "Max fraction compute software R&D", "frac_gwp_compute_ceiling": "Max fraction GWP compute", "frac_compute_training_ceiling": "Max fraction compute training", "initial_frac_capital_hardware_rnd": "Initial fraction capital hardware R&D", "initial_frac_labour_hardware_rnd": "Initial fraction labour hardware R&D", "initial_frac_compute_hardware_rnd": "Initial fraction compute hardware R&D", "initial_frac_labour_software_rnd": "Initial fraction labour software R&D", "initial_frac_compute_software_rnd": "Initial fraction compute software R&D", "initial_biggest_training_run": "Initial biggest training run", "ratio_initial_to_cumulative_input_hardware_rnd": "Initial vs cumulative input - hardware R&D", "ratio_initial_to_cumulative_input_software_rnd": "Initial vs cumulative input - software R&D", "initial_hardware_production": "Initial hardware production", "ratio_hardware_to_initial_hardware_production": "Accumulated hardware vs initial hardware production", "initial_buyable_hardware_performance": "Initial market hardware performance", "initial_gwp": "Initial GWP", "initial_population": "Initial world labour force.", "initial_cognitive_share_goods": "Initial cognitive share goods", "initial_cognitive_share_hardware_rnd": "Initial cognitive share in hardware R&D", "initial_compute_share_goods": "Initial compute share goods", "initial_compute_share_rnd": "Initial compute share R&D", "initial_experiment_share_software_rnd": "Initial experiment share software R&D", "rampup_trigger": "Wakeup trigger", "initial_capital_growth": "Initial capital growth rate", "labour_growth": "Population growth rate", "tfp_growth": "TFP growth rate", "compute_depreciation": "Compute depreciation rate", "money_cap_training_before_wakeup": "Money threshold training", "training_requirements_steepness": "Training requirements steepness (OOM)", "runtime_requirements_steepness": "Runtime requirements steepness (OOM)"};
let parameter_meanings = {"full_automation_requirements_training": "FLOP needed to train an AI capable of performing the most hard-to-train cognitive task using 2020 algorithms.", "flop_gap_training": "Ratio between the effective FLOP needed to train the most demanding AI task and the 20% most demanding task.", "goods_vs_rnd_requirements_training": "Ratio between the training requirements for fully automating goods and services production vs (hardware and software) R&D. The higher this is, the easier it is to automate R&D compared to goods and services. Note: takeoff metrics shown here and in the report refer to automation of goods and services unless it says otherwise.", "full_automation_requirements_runtime": "FLOP/s needed to substitute the output of one human worker in the most demanding AI task, using 2020 algorithms.", "flop_gap_runtime": "Ratio between the effective FLOP/year needed to automate the most demanding AI task and the 20% most demanding AI task.", "goods_vs_rnd_requirements_runtime": "Ratio between the effective FLOP/year needed to automate the most demanding cognitive task for goods production and for R&D. The higher this is, the easier it is to automate R&D with respect to goods and services production. This incorporates one-time gains for AIs doing R&D that are additional to the one-time gains for AIs providing goods and services; the time between the equivalent R&D automation milestones will typically be longer.", "runtime_training_tradeoff": "How efficiently you can substitute training for higher runtime compute requirements. A value of 0 or less indicates that you cannot perform this tradeoff at all. A value of N indicates that a 10X smaller training run increases runtime requirements by 10^N.       ", "runtime_training_max_tradeoff": "The maximum ratio of training that can be substituted with extra runtime", "labour_substitution_goods": "Substitution between cognitive tasks for goods and services.\nA more positive number means that it is easier to substitute the output of different cognitive tasks in the production of goods", "labour_substitution_rnd": "Substitution between cognitive tasks for R&D.\nA more positive number means that it is easier to substitute the output of different cognitive tasks in R&D", "capital_substitution_goods": "Substitution between capital and cognitive tasks for goods and services.\nA more positive number means that it is easier to substitute cognitive input by capital and viceversa in the production of goods ", "capital_substitution_rnd": "Substitution between capital and cognitive tasks for hardware R&D.\nA more positive number means that it is easier to substitute cognitive input by capital and viceversa in hardware R&D", "research_experiments_substitution_software": "Substitution between experiments and cognitive input for software R&D.\nA more positive number means that it is easier to substitute cognitive input by experiments and viceversa in software R&D", "compute_software_rnd_experiments_efficiency": "Diminishing returns rate of the physical compute used for experiments to improve software in R&D", "hardware_returns": "How cumulative inputs to hardware R&D translate to better performance improvements. Doubling the cumulative input of harware R&D results in 2^hardware_returns improvement of hardware_efficiency (not accounting for the damping effect of the ceiling)", "software_returns": "How cumulative inputs to software R&D translate to better performance improvements. Doubling the cumulative input of software R&D results in 2^software_returns improvement of hardware_efficiency (not accounting for the damping effect of the ceiling)", "hardware_performance_ceiling": "Maximum FLOP/year/$ achiveable. Performance improvements get increasingly penalized as they approach this quantitiy.", "software_ceiling": "Maximum 2020-FLOP / physical FLOP achievable. Performance improvements get increasingly penalized as they approach this quantitiy.", "rnd_parallelization_penalty": "Penalization to concurrent R&D efforts. The ouputs of the hardware and software production function R&D get raised to this penalty before being aggregated to the cumulative total.", "hardware_delay": "Years between a chip design and its commercial release.", "frac_capital_hardware_rnd_growth": "Growth rate of fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_growth": "Growth rate of fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_growth": "Growth rate of fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_growth": "Growth rate of fraction of labour dedicated to software R&D", "frac_compute_software_rnd_growth": "Growth rate of fraction of compute dedicated to software R&D", "frac_gwp_compute_growth": "Growth rate of fraction of GWP dedicated to buying compute", "frac_compute_training_growth": "Growth rate of fraction of compute dedicated to training AI models", "frac_capital_hardware_rnd_growth_rampup": "Post wake-up growth rate of fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_growth_rampup": "Post wake-up rate of fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_growth_rampup": "Post wake-up rate of fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_growth_rampup": "Post wake-up rate of fraction of labour dedicated to software R&D", "frac_compute_software_rnd_growth_rampup": "Post wake-up rate of fraction of compute dedicated to software R&D", "frac_gwp_compute_growth_rampup": "Post wake-up rate of fraction of GWP dedicated to buying compute", "frac_compute_training_growth_rampup": "Post wake-up rate of fraction of compute dedicated to training AI models", "frac_capital_hardware_rnd_ceiling": "Maximum of the fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_ceiling": "Maximum of the fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_ceiling": "Maximum of the fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_ceiling": "Maximum of the fraction of labour dedicated to software R&D", "frac_compute_software_rnd_ceiling": "Maximum of the fraction of compute dedicated to software R&D", "frac_gwp_compute_ceiling": "Maximum of the fraction of GWP dedicated to buying compute", "frac_compute_training_ceiling": "Maximum of the fraction of compute dedicated to training AI models", "initial_frac_capital_hardware_rnd": "Initial fraction of capital dedicated to hardware R&D", "initial_frac_labour_hardware_rnd": "Initial fraction of labour dedicated to hardware R&D", "initial_frac_compute_hardware_rnd": "Initial fraction of compute dedicated to hardware R&D", "initial_frac_labour_software_rnd": "Initial fraction of labour dedicated to software R&D", "initial_frac_compute_software_rnd": "Initial fraction of compute dedicated to software R&D", "initial_biggest_training_run": "Amount of compute used for the largest AI training run up to date, measured in current-FLOP", "ratio_initial_to_cumulative_input_hardware_rnd": "Ratio of budget of hardware R&D in 2022 versus cumulative budget through history", "ratio_initial_to_cumulative_input_software_rnd": "Ratio of budget of software R&D in 2022 versus cumulative budget through history", "initial_hardware_production": "FLOP/year capabilities of hardware produced in 2022", "ratio_hardware_to_initial_hardware_production": "FLOP/year capabilities of total available hardware in 2022 vs hardware produced in 2022", "initial_buyable_hardware_performance": "FLOP/year/$ of the most cost efficient chip available in the market today.", "initial_gwp": "Initial real global world product. Measured in 2020 USD.", "initial_population": "Number of people in the world who work.", "initial_cognitive_share_goods": "Initial fraction of the GWP captured by labour and compute owners vs capital owners", "initial_cognitive_share_hardware_rnd": "Initial fraction of the R&D captured by labour and compute owners vs capital owners", "initial_compute_share_goods": "Initial fraction of cognitive output done by computers for goods production", "initial_compute_share_rnd": "Initial fraction of cognitive output done by computers for R&D", "initial_experiment_share_software_rnd": "Initial fraction of software research done by experiments with physical compute", "rampup_trigger": "Threshold of task automation in the production of goods and services that triggers massive investment growth.", "initial_capital_growth": "Initial rate of growth of capital", "labour_growth": "Population yearly growth rate. Controls the growth of labour.", "tfp_growth": "Yearly rate of growth of the total factor productivity. Each year the production gets passively better by a factor TFP that grows at this rate.", "compute_depreciation": "Yearly rate of depreciation of hardware. Each year this portion of the accumulated compute is depreciated.", "money_cap_training_before_wakeup": "After the cost of compute of the largest training run reaches this number, the growth of the fraction of compute alloctaed to training stops growing until wake-up.", "training_requirements_steepness": "How \"discontinuous\" the training requirements distribution is. A steepness of 3 OOM, for example, allows the distribution to jump only each 3 OOM.", "runtime_requirements_steepness": "How \"discontinuous\" the runtime requirements distribution is. A steepness of 3 OOM, for example, allows the distribution to jump only each 3 OOM."};
let parameter_justifications = {"": "", "full_automation_requirements_training": "I'm mostly anchoring to the Bio Anchors report, with an extra OOM to account for TAI being a lower bar than full automation (AGI).", "flop_gap_training": "See discussion in <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1Z7HJ9pHctgDi1XYbgRW9-7J1bxTL98KW1qb7HN7Mv-A/edit#heading=h.grg4srb18f02\" target=\"_blank\">summary</a></span> and <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.o4db3tcgrq28\" target=\"_blank\">full report</a></span>.", "goods_vs_rnd_requirements_training": "I expect fully automating the cognitive tasks in good production or R&amp;D will require AI to (collectively) be extremely general and flexible. I have slightly lower requirements for R&amp;D because it may be easier to gather data, and there may be fewer regulatory restrictions. I think you could defend even lower requirements for software R&amp;D.", "full_automation_requirements_runtime": "I anchor on the Bio Anchors report (~1e16 FLOP/s). Then I adjust upwards by 1 OOM to account for TAI being a lower bar than full automation (AGI). Then I adjust downwards by 6X to account for <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.wi3vto6myw5\" target=\"_blank\">one-time advantages</a></span> for AGI over humans in goods and services.", "flop_gap_runtime": "The spread of runtime requirements is smaller than the spread of training requirements for four reasons. First, a 10X increase in runtime compute typically corresponds to a 100X increase in training, e.g. for Chinchilla scaling. Secondly, increasing the horizon length of training tasks will increase training compute but not runtime. Thirdly, some of the \"one time gains\" for AGI over humans won't apply as much to pre-AGI systems; e.g. the benefits of thinking 100X faster are less for limited AIs that cannot learn independently over the course of weeks. Fourthly, a smaller spread is a hacky way to capture the fact it's harder to trade-off training compute for runtime compute today than it will be in the future.", "goods_vs_rnd_requirements_runtime": "I estimate the <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.wi3vto6myw5\" target=\"_blank\">one-time gains</a></span> of AGIs over humans to be 60X in R&amp;D vs 6X in goods and services. This is a difference of 10X. <br/><br/>Then I add on another 10X because the model implicitly assumes that there are 0.8M ppl doing software R&amp;D in 2022, and 8M ppl doing hardware R&amp;D. (Bc it multiplies the fraction of $ spent in these areas by the total labour force.) In fact I think the number of ppl working in these areas is ~10X less than this.", "runtime_training_tradeoff": "I describe the tradeoff dynamic <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.wa8izz23etjx\" target=\"_blank\">here</a></span>, and evidence about the parameter value <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1DZy1qgSal2xwDRR0wOPBroYE_RDV1_2vvhwVz4dxCVc/edit#bookmark=id.eqgufka8idwl\" target=\"_blank\">here</a></span>.", "runtime_training_max_tradeoff": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#bookmark=id.kcldlzp2l745\" target=\"_blank\">here</a></span>; I discuss the cap <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#bookmark=id.kcldlzp2l745\" target=\"_blank\">here</a></span>. Ideally the cap would increase over time as we discover new techniques and more capable AIs are capable of leveraging runtime compute in novel ways.", "labour_substitution_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.qazoq7gf2vgm\" target=\"_blank\">here</a></span>.", "labour_substitution_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.9t2n0pf04b2e\" target=\"_blank\">here</a></span>.", "capital_substitution_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.fo846mq256bx\" target=\"_blank\">here</a></span>.", "capital_substitution_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.o7tmwweugbb\" target=\"_blank\">here</a></span>.", "research_experiments_substitution_software": "<span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.limc1xpm5tfc\" target=\"_blank\">This section</a></span> argues that having a fixed quantity of physical compute for experiments wouldn't bottleneck algorithmic efficiency progress. Using a value close to 0 implies there aren't hard bottlenecks, approximating a Cobb Douglas production function.", "compute_software_rnd_experiments_efficiency": "This parameter is chosen so that the effective R&amp;D input from researchers and compute for experiments rose at the same rate over the last 10 years. (This is needed to keep their share of R&amp;D constant in a CES production function. It also means we can change the importance of experiments without changing the retrodicted rate of recent progress.) In particular, we estimate #researchers grew at 20% but physical compute grew at 50%. An exponent of 0.4 means that the effective input of compute for experiments also rose at 20%.", "hardware_returns": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.9us1ymg9hau0\" target=\"_blank\">here</a></span>.", "software_returns": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yzbcl83o650l\" target=\"_blank\">here</a></span>.", "hardware_performance_ceiling": "Based on a rough estimate from a technical advisor. They guessed energy prices could eventually fall 10X from today to $0.01/kWh. And that (based on <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#56a3f1;\"><a href=\"https://en.wikipedia.org/wiki/Landauer%27s_principle\" target=\"_blank\">Landauer\u2019s limit</a></span>) you might eventually do 1e27 bit erasures per kWh. That implies 1e29 bit erasures per $. If we do 1 FLOP per bit, that's 1e29 FLOP/$. You could go get more FLOP/$ than this with reversible computing, at least 1e30 FLOP/$.<br/>The advisor separately estimated 1e24 FLOP/$ as the limit within the current paradigm (the value used in Bio Anchors).<br/>I'll be somewhat conservative and use the mid-point as my best guess, 1e27 FLOP/$.<br/>Lastly, I'll adjust these FLOP/$ down an OOM to get FLOP/year/$ (implying that we use chips for 10 years). So best-guess 1e26 FLOP/year/$.", "software_ceiling": "If training AGI currently requires 1e36 FLOP, but this can in the limit be reduced to human life-time learning FLOP of 1e24, that's 12 OOMs improvement. ", "rnd_parallelization_penalty": "Econ models often use a value of 1. A prominent growth economist thought that values between 0.4 and 1 are reasonable. If you think adding new people really won't help much with AI progress, you could use a lower value. Besiroglu's <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://tamaybesiroglu.com/papers/AreModels.pdf\" target=\"_blank\">thesis</a></span> cites estimates as low as 0.2 (p14).", "hardware_delay": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.2gnv7nk1tdrv\" target=\"_blank\">here</a></span>. The conservative value of 2.5 years corresponds to an <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://semiwiki.com/semiconductor-manufacturers/tsmc/2212-how-long-does-it-take-to-go-from-a-muddy-field-to-full-28nm-capacity/\" target=\"_blank\">estimate</a></span> of the time needed to make a new fab. The aggressive value (no delay) corresponds to fabless improvements in chip design that can be printed with existing production lines with ~no delay. ", "frac_capital_hardware_rnd_growth": "Real $ investments in hardware R&amp;D have <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1bGbzR0c3TqsRYTWS3s6Bysgh9ZOuKS1w6qH1SOI11iE/edit#gid=186138651\" target=\"_blank\">recently grown</a></span> at ~4%; subtracting out ~3% GWP growth implies ~1% growth in the fraction of GWP invested.", "frac_labour_hardware_rnd_growth": "As above.", "frac_compute_hardware_rnd_growth": "As above.", "frac_labour_software_rnd_growth": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.1v8m5dp6xefi\" target=\"_blank\">here</a></span>, calcs <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1qmiomnNLpjcWSaeT54KC1PH1hfi_jUFIkWszxJGVU5w/edit#gid=0\" target=\"_blank\">here</a></span>. I subtract out 1% population growth from 19% growth in #reseachers (which is smaller than the 20% growth in real $).", "frac_compute_software_rnd_growth": "As above.", "frac_gwp_compute_growth": "Assumed to be equal to the growth rate post \"wake up\" (below). Why? Demand for AI chips is smaller today than after ramp-up, pushing towards slower growth today. But growth today is from a smaller base, and can come from the share of GPUs growing as a fraction of semiconductor production (which won't be possible once it's already ~100% of production). I'm lazily assuming these effects cancel out.", "frac_compute_training_growth": "This corresponds to the assumption that there will be a $4b training run in 2030, in line with Bio Anchors' prediction. Discussed a little <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yyref6x2mzxu\" target=\"_blank\">here</a></span>.", "frac_capital_hardware_rnd_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.612idx97x187\" target=\"_blank\">here</a></span>. I substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.", "frac_labour_hardware_rnd_growth_rampup": "As above.", "frac_compute_hardware_rnd_growth_rampup": "I assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&amp;D.", "frac_labour_software_rnd_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.vi6088puv22e\" target=\"_blank\">here</a></span>. I substract out 3% annual GWP growth to calculate the growth in the <span style=\"font-style:italic;\">fraction</span> of GWP invested.", "frac_compute_software_rnd_growth_rampup": "I assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&amp;D.", "frac_gwp_compute_growth_rampup": "Discussed here. I substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.", "frac_compute_training_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.5xk4lbt60vr0\" target=\"_blank\">here</a></span>.", "frac_capital_hardware_rnd_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48\" target=\"_blank\">here</a></span>.", "frac_labour_hardware_rnd_ceiling": "As above.", "frac_compute_hardware_rnd_ceiling": "If people anticipate an AI driven singularity, the demand for progress in AI R&amp;D should become huge, such that the world allocates using a macroscopic of compute to AIs doing Ai R&amp;D.", "frac_labour_software_rnd_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48\" target=\"_blank\">here</a></span>.", "frac_compute_software_rnd_ceiling": "If people anticipate an AI driven singularity, the demand for progress in AI R&amp;D should become huge, such that the world allocates using a macroscopic of compute to AIs doing Ai R&amp;D.", "frac_gwp_compute_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.rh3lt1q0f0hl\" target=\"_blank\">here</a></span>.", "frac_compute_training_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.tz6v7gxroefr\" target=\"_blank\">here</a></span>.", "initial_frac_capital_hardware_rnd": "Hardware R&amp;D spend is <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://www.onlinecomponents.com/en/blogpost/2021-semiconductor-rd-spend-to-rise-4-357/\" target=\"_blank\">~$70b</a></span>, which is ~0.1%. I double this to include some of the spending on new capital equipment (which totals <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://www.statista.com/statistics/864897/worldwide-capital-spending-in-the-semiconductor-industry/\" target=\"_blank\">~$130b</a></span>).", "initial_frac_labour_hardware_rnd": "As above.", "initial_frac_compute_hardware_rnd": "As above.", "initial_frac_labour_software_rnd": "I'd guess annual spending on software workers for SOTA AI is $10-20b (DM spend is ~$1b); $16b is 0.02%.", "initial_frac_compute_software_rnd": "As above.", "initial_biggest_training_run": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.b15qyylbeqxp\" target=\"_blank\">here.</a></span>", "ratio_initial_to_cumulative_input_hardware_rnd": "See <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1bGbzR0c3TqsRYTWS3s6Bysgh9ZOuKS1w6qH1SOI11iE/edit#gid=186138651\" target=\"_blank\">calcs</a></span>, the cell called \"2022 inputs / cumulative inputs\"", "ratio_initial_to_cumulative_input_software_rnd": "Calculated assuming inputs have always been growing at 20%.", "initial_hardware_production": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yyref6x2mzxu\" target=\"_blank\">here</a></span>.", "ratio_hardware_to_initial_hardware_production": "Assumes annual FLOP production has been growing exponentially at 50%; ~28% from more FLOP/$ and ~22% from more $ spending.", "initial_buyable_hardware_performance": "The training of Palm cost <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://blog.heim.xyz/palm-training-cost/\" target=\"_blank\">~$10m</a></span>, and used 3e24 FLOP. This implies 3e17 FLOP/$. If companies renting chips make back their money over 2 years then that corresponds to 1.5e17 FLOP/year/$.", "initial_gwp": "See value for 2020 <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=1W\" target=\"_blank\">here</a></span>.", "initial_population": "<span style=\"color:#000000;\">50% of </span><span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://www.google.com/search?q=world+population+&amp;ei=bdAOY5mXFvXg0PEP9eeygAY&amp;ved=0ahUKEwjZgePejPD5AhV1MDQIHfWzDGAQ4dUDCA4&amp;uact=5&amp;oq=world+population+&amp;gs_lcp=Cgdnd3Mtd2l6EAMyBwgAELEDEEMyCAgAEIAEELEDMgoIABCxAxCDARBDMgcIABCxAxBDMgcIABCxAxBDMgsIABCABBCxAxCDATIKCAAQsQMQgwEQQzIFCAAQgAQyBQgAEIAEMgcIABCxAxBDOgcIABBHELADSgUIPBIBMUoECEEYAEoECEYYAFDHBljXB2DaCWgBcAF4AIABS4gBiQGSAQEymAEAoAEByAEIwAEB&amp;sclient=gws-wiz\" target=\"_blank\">world population.</a></span>", "initial_cognitive_share_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.osubru9caix\" target=\"_blank\">here</a></span>.", "initial_cognitive_share_hardware_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.myon1lrlwj34\" target=\"_blank\">here</a></span>.", "initial_compute_share_goods": "The semiconductor industry's annual revenues are ~$600b, which is ~1% of GWP.", "initial_compute_share_rnd": "As above. (I'm assuming compute is used equally in hardware+software as in the general economy - excluding compute used on training runs.)", "initial_experiment_share_software_rnd": "Matches the capital share for hardware R&amp;D.", "rampup_trigger": "AI that can readily automate 6% of cognitive tasks can readily generate ~$3tr. (Assuming cognitive work is paid ~$50tr, ~50% of GDP). This is ~5X current annual semiconductor revenues. Of course, there will be a lag before this value is added, but people will also invest on the basis of promising demos. AI companies will probably only capture a small fraction of the value, which is why I don't use a smaller number.", "initial_capital_growth": "Recent GWP growth is ~3%; in equilibrium capital grows at the same rate as GDP.", "labour_growth": "<a href=\"https://www.google.com/search?q=world+population+growth+rate&amp;ei=VdAOY_7TDqu00PEP6pOViAM&amp;ved=0ahUKEwi-0qLTjPD5AhUrGjQIHepJBTEQ4dUDCA4&amp;uact=5&amp;oq=world+population+growth+rate&amp;gs_lcp=Cgdnd3Mtd2l6EAMyCAgAEIAEELEDMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIICAAQgAQQyQMyBQgAEIAEOgcIABBHELADOgQIABBDSgUIPBIBMUoECEEYAEoECEYYAFCOBVjcCGDnCWgBcAF4AIABVogB0gKSAQE1mAEAoAEByAEIwAEB&amp;sclient=gws-wiz\" target=\"_blank\">Source.</a>", "tfp_growth": "Average TFP growth over the last 20 years, <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1C-RUowD3Nwo51UF5ZeBjbeLwaw4HQ1o13KyJlhuXCcU/edit#gid=2116796644\" target=\"_blank\">source</a></span>.", "compute_depreciation": "Rough guess.", "money_cap_training_before_wakeup": "", "training_requirements_steepness": "", "runtime_requirements_steepness": ""};
let metric_names = {"automation_gns_20%": "20% automation year", "automation_gns_100%": "100% automation year", "sub_agi_year": "Powerful sub-AGI year", "agi_year": "AGI year", "automation_rnd_20%": "20% R&D automation year", "automation_rnd_100%": "100% R&D automation year", "rampup_start": "Wake-up year", "full_automation_gns": "20-100% economic automation", "full_automation_rnd": "20-100% R&D automation", "sub_agi_to_agi": "Powerful sub-AGI to AGI", "cog_output_multiplier": "2X to 10X cognitive output multiplier", "gwp_growth": "5% to 20% GWP growth", "doubling_times": "Pattern of GWP doublings", "full_automation_gns skew": "Skew of 20-100% economic automation", "agi_year skew": "Skew of time to AGI date", "importance": "Importance", "variance_reduction": "Variance reduction"};
let metric_meanings = {"automation_gns_20%": "Year when AI has automated 20% of goods and services", "automation_gns_100%": "Year when AI has automated 100% of goods and services", "sub_agi_year": "Year when AI could readily perform 20% economic tasks without trading off training compute with runtime compute (based on largest training run)", "agi_year": "Year when AI could readily perform 100% economic tasks without trading off training compute with runtime compute (based on largest training run)", "automation_rnd_20%": "Year when AI has automated 20% of R&D tasks (in software and hardware)", "automation_rnd_100%": "Year when AI has automated 100% of R&D tasks (in software and hardware)", "rampup_start": "Year when AI becomes economically significant. This happens when a certain percentage of goods and services cognitive tasks have been automated (usually 3% but depends on the inputs) ", "full_automation_gns": "Years from AI that has automated 20% of goods and services to AI that has automated 100%.", "full_automation_rnd": "Years from AI that has automated 20% of R&D tasks to AI that has automated 100%.", "sub_agi_to_agi": "Years from \u201cAI could readily perform 20% economic tasks without trading off training compute with runtime compute\u201d to \u201cAI could readily perform 100% economic tasks without trading off training compute with runtime compute\u201d (based on largest training run)", "cog_output_multiplier": "Years between <a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.vg16ijfr9w9a/\">AI's cognitive value-add</a> in hardware R&D being 2X and 10X.", "gwp_growth": "Years between instantaneous gwp growth reaching 5% and reaching 20%.\n\n(Not very meaningful because of zigzagging issues.)", "doubling_times": "Years between successive GWP doublings.", "full_automation_gns skew": "How the intervals are biased towards more aggressive or more conservative takeoffs.\nA more positive value means that the intervals are biased towards shorter takeoffs.\nComputed using the <i>20-100% economic automation</i> as |best guess metric value - conservative metric value| - |aggressive metric value - best guess metric value| ", "agi_year skew": "How the intervals are biased towards more aggressive or more conservative AGI dates.\nA more positive value means that the intervals are biased towards shorter takeoffs.\nComputed using the <i>AGI year</i> metric as |best guess metric value - conservative metric value| - |aggressive metric value - best guess metric value| ", "importance": "Influence of this parameter on the <i>20-100% economic automation</i> metric. Computed as |aggressive metric value - conservative metric value|.", "variance_reduction": "Expected reduction of the variance of the <i>20-100% economic automation</i> metric if we knew the value of the parameter. Computed as 1 \u2212 E[var(metric | \u03b8<sub>i</sub>)] / var(metric)."};
let variable_names = {"period": "Period", "year": "Year", "biggest_training_run": "Biggest training run", "hardware_performance": "Hardware performance", "software": "Software", "compute_investment": "Investment on compute", "gwp": "GWP", "capital": "Capital", "labour": "Labour", "tfp_rnd": "R&D TFP", "rnd_input_software": "Software R&D input", "cumulative_rnd_input_software": "Cumulative software R&D input", "frac_tasks_automated_goods": "Fraction of goods and services tasks automated", "frac_tasks_automated_rnd": "Fraction of R&D tasks automated", "frac_compute_training": "Fraction of compute invested in training", "frac_gwp_compute": "Fraction of GWP invested in compute"};

parameter_meanings['t_step'] = 'Duration of each simulation step in years';
parameter_meanings['t_end'] = 'When to stop the simulation';

for (let k of Object.keys(variable_names)) {
  variable_names[`${k} growth rate`] = `${variable_names[k]} growth rate`;
  variable_names[`${k} doubling time`] = `${variable_names[k]} doubling time`;
}

let tippy_instances = [];
for (let input of document.querySelectorAll('.input-parameter')) {
  let elements = input.querySelectorAll('input');
  let id = elements[elements.length-1].id;
  let value = elements[elements.length-1].value;
  let meaning = parameter_meanings[id];
  let justification = parameter_justifications[id];

  let tooltip = '';

  if (meaning) {
    tooltip += `<span style="font-weight: bold">Meaning:</span> ${meaning}`
  }

  if (id == 'training_requirements_steepness' || id == 'runtime_requirements_steepness' ) {
    // Add illustration
    let runtime_or_training = (id == 'training_requirements_steepness') ? 'training' : 'runtime';
    tooltip += '<br>';
    tooltip += '<br>';
    tooltip += `<div class="illustration" style="margin-bottom: -3em">This is what the ${runtime_or_training} R&D requirements look like with the current value for the steepness of <span class="steepness"></span>:<div class="requirements-graph-container"></div></div>`;
  }

  if (tooltip.length > 0) tooltip += '<br><br>';
  if (input.classList.contains('simulation-parameter')) {
    tooltip += `<span style="font-weight: bold">Initial value:</span> ${value}`;
  } else {
    tooltip += `<span style="font-weight: bold">Best guess value:</span> ${value}`;
  }

  if (justification) {
    if (tooltip.length > 0) tooltip += '<br><br>';
    tooltip += '<span style="font-weight: bold">Justification for best guess value:</span> ' + justification;
  }

  let initialized = false;
  tippy_instances.push(
    tippy(input.querySelector('input'), {
      content: tooltip,
      triggerTarget: input,
      allowHTML: true,
      interactive: true,
      placement: 'right',
      appendTo: document.body,
      hideOnClick: false,
      theme: 'light-border',
      maxWidth: (id == 'training_requirements_steepness' || id == 'runtime_requirements_steepness' ) ? '460px' : '350px',
      onMount: (instance) => {
        if (initialized) return;

        if (id == 'training_requirements_steepness' || id == 'runtime_requirements_steepness' ) {
          let graph_container = instance.popper.querySelector('.requirements-graph-container');
          let steepness_node = instance.popper.querySelector('.steepness');
          let runtime_or_training = (id == 'training_requirements_steepness') ? 'training' : 'runtime';

          let input = document.getElementById(id);
          input.addEventListener('input', () => {
            draw_requirements(runtime_or_training, graph_container, steepness_node);
          });

          draw_requirements(runtime_or_training, graph_container, steepness_node);
        }

        initialized = true;
      },
    })
  );
}

function draw_requirements(runtime_or_training, graph_container, steepness_node) {
  let n_labour_tasks = 100;
  let params = get_parameters();

  if (!params) return;

  let full_requirements = params['full_automation_requirements_' + runtime_or_training];
  let gap = params['flop_gap_' + runtime_or_training];
  let steepness = params[runtime_or_training + '_requirements_steepness'];

  steepness_node.innerHTML = steepness;

  let automation_costs = ftm.Model.process_automation_costs(
    full_requirements,
    gap,
    n_labour_tasks,
    steepness,
  );
  automation_costs.shift(); // remove first automatable tasks

  automation_costs.splice(0, 0, full_requirements / gap**1.8); // remove first automatable tasks
  automation_costs.push(full_requirements * gap**0.2)

  let percentage_automated = nj.zeros(n_labour_tasks + 1);
  percentage_automated[0] = 0;
  for (let i = 1; i < n_labour_tasks + 1; i++) {
    percentage_automated[i] = 100 * (i - 1)/(n_labour_tasks - 1);
  }
  percentage_automated[percentage_automated.length] = 100;

  if (steepness != 0) {
    // Make the steps look flat

    let x = [];
    let y = [];
    for (let i = 0; i < n_labour_tasks + 1; i++) {
      if (i > 0) {
        x.push(automation_costs[i]);
        y.push(percentage_automated[i-1]);
      }

      x.push(automation_costs[i]);
      y.push(percentage_automated[i]);
    }

    automation_costs = x;
    percentage_automated = y;
  }

  plt.clear(graph_container);
  plt.set_size(450, 250);
  plt.plot(automation_costs, percentage_automated);
  plt.set_interactive(false);
  plt.yscale('linear');
  plt.xscale('log');
  plt.xlabel(`${runtime_or_training} FLOP`);
  plt.ylabel('% of automatable tasks');
  plt.show(graph_container);
}

let humanNames = {...parameter_names, ...metric_names, ...variable_names};

for (let parameter of document.querySelectorAll('.input-parameter')) {
  let paramId = parameter.querySelector('input').id;
  if (paramId in humanNames) {
    parameter.querySelector('label').innerHTML = humanNames[paramId];
  }
}

// Hide sidebar tooltips on scroll
document.querySelector('#parameter-tabs .tab-content').addEventListener('scroll', () => {
  for (let instance of tippy_instances) {
    let duration = instance.props.duration;
    instance.setProps({duration: [null, 0]});
    instance.hide();
    instance.setProps({duration: duration});
  }
});

// ------------------------------------------------------------------------
// Presets
// ------------------------------------------------------------------------
let presetModal = document.querySelector('#preset-selector-modal');
let presetModalButton = document.querySelector('#preset-modal-button');
let presetContainer = document.querySelector('#preset-container');

let presets = {
  "Aggressive": aggressive_parameters,
  "Best guess": best_guess_parameters,
  "Conservative": conservative_parameters,
};

for (let presetName in presets) {
  let button = html(`<div><button class="preset-load-button">${presetName}</button></div>`);
  presetContainer.appendChild(button);
  button.addEventListener('click', () => {
    import_scenario(presets[presetName]);
    presetModal.classList.add('hidden');
  });
}

presetModalButton.addEventListener('click', () => {
  if (presetModal.classList.contains('hidden')) {
    presetModal.classList.remove('hidden');
  } else {
    presetModal.classList.add('hidden');
  }
});

// Hide modal if the user clicks outside
document.body.addEventListener('mousedown', (e) => {
  function isInsideModalOrButton(node) {
    if (node == null) return false;
    if (node == presetModal) return true;
    if (node == presetModalButton) return true;
    return isInsideModalOrButton(node.parentElement);
  }

  if (!isInsideModalOrButton(e.target)) {
    presetModal.classList.add('hidden');
  }
});

// Hide modal if the user presses Esc
document.addEventListener('keyup', (e) => {
  if(e.key === "Escape") {
    presetModal.classList.add('hidden');
  }
});

// ------------------------------------------------------------------------
// Misc
// ------------------------------------------------------------------------

let backgroundTimers = [];

function dispatchBackgroundProcess(f, debounceTimeout) {
  if (debounceTimeout == 0) {
    f();
    return null;
  }

  let timer = setTimeout(f, debounceTimeout);
  backgroundTimers.push(timer);
  return timer;
}

function cancelBackgroundProcesses() {
  for (let timer of backgroundTimers) {
    clearTimeout(timer);
  }
  backgroundTimers = [];
}

function injectMeaningTooltips() {
  for (let node of document.querySelectorAll('[data-param-metric-id]')) {
    let id = node.dataset.paramMetricId;
    let content = parameter_meanings[id] || metric_meanings[id];
    injectMeaningTooltip(node, content);
  }
}

function injectMeaningTooltip(node, meaning) {
  if (node._meaningInjected) return;

  if (meaning) {
    let icon = document.createElement('i');
    icon.classList.add('bi', 'bi-info-circle', 'info-icon');
    node.append(icon);

    tippy(icon, {
      content: meaning,
      allowHTML: true,
      interactive: true,
      placement: (node.parentElement.parentElement.tagName == 'THEAD') ? 'top' : 'right',
      appendTo: document.body,
      plugins: [hideOnEsc],
      theme: 'light-border',
    });

    node._meaningInjected = true;
  }
}

// See https://atomiks.github.io/tippyjs/v6/plugins/#hideonesc
const hideOnEsc = {
  name: 'hideOnEsc',
  defaultValue: true,
  fn({hide}) {
    function onKeyDown(event) {
      if (event.keyCode === 27) {
        hide();
      }
    }

    return {
      onShow() {
        document.addEventListener('keydown', onKeyDown);
      },
      onHide() {
        document.removeEventListener('keydown', onKeyDown);
      },
    };
  },
};

// ------------------------------------------------------------------------
// And... run
// ------------------------------------------------------------------------

run_simulation(true, sim => {best_guess_sim = sim;});
