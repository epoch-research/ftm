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
  show_bioanchors: false,
  show_normalized_decomposition: false,

  metrics_to_show: 'important-metrics',

  // TODO Generalize these
  metrics_graph_top_selection: 'raw metric',
  metrics_graph_side_selection: 'GWP',
};

let metrics_to_show = document.getElementById('metrics-to-show');
metrics_to_show.addEventListener('change',  () => {
  ui_state.metrics_to_show = metrics_to_show.value;

  let all_metrics_table = {
    ...sim.get_takeoff_metrics(),
    'rampup_start': sim.rampup_start,
    'agi_year': sim.agi_year,
  };

  let metrics_table;

  if (ui_state.metrics_to_show == 'important-metrics') {
    let important_metrics = ["billion_agis", "full_automation", "agi_year", "rampup_start"];

    metrics_table = {};
    for (let key of important_metrics) metrics_table[key] = all_metrics_table[key];
  } else {
    metrics_table = all_metrics_table;
  }

  clear_tables('#takeoff-metrics-table-container');
  add_table('#takeoff-metrics-table-container', metrics_table);

  injectMeaningTooltips();
});

function run_simulation(immediate, callback) {
  let params = get_parameters();
  if (params) {
    cancelBackgroundProcesses();
    dispatchBackgroundProcess(() => {
      let js_params = transform_python_to_js_params(params);
      sim = ftm.run_simulation(js_params);

      if (callback) {
        callback(sim);
      }

      let takeoff_metrics = sim.get_takeoff_metrics();

      let t = sim.get_thread('t_year');
      let indices = sim.get_thread('t_idx');
      //let b = sim.get_thread('biggest_training_run');
      let b = sim.get_thread('compute');

      metrics_to_show.value = ui_state.metrics_to_show;
      metrics_to_show.dispatchEvent(new Event('change'));

      clear_tables('#summary-table-container');
      clear_tables('#year-by-year-table-container');

      add_table('#summary-table-container', sim.get_summary_table());

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

      for (let i of indices) {
        if (t[i] == Math.floor(t[i])) {
          for (let v in detailed_table) {
            if (!(v in yearly_table)) yearly_table[v] = [];
            yearly_table[v].push(detailed_table[v][i]);
          }
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

      add_multigraph(sim, [
        { label: 'GWP',                     var: 'gwp'},
        { label: 'Software',                var: 'software.v'},
        { label: 'Hardware',                var: 'hardware'},
        { label: 'Hardware efficiency',     var: 'hardware_performance.v'},
        { label: 'Labour',                  var: 'labour'},
        { label: 'Capital',                 var: 'capital'},
        { label: 'Compute',                 var: 'compute'},
        { label: 'Biggest training run',    var: 'biggest_training_run'},
        { label: 'Money spent on training', var: 'money_spent_training'},
        { label: 'Fraction of compute invested in training', var: 'frac_compute.training.v'},
      ], '#metrics-graph-container');
    }, immediate ? 1 : 500);
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

update_tradeoff_disabled();

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

  if (!runtime_training_tradeoff_enabled) {
    params['runtime_training_tradeoff'] = 0;
  }

  return params;
}

// Internal, for debugging
function set_parameters(params) {
  for (let param in params) {
    let input = document.getElementById(param);
    if (!input) {
      console.log(param);
    }
    input.value = params[param];
  }

  run_simulation(false);
}


// ------------------------------------------------------------------------
// Plotting stuff
// ------------------------------------------------------------------------

class Graph {
  colors = [
    "steelblue", "orange", "pink",
  ];


  nameToScale = {
    'linear': d3.scaleLinear,
    'log': d3.scaleLog,
  };

  constructor(options) {
    this.nodes = {};
    this.nodes.wrapper = document.createElement('div');
    this.nodes.wrapper.classList.add('graph-wrapper');

    this.dataset = [];
    this.xAxis = {scale: options.xscale, lims: [null, null]};
    this.yAxis = {scale: options.yscale, lims: [null, null]};
    this.axvlines = [];
    this.axhlines = [];
    this.current_color = 0;
    this.title = "";

    this.top_selections = [];
    this.side_selections = [];
    this.on_select_callback = null;
  }

  attach(container) {
    this.container = nodify(container);
    this.container.appendChild(this.nodes.wrapper);
    this.render();
  }

  get_dataset() {
    return this.dataset;
  }

  clear() {
    this.dataset = [];
    this.axhlines = [];
    this.axvlines = [];
    this.current_color = 0;
  }

  clear_dataset() {
    this.dataset = [];
    this.current_color = 0;
  }

  set_title(title) {
    this.title = title;
  }

  add_data_xy(x, y, options = {}) {
    let o = {x, y, ...options};
    o.color = o.color || this.colors[(this.current_color++) % this.colors.length]
    this.dataset.push(o);
  }

  xscale(scale) {
    this.xAxis.scale = scale;
  }

  yscale(scale) {
    this.yAxis.scale = scale;
  }

  xlims(lims) {
    this.xAxis.lims = lims;
  }

  ylims(lims) {
    this.yAxis.lims = lims;
  }

  axvline(x, options = {}) {
    this.axvlines.push({x, ...options});
  }

  axhline(y, options = {}) {
    this.axhlines.push({y, ...options});
  }

  update() {
    if (!this.nodes.plot) {
      // The graph hasn't been initialized yet. Exiting...
      return;
    }

    this.nodes.plot.innerHTML = "";

    // set the dimensions and margins of the graph
    let margin = {top: 20, right: 10, bottom: 30, left: 40};
    let width = 770 - margin.left - margin.right;
    let height = 520 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svg_container = d3.select(this.nodes.plot)
      .append('div')
      .attr('class', 'svg-container')
    ;

    if (this.nodes.header) {
      svg_container.node().append(this.nodes.header);
    }

    let svg = svg_container
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .attr('class', 'plot')
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    // Add a margin so that we don't clip half a line
    let top_clip_margin = 1;

    let clip = svg.append("defs").append("SVG:clipPath")
      .attr("id", "clip")
      .append("SVG:rect")
      .attr("width", width)
      .attr("height", height + top_clip_margin)
      .attr("x", 0)
      .attr("y", -top_clip_margin)
    ;

    let content = svg.append('g')
      .attr("clip-path", "url(#clip)");

    let first = this.dataset[0];
    let indices = [...first.x.keys()];

    let xlims = [Infinity, -Infinity];
    let ylims = [Infinity, -Infinity];
    for (let data of this.dataset) {
      xlims[0] = Math.min(nj.min(data.x), xlims[0]);
      xlims[1] = Math.max(nj.max(data.x), xlims[1]);

      ylims[0] = Math.min(nj.min(data.y), ylims[0]);
      ylims[1] = Math.max(nj.max(data.y), ylims[1]);
    }

    if ((ylims[1] - ylims[0]) < 0.001) {
      ylims[0] = 0;
      ylims[1] *= 10;
    }

    if (this.xAxis.lims[0] != null) xlims[0] = this.xAxis.lims[0];
    if (this.xAxis.lims[1] != null) xlims[1] = this.xAxis.lims[1];
    if (this.yAxis.lims[0] != null) ylims[0] = this.yAxis.lims[0];
    if (this.yAxis.lims[1] != null) ylims[1] = this.yAxis.lims[1];

    let xScale = this.nameToScale[this.xAxis.scale]();
    let yScale = this.nameToScale[this.yAxis.scale]();

    let x = xScale
      .domain(xlims)
      .range([0, width])
    ;

    // Add Y axis
    let y = yScale
      .domain(ylims)
      .range([height, 0])
    ;

    let currentX = x;
    let currentY = y;

    // Vertical lines
    for (let axvline of this.axvlines) {
      content.append('line')
        .datum([axvline.x])
        .attr('class', 'axvline')
        .attr('x1', d => x(d))
        .attr('x2', d => x(d))
        .attr('y1', 0)
        .attr('y2', height)
        .attr("stroke", axvline.color || this.defaults.axline_color)
        .style("stroke-dasharray", this.get_dasharray(axvline.linestyle || this.defaults.axline_linestyle))
        .attr("stroke-width", 2)
      ;
    }

    // Horizontal lines
    for (let axhline of this.axhlines) {
      content.append('line')
        .datum([axhline.y])
        .attr('class', 'axhline')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', y(axhline.y))
        .attr('y2', y(axhline.y))
        .attr("stroke", axhline.color || this.defaults.axline_color)
        .style("stroke-dasharray", this.get_dasharray(axhline.linestyle || this.defaults.axline_linestyle))
        .attr("stroke-width", 2)
      ;
    }

    // Add the paths
    for (let data of this.dataset) {
      let datum = [];
      for (let i = 0; i < data.x.length; i++) {
        datum.push({x: data.x[i], y: data.y[i]});
      }

      let path = content.append("path")
        .datum(datum)
        .attr("class", "data-path")
        .attr("fill", "none")
        .attr("stroke", data.color)
        .attr("stroke-width", 2.5)
        .style("stroke-dasharray", this.get_dasharray(data.linestyle || '-'))
        .attr("d", d3.line()
          .x(d => x(d.x))
          .y(d => y(d.y) - (parseFloat(this.dataset.displacement) || 0))
        );

      if (data.displacement) {
        path.node().dataset.displacement = data.displacement;
      }
    }

    let xAxis = svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .attr("stroke-width", 2)
      .call(d3.axisBottom(x));

    let yAxis = svg.append("g")
      .attr("stroke-width", 2)
      .call(d3.axisLeft(y));

    let legend_objects = [];

    for (let data of this.dataset) {
      if (!data.label) continue;
      legend_objects.push(data);
    }

    for (let axvline of this.axvlines) {
      if (!axvline.label) continue;
      legend_objects.push(axvline);
    }

    if (legend_objects.length > 0) {
      let legend = svg_container
          .append('div')
            .style('top', margin.top + 20 + 'px')
            .style('left', margin.left + 20 + 'px')
            .attr('class', 'legend')
            .style('position', 'absolute')
        ;

      let add_legend_section = (objects, name) => {
        let section = legend.append('div')
          .attr('class', 'legend-section')
        ;

        if (name) {
          let header = section.append('div')
            .attr('class', 'legend-section-header')
            .text(name)
          ;
        }

        for (let object of objects) {
          let item = section.append('div')
            .attr("class", "legend-item")
          ;

          item.append('svg')
            .attr("width", 20)
            .attr("height", 4)
            .attr("class", "legend-item-line")
            .append('line')
              .attr('x1', 0)
              .attr('x2', 20)
              .attr('y1', 2)
              .attr('y2', 2)
              .attr('stroke', object.color)
              .style("stroke-dasharray", object.linestyle ? this.get_dasharray(object.linestyle) : "")
              .attr('stroke-width', 2)
          ;

          item.append('span')
            .text(object.label)
          ;

          if (object.description) {
            tippy(item.node(), {
              content: object.description,
              allowHTML: true,
              interactive: true,
              placement: 'left',
              appendTo: document.body,
              hideOnClick: false,
              theme: 'light-border',
            });
          }
        }
      }

      let sections = {};
      let misc_section = [];
      for (let object of legend_objects) {
        if (object.section) {
          if (!(object.section in sections)) {
            sections[object.section] = [];
          }
          sections[object.section].push(object);
        } else {
          misc_section.push(object);
        }
      }

      for (let section in sections) {
        add_legend_section(sections[section], section);
      }
      add_legend_section(misc_section);
    }

    if (this.graph_title) {
      svg.append('text')
        .attr('x', (width / 2))             
        .attr('y', -8)
        .attr('text-anchor', 'middle')  
        .style('font-size', '16px') 
        .text(this.graph_title)
      ;
    }

    let tooltip = d3.select('body')
      .append("div")
      .style("position", "absolute")
      .style("display", "none")
      .attr("class", "tooltip")
      .style("background-color", "white")
      .style("border", "solid")
      .style("border-width", "2px")
      .style("border-radius", "5px")
      .style("padding", "5px")

    for (let data of this.dataset) {
      content
        .append('circle')
        .datum(data)
        .attr('class', 'mouse-circle')
        .style('display', 'none')
        .attr('stroke', data.color)
        .attr('stroke-width', 2)
        .attr('fill', 'none')
        .attr('r', 4)
        .attr('cx', x(2030))
        .attr('cy', y(data.y[0]))
      ;
    }

    let mouseover = function(d) {
      let mouseP = d3.mouse(this)
      updateTooltip(d3.event, mouseP);
      //tooltip.style("display", "");
      //content.selectAll('.mouse-circle').style("display", "");
      updateCircles(d3.event, mouseP);
    };

    let mousemove = function(d) {
      let mouseP = d3.mouse(this)
      updateTooltip(d3.event, mouseP);
      updateCircles(d3.event, mouseP);
    };

    let mouseleave = function(d) {
      tooltip.style("display", "none");
      content.selectAll('.mouse-circle').style("display", "none");
    };

    let updateTooltip = function(event, mouseP) {
      let date = x.invert(mouseP[0]);
      tooltip
        .html("Year: " + date.toFixed(1))
        .style("left", (event.pageX + 20) + "px")
        .style("top", (event.pageY - 20) + "px")
    }

    let updateCircles = function(event, mouseP) {
      let date = currentX.invert(mouseP[0]);

      content.selectAll('.mouse-circle')
        .attr('cx', mouseP[0])
        .attr('cy', data => {
          // Poor way of doing this
          let index = d3.bisect(data.x, date);

          if (index == 0) {
            return currentY(data.y[0]);
          } else {
            let beta = (data.x[index] - date)/(data.x[index] - data.x[index-1]);
            return currentY(data.y[index] - beta*(data.y[index] - data.y[index-1]));
          }
        })
      ;
    }

    let self = this;

    // Zooming
    let zoom = d3.zoom()
      .scaleExtent([1, 20])
      .extent([[0, 0], [width, height]])
      .translateExtent([[0, 0], [width, height]])
      .on("zoom", updateChart)
    ;

    let overlay = content.append("rect")
      .attr("width", width)
      .attr("height", height)
      .style("fill", "none")
      .style("pointer-events", "all")
      .call(zoom.transform, self.lastTransform || d3.zoomIdentity)
      .call(zoom)
    ;

    overlay
      .on('mouseover', mouseover)
      .on('mousemove', mousemove)
      .on('mouseleave', mouseleave)
    ;

    content.on('wheel', (e) => {
      d3.event.preventDefault()
      return false;
    });

    function updateChart(transform) {
      currentX = d3.event.transform.rescaleX(x);
      //currentY = d3.event.transform.rescaleY(y);

      self.lastTransform = d3.event.transform;

      xAxis.call(d3.axisBottom(currentX));
      yAxis.call(d3.axisLeft(currentY));

      content
        .selectAll('.data-path')
        .select(function() {
          // Sigh
          d3.select(this).attr("d", d3.line()
            .x(d => currentX(d.x))
            .y(d => {
              return currentY(d.y) - (parseFloat(this.dataset.displacement) || 0)
            })
          );
        })
      ;

      content
        .selectAll('.axvline')
        .attr('x1', d => currentX(d) )
        .attr('x2', d => currentX(d) )
      ;

      content
        .selectAll('.axhline')
        .attr('y1', d => currentY(d) )
        .attr('y2', d => currentY(d) )
      ;
    }
  }

  add_header() {
    this.nodes.header = document.createElement('div');
    this.nodes.header.classList.add('graph-header');
    return this.nodes.header;
  }

  render() {
    let nodes = this.nodes;

    nodes.plot = document.createElement('div');
    nodes.plot.classList.add('graph-plot');

    nodes.wrapper.append(nodes.plot);

    if (this.top_selections.length || this.side_selections.length) {
      // This is an interactive graph. Let the manager generate the first state.
      this.on_select_callback(this.selected_top, this.selected_side, this);
    }

    if (this.top_selections.length) {
      this.add_header();

      let top_selector = document.createElement('div');
      top_selector.classList.add('selector');
      let labels = [];
      for (let field of this.top_selections) {
        let label = document.createElement('label');
        label.innerHTML = field;
        top_selector.append(label);
        label.addEventListener('click', () => {
          this.selected_top = field;
          this.on_select_callback(this.selected_top, this.selected_side, this);
          for (let label of labels) {
            label.classList.remove('active');
          }
          label.classList.add('active');
          this.update();
        });
        labels.push(label);

        if (field == this.selected_top) {
          label.classList.add('active');
        }
      }
      this.nodes.header.prepend(top_selector);
    }

    if (this.side_selections.length) {
      nodes.side = document.createElement('div');
      nodes.side.classList.add('graph-side');
      nodes.wrapper.append(nodes.side);

      nodes.wrapper.classList.add('with-side-panel');

      let side_selector = document.createElement('div');
      side_selector.classList.add('selector');
      let labels = [];
      for (let field of this.side_selections) {
        let label = document.createElement('label');
        label.innerHTML = field;
        side_selector.append(label);
        label.addEventListener('click', () => {
          this.selected_side = field;
          this.on_select_callback(this.selected_top, this.selected_side, this);
          for (let label of labels) {
            label.classList.remove('active');
          }
          label.classList.add('active');
          this.update();
        });
        labels.push(label);

        if (field == this.selected_side) {
          label.classList.add('active');
        }
      }
      this.nodes.side.append(side_selector);
    }

    this.update();
  }

  get_dasharray(dash_type) {
    let types = {
      'dotted': '2,5',
      '-.':     '3,1',
      'dashed': '4,4',
    };

    let dasharray = (dash_type in types) ? types[dash_type] : dash_type;

    return dasharray;
  }
}

class Plotter {
  constructor(container) {
    this.container = nodify(container);
    this.defaults = {
      xscale: 'linear',
      yscale: 'linear',
      axline_color: 'grey',
      axline_linestyle: 'dashed',
    };
    this.reset();
  }

  reset() {
    this.graph = new Graph(this.defaults);
  }

  get_dataset() {
    return this.graph.get_dataset();
  }

  set_title(title) {
    this.graph.set_title(title);
  }

  set_top_selections(fields, selection) {
    this.graph.top_selections = fields;
    this.graph.selected_top = (typeof selection === 'undefined') ? fields[0] : selection;
  }

  set_side_selections(fields, selection) {
    this.graph.side_selections = fields;
    this.graph.selected_side = (typeof selection === 'undefined') ? fields[0] : selection;
  }

  on_select(callback) {
    this.graph.on_select_callback = callback;
  }

  xscale(scale) {
    this.graph.xscale(scale);
  }

  yscale(scale) {
    this.graph.yscale(scale);
  }

  axvline(x, options = {}) {
    this.graph.axvline(x, options);
  }

  axhline(y, options = {}) {
    this.graph.axhline(y, options);
  }

  set_defaults(defaults) {
    override(this.defaults, defaults);
    if ('xscale' in defaults) this.graph.xscale(this.defaults.xscale);
    if ('yscale' in defaults) this.graph.yscale(this.defaults.yscale);
  }

  plot(x, y, options = {}) {
    this.graph.add_data_xy(x, y, options);
  }

  show(container) {
    this.graph.attach(container ? nodify(container) : this.container);
    this.reset();
  }

  clear(container) {
    (container ? nodify(container) : this.container).innerHTML = '';
  }
}

let plt = new Plotter();

function nodify(node_or_query) {
  if (typeof(node_or_query) == 'string') {
    return document.querySelector(node_or_query);
  }
  return node_or_query;
}

function plot_compute_decomposition(sim, container, crop_after_agi = true) {
  let t = sim.timesteps;

  let bioanchors = run_bioanchors_model({
    t_start: sim.t_start,
    t_end: sim.timesteps[sim.timesteps.length-1],

    t_step: best_guess_sim.consts.t_step,
    initial_software: best_guess_sim.states[0].software.v,
    initial_hardware: best_guess_sim.states[0].hardware_performance.v,
    initial_training_investment: 50 * best_guess_sim.states[0].compute_investment * best_guess_sim.states[0].frac_compute.training.v
  });

  let graph = plt.graph;

  let header = graph.add_header();
  header.innerHTML =
    '<input type="checkbox" id="bioanchors-button"><label for="bioanchors-button">Compare with bioanchors</label>' +
    '<input type="checkbox" id="normalized-button" style="margin-left: 2em"><label for="normalized-button">Normalized at wake-up year</label>';

  let show_bioanchors_button = header.querySelector('#bioanchors-button');
  let show_normalized_button = header.querySelector('#normalized-button');

  if (ui_state.show_bioanchors) {
    show_bioanchors_button.checked = true;
  }

  if (ui_state.show_normalized_decomposition) {
    show_normalized_button.checked = true;
  }

  function add_data(t, v, line_options) {
    let start_idx = 0;
    let reference_idx = (sim.rampup_start != null) ? nj.argmax(nj.gte(t, sim.rampup_start)) : 0;
    let end_idx = (crop_after_agi && sim.agi_year != null) ? nj.argmax(nj.gte(t, sim.agi_year + 5)) : t.length;

    t = t.slice(start_idx, end_idx);
    v = v.slice(start_idx, end_idx);
    if (ui_state.show_normalized_decomposition) {
      v = nj.div(v, v[reference_idx]);
    }

    graph.add_data_xy(t, v, line_options);
  }

  function update_graph() {
    graph.clear_dataset();

    let section = ui_state.show_bioanchors ? 'Our model' : null;

    if (!ui_state.show_bioanchors) {
      add_data(t, sim.get_thread('compute_investment'), {label: '$ on FLOP globally', color: 'blue', section});
    } else {
      add_data(
        bioanchors.timesteps,
        nj.mult(sim.get_thread('compute_investment'), sim.get_thread('frac_compute.training.v')),
        {label: 'Training compute investment', color: 'purple', section}
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

      add_data(bioanchors.timesteps, bioanchors.training_investment, {displacement: +displacement, label: 'Training compute investment', color: 'purple', linestyle: linestyle, section});
      add_data(bioanchors.timesteps, bioanchors.hardware,            {displacement: 0, label: 'Hardware (FLOP/$)', color: 'orange', linestyle: linestyle, section});
      add_data(bioanchors.software_timesteps, bioanchors.software,   {displacement: -displacement, label: 'Software (2022-FLOP per FLOP)', color: 'green', linestyle: linestyle, section});
    }

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

  plot_vlines(sim)

  plt.set_title('Compute increase decomposition');

  plot_oom_lines();

  plt.show(container);
}

function add_multigraph(sim, variables, container, crop_after_agi = true) {
  let label_to_var = {};
  let labels = [];
  for (let o of variables) {
    labels.push(o.label);
    label_to_var[o.label] = o.var;
  }

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
      let r = sim.get_growth(label_to_var[side]);
      x = r.t;
      y = r.growth;
      graph.yscale('linear');
    } else {
      x = sim.timesteps;
      y = sim.get_thread(label_to_var[side]);
      graph.yscale('log');
    }

    if (crop_after_agi) {
      let end_idx = (crop_after_agi && sim.agi_year != null) ? Math.min(sim.time_to_index(sim.agi_year + 5), sim.states.length) : sim.states.length;
      x = x.slice(0, end_idx);
      y = y.slice(0, end_idx);
    }

    let label = side;
    if (show_growth) {
      label += ' growth';
    }
    graph.add_data_xy(x, y, {label: label});
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

function plot_vlines(sim, line_color = 'black') {
  if (sim.rampup_start) {
    plt.axvline(sim.rampup_start, {
      linestyle: 'dotted',
      color: line_color,
      label: 'Wake-up',
    });
  }
              
  if (sim.rampup_mid) {
    plt.axvline(sim.rampup_mid, {
      linestyle: '-.',
      color: line_color,
      label: '30% automation',
    });
  }
              
  if (sim.agi_year) {
    plt.axvline(sim.agi_year, {
      linestyle: 'dashed',
      color: line_color,
      label: 'Full automation',
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

function add_table(container, table) {
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

let parameter_names = {"full_automation_requirements_training": "AGI training requirements", "flop_gap_training": "FLOP gap (training)", "goods_vs_rnd_requirements_training": "Training goods vs R&D", "full_automation_requirements_runtime": "AGI runtime requirements", "flop_gap_runtime": "FLOP gap (runtime)", "goods_vs_rnd_requirements_runtime": "Runtime goods vs R&D", "runtime_training_tradeoff": "Trade-off efficiency", "runtime_training_max_tradeoff": "Maximum trade-off", "labour_substitution_goods": "Labour substitution goods", "labour_substitution_rnd": "Labour substitution R&D", "capital_substitution_goods": "Substitution between capital and cognitive tasks for goods and services", "capital_substitution_rnd": "Substitution between capital and cognitive tasks for R&D", "hardware_returns": "Returns to hardware", "software_returns": "Returns to software", "hardware_performance_ceiling": "Maximum hardware performance", "software_ceiling": "Maximum software performance", "rnd_parallelization_penalty": "R&D parallelization penalty", "hardware_delay": "Hardware adoption delay", "frac_capital_hardware_rnd_growth": "Growth rate fraction capital hardware R&D", "frac_labour_hardware_rnd_growth": "Growth rate fraction labour hardware R&D", "frac_compute_hardware_rnd_growth": "Growth rate fraction compute hardware R&D", "frac_labour_software_rnd_growth": "Growth rate fraction labour software R&D", "frac_compute_software_rnd_growth": "Growth rate fraction compute software R&D", "frac_gwp_compute_growth": "Growth rate fraction GWP compute", "frac_compute_training_growth": "Growth rate fraction compute training", "frac_capital_hardware_rnd_growth_rampup": "Wake-up growth rate fraction capital hardware R&D", "frac_labour_hardware_rnd_growth_rampup": "Wake-up growth rate fraction labour hardware R&D", "frac_compute_hardware_rnd_growth_rampup": "Wake-up growth rate fraction compute hardware R&D", "frac_labour_software_rnd_growth_rampup": "Wake-up growth rate fraction labour software R&D", "frac_compute_software_rnd_growth_rampup": "Wake-up growth rate fraction of compute software R&D", "frac_gwp_compute_growth_rampup": "Wake-up growth rate fraction of GWP buying compute", "frac_compute_training_growth_rampup": "Wake-up growth rate fraction compute training AI models", "frac_capital_hardware_rnd_ceiling": "Max fraction capital hardware R&D", "frac_labour_hardware_rnd_ceiling": "Max fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_ceiling": "Max fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_ceiling": "Max fraction labour software R&D", "frac_compute_software_rnd_ceiling": "Max fraction compute software R&D", "frac_gwp_compute_ceiling": "Max fraction GWP compute", "frac_compute_training_ceiling": "Max fraction compute training", "initial_frac_capital_hardware_rnd": "Initial fraction capital hardware R&D", "initial_frac_labour_hardware_rnd": "Initial fraction labour hardware R&D", "initial_frac_compute_hardware_rnd": "Initial fraction compute hardware R&D", "initial_frac_labour_software_rnd": "Initial fraction labour software R&D", "initial_frac_compute_software_rnd": "Initial fraction compute software R&D", "initial_biggest_training_run": "Initial biggest training run", "ratio_initial_to_cumulative_input_hardware_rnd": "Initial vs cumulative input - hardware R&D", "ratio_initial_to_cumulative_input_software_rnd": "Initial vs cumulative input - software R&D", "initial_hardware_production": "Initial hardware production", "ratio_hardware_to_initial_hardware_production": "Accumulated hardware vs initial hardware production", "initial_buyable_hardware_performance": "Initial market hardware performance", "initial_gwp": "Initial GWP", "initial_population": "Initial population", "initial_cognitive_share_goods": "Initial cognitive share goods", "initial_cognitive_share_rnd": "Initial cognitive share in R&D", "initial_compute_share_goods": "Initial compute share goods", "initial_compute_share_rnd": "Initial compute share R&D", "rampup_trigger": "Wakeup trigger", "initial_capital_growth": "Initial capital growth rate", "labour_growth": "Population growth rate", "tfp_growth": "TFP growth rate", "compute_depreciation": "Compute depreciation rate", "money_cap_training_before_wakeup": "Money cap training before wakeup"};
let parameter_meanings = {"full_automation_requirements_training": "FLOP needed to train an AI capable of performing the most hard-to-train cognitive task.", "flop_gap_training": "Ratio between the FLOP needed to train the most demanding AI task and the 20% most demanding task.", "goods_vs_rnd_requirements_training": "Ratio between the training requirements for fully automating goods and services production vs (hardware and software) R&D. The higher this is, the easier it is to automate R&D compared to goods and services.", "full_automation_requirements_runtime": "FLOP/year needed to substitute the output of one human worker in the most demanding AI task.", "flop_gap_runtime": "Ratio between the FLOP/year needed to automate the most demanding AI task and the 20% most demanding AI task.", "goods_vs_rnd_requirements_runtime": "Ratio between the FLOP/year needed to automate the most demanding cognitive task for goods production and for R&D. The higher this is, the easier it is to automate R&D with respect to goods and services production. This incorporates one-time gains for AIs doing R&D that are additional to the one-time gains for AIs providing goods and services.", "runtime_training_tradeoff": "How efficiently you can substitute training for higher runtime compute requirements. A value of 0 or less indicates that you cannot perform this tradeoff at all. A value of N indicates that a 10X smaller training run increases runtime requirements by 10^N.", "runtime_training_max_tradeoff": "The maximum ratio of training that can be substituted with extra runtime", "labour_substitution_goods": "Substitution between cognitive tasks for goods and services.\nA more positive number means that it is easier to substitute the output of different cognitive tasks in the production of goods", "labour_substitution_rnd": "Substitution between cognitive tasks for R&D.\nA more positive number means that it is easier to substitute the output of different cognitive tasks in R&D", "capital_substitution_goods": "Substitution between capital and cognitive tasks for goods and services.\nA more positive number means that it is easier to substitute cognitive output by capital and viceversa in the production of goods ", "capital_substitution_rnd": "Substitution between capital and cognitive tasks for R&D.\nA more positive number means that it is easier to substitute cognitive output by capital and viceversa in R&D", "hardware_returns": "How cumulative inputs to hardware R&D translate to better performance improvements. Doubling the cumulative input of harware R&D results in 2^hardware_returns improvement of hardware_efficiency (not accounting for the damping effect of the ceiling)", "software_returns": "How cumulative inputs to software R&D translate to better performance improvements. Doubling the cumulative input of software R&D results in 2^software_returns improvement of hardware_efficiency (not accounting for the damping effect of the ceiling)", "hardware_performance_ceiling": "Maximum FLOP/year/$ achiveable. Performance improvements get increasingly penalized as they approach this quantitiy.", "software_ceiling": "Maximum 2020-FLOP / physical FLOP achievable. Performance improvements get increasingly penalized as they approach this quantitiy.", "rnd_parallelization_penalty": "Penalization to concurrent R&D efforts. The ouputs of the hardware and software production function R&D get raised to this penalty before being aggregated to the cumulative total.", "hardware_delay": "Years between a chip design and its commercial release.", "frac_capital_hardware_rnd_growth": "Growth rate of fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_growth": "Growth rate of fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_growth": "Growth rate of fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_growth": "Growth rate of fraction of labour dedicated to software R&D", "frac_compute_software_rnd_growth": "Growth rate of fraction of compute dedicated to software R&D", "frac_gwp_compute_growth": "Growth rate of fraction of GWP dedicated to buying compute", "frac_compute_training_growth": "Growth rate of fraction of compute dedicated to training AI models", "frac_capital_hardware_rnd_growth_rampup": "Post wake-up growth rate of fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_growth_rampup": "Post wake-up rate of fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_growth_rampup": "Post wake-up rate of fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_growth_rampup": "Post wake-up rate of fraction of labour dedicated to software R&D", "frac_compute_software_rnd_growth_rampup": "Post wake-up rate of fraction of compute dedicated to software R&D", "frac_gwp_compute_growth_rampup": "Post wake-up rate of fraction of GWP dedicated to buying compute", "frac_compute_training_growth_rampup": "Post wake-up rate of fraction of compute dedicated to training AI models", "frac_capital_hardware_rnd_ceiling": "Maximum of the fraction of capital dedicated to hardware R&D", "frac_labour_hardware_rnd_ceiling": "Maximum of the fraction of labour dedicated to hardware R&D", "frac_compute_hardware_rnd_ceiling": "Maximum of the fraction of compute dedicated to hardware R&D", "frac_labour_software_rnd_ceiling": "Maximum of the fraction of labour dedicated to software R&D", "frac_compute_software_rnd_ceiling": "Maximum of the fraction of compute dedicated to software R&D", "frac_gwp_compute_ceiling": "Maximum of the fraction of GWP dedicated to buying compute", "frac_compute_training_ceiling": "Maximum of the fraction of compute dedicated to training AI models", "initial_frac_capital_hardware_rnd": "Initial fraction of capital dedicated to hardware R&D", "initial_frac_labour_hardware_rnd": "Initial fraction of labour dedicated to hardware R&D", "initial_frac_compute_hardware_rnd": "Initial fraction of compute dedicated to hardware R&D", "initial_frac_labour_software_rnd": "Initial fraction of labour dedicated to software R&D", "initial_frac_compute_software_rnd": "Initial fraction of compute dedicated to software R&D", "initial_biggest_training_run": "Amount of compute used for the largest AI training run up to date, measured in current-FLOP", "ratio_initial_to_cumulative_input_hardware_rnd": "Ratio of budget of hardware R&D in 2022 versus cumulative budget through history", "ratio_initial_to_cumulative_input_software_rnd": "Ratio of budget of software R&D in 2022 versus cumulative budget through history", "initial_hardware_production": "FLOP/year capabilities of hardware produced in 2022", "ratio_hardware_to_initial_hardware_production": "FLOP/year capabilities of total available hardware in 2022 vs hardware produced in 2022", "initial_buyable_hardware_performance": "FLOP/year/$ of the most cost efficient chip available in the market today.", "initial_gwp": "Initial real global world product. Measured in 2020 USD.", "initial_population": "Initial world population.", "initial_cognitive_share_goods": "Initial fraction of the GWP captured by labour and compute owners vs capital owners", "initial_cognitive_share_rnd": "Initial fraction of the R&D captured by labour and compute owners vs capital owners", "initial_compute_share_goods": "Initial fraction of cognitive output done by computers for goods production", "initial_compute_share_rnd": "Initial fraction of cognitive output done by computers for R&D", "rampup_trigger": "Threshold of task automation in the production of goods and services that triggers massive investment growth.", "initial_capital_growth": "Initial rate of growth of capital", "labour_growth": "Population yearly growth rate. Controls the growth of labour.", "tfp_growth": "Yearly rate of growth of the total factor productivity. Each year the production gets passively better by a factor TFP that grows at this rate.", "compute_depreciation": "Yearly rate of depreciation of hardware. Each year this portion of the accumulated compute is depreciated.", "money_cap_training_before_wakeup": NaN};
let parameter_justifications = {"": "", "full_automation_requirements_training": "I'm mostly anchoring to the Bio Anchors report, with an extra OOM to account for TAI being a lower bar than full automation (AGI).", "flop_gap_training": "See discussion in <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1Z7HJ9pHctgDi1XYbgRW9-7J1bxTL98KW1qb7HN7Mv-A/edit#heading=h.grg4srb18f02\" target=\"_blank\">summary</a></span> and <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.o4db3tcgrq28\" target=\"_blank\">full report</a></span>.", "goods_vs_rnd_requirements_training": "I expect fully automating the cognitive tasks in good production or R&amp;D will require AI to (collectively) be extremely general and flexible. I have slightly lower requirements for R&amp;D because it may be easier to gather data, and there may be fewer regulatory restrictions. I think you could defend even lower requirements for software R&amp;D.", "full_automation_requirements_runtime": "I anchor on the Bio Anchors report (~1e16 FLOP/s). Then I adjust upwards by 1 OOM to account for TAI being a lower bar than full automation (AGI). Then I adjust downwards by 6X to account for <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.wi3vto6myw5\" target=\"_blank\">one-time advantages</a></span> for AGI over humans in goods and services.", "flop_gap_runtime": "The spread of runtime requirements is smaller than the spread of training requirements for two reasons. First, a 10X increase in runtime compute typically corresponds to a 100X increase in training, e.g. for Chinchilla scaling. Secondly, increasing the horizon length of training tasks will increase training compute but not runtime.", "goods_vs_rnd_requirements_runtime": "I estimate the <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.wi3vto6myw5\" target=\"_blank\">one-time gains</a></span> of AGIs over humans to be 60X in R&amp;D vs 6X in goods and services. This is a difference of 10X.", "runtime_training_tradeoff": "I describe the tradeoff dynamic <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.wa8izz23etjx\" target=\"_blank\">here</a></span>.", "runtime_training_max_tradeoff": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#bookmark=id.kcldlzp2l745\" target=\"_blank\">here</a></span>.", "labour_substitution_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.qazoq7gf2vgm\" target=\"_blank\">here</a></span>.", "labour_substitution_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.9t2n0pf04b2e\" target=\"_blank\">here</a></span>.", "capital_substitution_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.fo846mq256bx\" target=\"_blank\">here</a></span>.", "capital_substitution_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/15EmltGq-kkiLO95AbvoB4ODVpyg26BgghvHBy1JDyZY/edit#heading=h.o7tmwweugbb\" target=\"_blank\">here</a></span>.", "hardware_returns": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.9us1ymg9hau0\" target=\"_blank\">here</a></span>.", "software_returns": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yzbcl83o650l\" target=\"_blank\">here</a></span>.", "hardware_performance_ceiling": "Based on a rough estimate from a technical advisor. They guessed energy prices could eventually fall 10X from today to $0.01/kWh. And that (based on <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#56a3f1;\"><a href=\"https://en.wikipedia.org/wiki/Landauer%27s_principle\" target=\"_blank\">Landauer\u2019s limit</a></span>) you might eventually do 1e27 bit erasures per kWh. That implies 1e29 bit erasures per $. If we do 1 FLOP per bit, that's 1e29 FLOP/$. ", "software_ceiling": "If training AGI currently requires 1e36 FLOP, but this can be reduced to human life-time learning FLOP of 1e29, that's 7 OOMs improvement. ", "rnd_parallelization_penalty": "Econ models often use a value of 1. A prominent growth economist thought that values between 0.4 and 1 are reasonable. If you think adding new people really won't help much with AI progress, you could use a lower value.", "hardware_delay": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.2gnv7nk1tdrv\" target=\"_blank\">here</a></span>. The conservative value of 2.5 years corresponds to an <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://semiwiki.com/semiconductor-manufacturers/tsmc/2212-how-long-does-it-take-to-go-from-a-muddy-field-to-full-28nm-capacity/\" target=\"_blank\">estimate</a></span> of the time needed to make a new fab. The aggressive value (no delay) corresponds to fabless improvements in chip design that can be printed with existing production lines with ~no delay. ", "frac_capital_hardware_rnd_growth": "Real $ investments in hardware R&amp;D have <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1bGbzR0c3TqsRYTWS3s6Bysgh9ZOuKS1w6qH1SOI11iE/edit#gid=186138651\" target=\"_blank\">recently grown</a></span> at ~4%; subtracting out ~3% GWP growth impleis ~1% growth in the fraction of GWP invested.", "frac_labour_hardware_rnd_growth": "As above.", "frac_compute_hardware_rnd_growth": "As above.", "frac_labour_software_rnd_growth": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.1v8m5dp6xefi\" target=\"_blank\">here</a></span>, calcs <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1qmiomnNLpjcWSaeT54KC1PH1hfi_jUFIkWszxJGVU5w/edit#gid=0\" target=\"_blank\">here</a></span>. I subtract out 1% population growth from 19% growth in #reseachers (which is smaller than the 20% growth in real $).", "frac_compute_software_rnd_growth": "As above.", "frac_gwp_compute_growth": "Assumed to be equal to the growth rate post \"wake up\" (below). Why? Demand for AI chips is smaller today than after ramp-up, pushing towards slower growth today. But growth today is from a smaller base, and can come from the share of GPUs growing as a fraction of semiconductor production (which won't be possible once it's already ~100% of production). I'm lazily assuming these effects cancel out.", "frac_compute_training_growth": "This corresponds to the assumption that there will be a $4b training run in 2030, in line with Bio Anchors' prediction. Discussed a little <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yyref6x2mzxu\" target=\"_blank\">here</a></span>.", "frac_capital_hardware_rnd_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.612idx97x187\" target=\"_blank\">here</a></span>. I substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.", "frac_labour_hardware_rnd_growth_rampup": "As above.", "frac_compute_hardware_rnd_growth_rampup": "I assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&amp;D.", "frac_labour_software_rnd_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.vi6088puv22e\" target=\"_blank\">here</a></span>. I substract out 3% annual GWP growth to calculate the growth in the <span style=\"font-style:italic;\">fraction</span> of GWP invested.", "frac_compute_software_rnd_growth_rampup": "I assume a one-year doubling as compute can be easily reallocated to the now-extremely-lucrative field of AI R&amp;D.", "frac_gwp_compute_growth_rampup": "Discussed here. I substract out 3% annual GWP growth to calculate the growth in the fraction of GWP invested.", "frac_compute_training_growth_rampup": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.5xk4lbt60vr0\" target=\"_blank\">here</a></span>.", "frac_capital_hardware_rnd_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48\" target=\"_blank\">here</a></span>.", "frac_labour_hardware_rnd_ceiling": "As above.", "frac_compute_hardware_rnd_ceiling": "If people anticipate an AI driven singularity, the demand for progress in AI R&amp;D should become huge, such that the world allocates using a macroscopic of compute to AIs doing Ai R&amp;D.", "frac_labour_software_rnd_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.7ug8akmppx48\" target=\"_blank\">here</a></span>.", "frac_compute_software_rnd_ceiling": "If people anticipate an AI driven singularity, the demand for progress in AI R&amp;D should become huge, such that the world allocates using a macroscopic of compute to AIs doing Ai R&amp;D.", "frac_gwp_compute_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.rh3lt1q0f0hl\" target=\"_blank\">here</a></span>.", "frac_compute_training_ceiling": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.tz6v7gxroefr\" target=\"_blank\">here</a></span>.", "initial_frac_capital_hardware_rnd": "Hardware R&amp;D spend is <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://www.onlinecomponents.com/en/blogpost/2021-semiconductor-rd-spend-to-rise-4-357/\" target=\"_blank\">~$70b</a></span>, which is ~0.1%. I double this to include some of the spending on new capital equipment (which totals <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://www.statista.com/statistics/864897/worldwide-capital-spending-in-the-semiconductor-industry/\" target=\"_blank\">~$130b</a></span>).", "initial_frac_labour_hardware_rnd": "As above.", "initial_frac_compute_hardware_rnd": "As above.", "initial_frac_labour_software_rnd": "I'd guess annual spending on software workers for SOTA AI is $10-20b (DM spend is ~$1b); $16b is 0.02%.", "initial_frac_compute_software_rnd": "As above.", "initial_biggest_training_run": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.b15qyylbeqxp\" target=\"_blank\">here.</a></span>", "ratio_initial_to_cumulative_input_hardware_rnd": "See <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1bGbzR0c3TqsRYTWS3s6Bysgh9ZOuKS1w6qH1SOI11iE/edit#gid=186138651\" target=\"_blank\">calcs</a></span>, the cell called \"2022 inputs / cumulative inputs\"", "ratio_initial_to_cumulative_input_software_rnd": "Calculated assuming inputs have always been growing at 20%.", "initial_hardware_production": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.yyref6x2mzxu\" target=\"_blank\">here</a></span>.", "ratio_hardware_to_initial_hardware_production": "Assumes annual FLOP production has been growing exponentially at 50%; ~28% from more FLOP/$ and ~22% from more $ spending.", "initial_buyable_hardware_performance": "The training of Palm cost <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://blog.heim.xyz/palm-training-cost/\" target=\"_blank\">~$10m</a></span>, and used 3e24 FLOP. This implies 3e17 FLOP/$. If companies renting chips make back their money over 2 years then that corresponds to 1.5e17 FLOP/year/$.", "initial_gwp": "See value for 2020 <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?locations=1W\" target=\"_blank\">here</a></span>.", "initial_population": "<a href=\"https://www.google.com/search?q=world+population+&amp;ei=bdAOY5mXFvXg0PEP9eeygAY&amp;ved=0ahUKEwjZgePejPD5AhV1MDQIHfWzDGAQ4dUDCA4&amp;uact=5&amp;oq=world+population+&amp;gs_lcp=Cgdnd3Mtd2l6EAMyBwgAELEDEEMyCAgAEIAEELEDMgoIABCxAxCDARBDMgcIABCxAxBDMgcIABCxAxBDMgsIABCABBCxAxCDATIKCAAQsQMQgwEQQzIFCAAQgAQyBQgAEIAEMgcIABCxAxBDOgcIABBHELADSgUIPBIBMUoECEEYAEoECEYYAFDHBljXB2DaCWgBcAF4AIABS4gBiQGSAQEymAEAoAEByAEIwAEB&amp;sclient=gws-wiz\" target=\"_blank\">Source.</a>", "initial_cognitive_share_goods": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.osubru9caix\" target=\"_blank\">here</a></span>.", "initial_cognitive_share_rnd": "Discussed <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#bookmark=id.myon1lrlwj34\" target=\"_blank\">here</a></span>.", "initial_compute_share_goods": "The semiconductor industry's annual revenues are ~$600b, which is ~1% of GWP.", "initial_compute_share_rnd": "As above. (I'm assuming compute is used equally in hardware+software as in the general economy - excluding compute used on training runs.)", "rampup_trigger": "AI that can readily automate 3% of cognitive tasks can readily generate ~$1.5tr. (Assuming cognitive work is paid ~$50tr, ~50% of GDP). Of course, there will be a lag before this value is added, but people will also invest on the basis of promising demos.", "initial_capital_growth": "Recent GWP growth is ~3%; in equilibrium capital grows at the same rate as GDP.", "labour_growth": "<a href=\"https://www.google.com/search?q=world+population+growth+rate&amp;ei=VdAOY_7TDqu00PEP6pOViAM&amp;ved=0ahUKEwi-0qLTjPD5AhUrGjQIHepJBTEQ4dUDCA4&amp;uact=5&amp;oq=world+population+growth+rate&amp;gs_lcp=Cgdnd3Mtd2l6EAMyCAgAEIAEELEDMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIICAAQgAQQyQMyBQgAEIAEOgcIABBHELADOgQIABBDSgUIPBIBMUoECEEYAEoECEYYAFCOBVjcCGDnCWgBcAF4AIABVogB0gKSAQE1mAEAoAEByAEIwAEB&amp;sclient=gws-wiz\" target=\"_blank\">Source.</a>", "tfp_growth": "Average TFP growth over the last 20 years, <span style=\"text-decoration:underline;-webkit-text-decoration-skip:none;text-decoration-skip-ink:none;color:#1155cc;\"><a href=\"https://docs.google.com/spreadsheets/d/1C-RUowD3Nwo51UF5ZeBjbeLwaw4HQ1o13KyJlhuXCcU/edit#gid=2116796644\" target=\"_blank\">source</a></span>.", "compute_depreciation": "Rough guess.", "money_cap_training_before_wakeup": ""};
let metric_names = {"cog_output_multiplier": "Time to tenfold cognitive output multiplier", "billion_agis": "Time to billion AGIs", "full_automation": "Time to full automation", "rampup_to_agi": "Time from wake-up and full automation", "combined": "Combined", "gwp_growth": "Time to 20% GWP growth", "agi_year": "AGI year", "rampup_start": "Wake-up year", "doubling_times": "Pattern of GWP doublings", "billion_agis skew": "Skew of time to billion AGI", "agi_date skew": "Skew of time to AGI date"};
let metric_meanings = {"cog_output_multiplier": "Years between <a href=\"https://docs.google.com/document/d/1rw1pTbLi2brrEP0DcsZMAVhlKp6TKGKNUSFRkkdP_hs/edit#heading=h.vg16ijfr9w9a/\">AI's cognitive value-add</a> in hardware R&D being 2X and 10X.", "billion_agis": "Years between AI that automates 10% of cognitive tasks in R&D to the point when you can train and run 10 billion AGI-equivalents (taking the AGI runtime compute to be the runtime compute for the most compute-intensive R&D task).", "full_automation": "Years between AI that can perform 50% of cognitive tasks to AI that can perform 100%.", "rampup_to_agi": "Years between AI automating 3% of cognitive tasks in goods and services and AI that can perform 100% of cognitive tasks.", "combined": "Average of previous metrics.", "gwp_growth": "Years between instantaneous gwp growth reaching 5% and reaching 20%.\n\n(Not very meaningful because of zigzagging issues.)", "agi_year": "Year when all cognitive tasks have been automated", "rampup_start": "Year when AI becomes economically significant. This happens when a certain percentage of goods and services cognitive tasks have been automated (usually 3% but depends on the inputs) ", "doubling_times": "Years between successive GWP doublings", "billion_agis skew": "How the intervals are biased towards more aggressive or more conservative takeoffs.\nA more positive value means that the intervals are biased towards shorter takeoffs.\nComputed as |billion_agis_best_guess - billion_agis_conservative| - |billion_agis_aggressive - billion_agis_best_guess| ", "agi_date skew": "How the intervals are biased towards more aggressive or more conservative AGI dates.\nA more positive value means that the intervals are biased towards shorter takeoffs.\nComputed as |agi_date_best_guess - agi_date_conservative| - |agi_date_aggressive - agi_date_best_guess|"};
let variable_names = {"period": "Period", "year": "Year", "biggest_training_run": "Biggest training run", "hardware_performance": "Hardware performance", "software": "Software", "compute_investment": "Investment on compute", "gwp": "GWP", "capital": "Capital", "labour": "Labour", "tfp_rnd": "R&D TFP", "rnd_input_software": "Software R&D input", "cumulative_rnd_input_software": "Cumulative software R&D input", "frac_tasks_automated_goods": "Fraction of goods and services tasks automated", "frac_tasks_automated_rnd": "Fraction of R&D tasks automated", "frac_compute_training": "Fraction of compute invested in training"};

parameter_meanings['t_step'] = 'Duration of each simulation step (in years)';
parameter_meanings['t_end'] = 'When to stop the simulation';

for (let k of Object.keys(variable_names)) {
  variable_names[`${k} growth rate`] = `${variable_names[k]} growth rate`;
  variable_names[`${k} doubling time`] = `${variable_names[k]} doubling time`;
}

let tippy_instances = [];
for (let input of document.querySelectorAll('.input-parameter')) {
  let elements = input.querySelectorAll('input');
  let id = elements[elements.length-1].id;
  let meaning = parameter_meanings[id];
  let justification = parameter_justifications[id];

  let tooltip = '';

  if (meaning) {
    tooltip += meaning;
  }

  if (justification) {
    if (tooltip.length > 0) tooltip += '<br><br>';
    tooltip += '<span style="font-weight: bold">Justification for best guess value:</span> ' + justification;
  }

  if (tooltip.length > 0) tooltip += '<br><br>';
  tooltip += `<span style="font-weight: bold">key</span>: ${id}`

  tippy_instances.push(
    tippy(input, {
      content: tooltip,
      allowHTML: true,
      interactive: true,
      placement: 'right',
      appendTo: document.body,
      //arrow: false,
      hideOnClick: false,
      theme: 'light-border',
    })
  );
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
    tippy(node, {
      content: meaning,
      allowHTML: true,
      interactive: true,
      //placement: (node.parentElement.parentElement.tagName == 'THEAD') ? 'top' : 'right',
      theme: 'light-border',
      appendTo: document.body,
    });

    node._meaningInjected = true;
  }
}

// ------------------------------------------------------------------------
// And... run
// ------------------------------------------------------------------------

run_simulation(true, sim => {best_guess_sim = sim;});