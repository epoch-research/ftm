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
    this.transform = null;
    this.transform_update_callback = null;

    // You can overwrite this one
    this.tooltipBuilder = (x, ys) => {
      let tooltipHtml = [];
      tooltipHtml.push(`x: ${x.toFixed(1)}`);
      for (let yIndex = 0; yIndex < ys.length; yIndex++) {
        let y = ys[yIndex];
        let label = 'y';
        if (ys.length > 1) label += (yIndex+1);
        tooltipHtml.push(`${label}: ${(this.yAxis.scale == 'log') ? y.toExponential(1) : y.toFixed(1)}`);
      }
      return tooltipHtml.join('<br>');
    }

    this.width = 770;
    this.height = 520;
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

  clear_axvlines(x, options = {}) {
    this.axvlines = [];
  }

  clear_axhlines(y, options = {}) {
    this.axhlines = [];
  }

  set_title(title) {
    this.title = title;
  }

  add_data_xy(x, y, options = {}) {
    let o = {x, y, ...options};
    o.color = o.color || this.colors[(this.current_color++) % this.colors.length]
    o.linestyle = o.linestyle || '-',
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

  hide_tooltip() {
    this.nodes.tooltip.style("display", "none");
    this.nodes.content.selectAll('.tooltip-circle').style("display", "none");
  }

  update() {
    if (!this.nodes.plot) {
      // The graph hasn't been initialized yet. Exiting...
      return;
    }

    this.nodes.plot.innerHTML = "";
    if (this.nodes.legend) {
      this.nodes.legend.remove();
    }

    // set the dimensions and margins of the graph
    let margin = {top: 5, right: 10, bottom: 30, left: 40};
    let width = this.width - margin.left - margin.right;
    let height = this.height - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svgContainer = d3.select(this.nodes.plot)
      .append('div')
      .attr('class', 'svg-container')
    ;

    if (this.nodes.header) {
      svgContainer.node().append(this.nodes.header);
    }

    let svg = svgContainer
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
    this.nodes.content = content;

    let first = this.dataset[0];
    let indices = [...first.x.keys()];

    let xlims = [Infinity, -Infinity];
    let ylims = [Infinity, -Infinity];
    for (let data of this.dataset) {
      xlims[0] = Math.min(arrayMin(data.x), xlims[0]);
      xlims[1] = Math.max(arrayMax(data.x), xlims[1]);

      ylims[0] = Math.min(arrayMin(data.y), ylims[0]);
      ylims[1] = Math.max(arrayMax(data.y), ylims[1]);
    }

    if ((ylims[1] - ylims[0]) < 0.001) {
      ylims[0] /= 10;
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

    let gridV = content.append("g")
      .attr('class', 'grid')     
      .attr("transform", "translate(0," + height + ")")
      .call(d3.axisBottom(x).tickSize(-height))
    ;

    let gridH = content.append("g")
      .attr('class', 'grid')     
      .call(d3.axisLeft(y).tickSize(-width))
    ;

    let currentX = x;
    let currentY = y;

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
          .y(d => y(d.y) - (this.dataset.displacement ? parseFloat(this.dataset.displacement) : 0))
        );

      if (data.displacement) {
        path.node().dataset.displacement = data.displacement;
      }
    }

    let xAxis = svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .attr("stroke-width", 2)
      .call(d3.axisBottom(x).tickSizeOuter(0).tickFormat(d3.format("d")));

    let yAxis = svg.append("g")
      .attr("stroke-width", 2)
      .call(d3.axisLeft(y).tickSizeOuter(0));

    let crossHairV = svg.append("line")
      .attr("opacity", 0)
      .attr("y1", height + 5)
      .attr("y2", height - 5)
      .attr("stroke", "black")
      .attr("stroke-width", 1)
      .attr("pointer-events", "none");

    let crossHairH = svg.append("line")
      .attr("opacity", 0)
      .attr("x1", -5)
      .attr("x2", 5)
      .attr("stroke", "black")
      .attr("stroke-width", 1)
      .attr("pointer-events", "none");

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
      let legend_container;
      let top = (margin.top + 20) + 'px';
      let left = (margin.left + 20) + 'px';

      if (this.legend_placement == 'outside') {
        if (!this.nodes.side) {
          this.nodes.side = document.createElement('div');
          this.nodes.side.classList.add('graph-side');
          this.nodes.wrapper.append(this.nodes.side);

          this.nodes.wrapper.classList.add('with-side-panel');
        }

        legend_container = d3.select(this.nodes.side);
        left = 0;
        top = (margin.top + 5) + 'px';
      } else {
        legend_container = svgContainer;
      }

      let legend = legend_container
        .append('div')
          .style('top', top)
          .style('left', left)
          .attr('class', 'legend')
          .style('position', 'absolute')
      ;

      legend.attr('data-rd-drag-enabled', true);
      legend.attr('data-rd-resize-enabled', false);

      this.nodes.legend = legend;

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

    let tooltip = this.nodes.tooltip;
    if (!tooltip) {
      tooltip = d3.select('body')
        .append("div")
        .style("position", "absolute")
        .style("display", "none")
        .attr("class", "graph-tooltip")
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "2px")
        .style("border-radius", "5px")
        .style("padding", "5px")
      this.nodes.tooltip = tooltip;
    }

    for (let data of this.dataset) {
      content
        .append('circle')
        .datum(data)
        .attr('class', 'tooltip-circle')
        .style('display', 'none')
        .attr('fill', data.color)
        .attr('r', 4)
        .attr('cx', x(0))
        .attr('cy', y(data.y[0]))
      ;
    }

    let mouseover = function(d) {
      let mouseP = d3.mouse(this)
      if (self.tooltipBuilder) {
        updateTooltip(d3.event, mouseP);
      }
    };

    let mousemove = function(d) {
      let mouseP = d3.mouse(this)
      if (self.tooltipBuilder) {
        updateTooltip(d3.event, mouseP);
      }
      updateCrossHairs(mouseP);
    };

    let mouseleave = function(d) {
      tooltip.style("display", "none");
      content.selectAll('.tooltip-circle').style("display", "none");
    };

    let mouseout = function() {
      crossHairV.attr("opacity", 0);
      crossHairH.attr("opacity", 0);
    }

    function visualInterpolate(data, date) {
      let index = d3.bisect(data.x, date);

      let y;
      if (index == 0) {
        y = data.y[0];
      } else {
        let beta = (date - data.x[index-1])/(data.x[index] - data.x[index-1]);
        if (self.yAxis.scale == 'log') {
          y = data.y[index-1] * (data.y[index]/data.y[index-1])**beta;
        } else {
          y = data.y[index-1] + beta*(data.y[index] - data.y[index-1]);
        }
      }
      return y;
    }


    let updateTooltip = function(event, mouseP, changeDisplay=true) {
      let x = currentX.invert(mouseP[0]);

      let ys = [];
      content.selectAll('.tooltip-circle').each(function(data) {
        let y = visualInterpolate(data, x);
        ys.push(y);
      });

      let svgContainerBox = svgContainer.node().getBoundingClientRect();
      let contentBox = content.node().getBoundingClientRect();

      let tooltipContent = self.tooltipBuilder(x, ys, self);
      if (typeof tooltipContent == "string") {
        tooltipContent = html(tooltipContent);
      }

      tooltip
        .style("z-index", 99999)
        .style("left", (window.scrollX + svgContainerBox.x + mouseP[0] + 60) + "px")
        .style("top", (window.scrollY + svgContainerBox.y + mouseP[1] - 10) + "px")
        .html('')
        .append(() => tooltipContent)
      ;

      updateCircles(d3.event, mouseP);

      if (changeDisplay) {
        let someNaN = Number.isNaN(x);
        for (let y of ys) {
          if (Number.isNaN(y)) {
            someNaN = true;
          }
        }

        if (someNaN) {
          tooltip.style("display", "none");
          content.selectAll('.tooltip-circle').style("display", "none");
        } else {
          tooltip.style("display", "");
          content.selectAll('.tooltip-circle').style("display", "");
        }
      }
    }

    let updateCrossHairs = function(mouseP) {
      crossHairV.attr("x1", mouseP[0]).attr("x2", mouseP[0]).attr("opacity", 1);
      crossHairH.attr("y1", mouseP[1]).attr("y2", mouseP[1]).attr("opacity", 1);
    }

    let updateCircles = function(event, mouseP) {
      let x = currentX.invert(mouseP[0]);

      content.selectAll('.tooltip-circle')
        .attr('cx', mouseP[0])
        .attr('cy', data => {
          // Poor way of doing this
          return currentY(visualInterpolate(data, x));
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
      .call(zoom.transform, this.transform || d3.zoomIdentity)
      .call(zoom)
    ;

    overlay
      .on('mouseover', mouseover)
      .on('mousemove', mousemove)
      .on('mouseleave', mouseleave)
      .on('mouseout', mouseout)
    ;

    content.on('wheel', (e) => {
      d3.event.preventDefault()
      return false;
    });

    function updateChart(transform) {
      let mouseP = d3.mouse(this);
      if (!Number.isNaN(mouseP[0]) && !Number.isNaN(mouseP[1])) {
        updateCrossHairs(mouseP);
      }

      currentX = d3.event.transform.rescaleX(x);
      currentY = d3.event.transform.rescaleY(y);

      self.transform = d3.event.transform;
      if (self.transform_update_callback) {
        self.transform_update_callback(self.transform);
      }

      xAxis.call(d3.axisBottom(currentX).tickSizeOuter(0).tickFormat(d3.format("d")));
      yAxis.call(d3.axisLeft(currentY).tickSizeOuter(0));

      content
        .selectAll('.data-path')
        .select(function() {
          // Sigh
          d3.select(this).attr("d", d3.line()
            .x(d => currentX(d.x))
            .y(d => {
              return currentY(d.y) - (this.dataset.displacement ? parseFloat(this.dataset.displacement) : 0)
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

      gridV.call(d3.axisBottom(currentX).tickSize(-height));
      gridH.call(d3.axisLeft(currentY).tickSize(-width));

      if (!Number.isNaN(mouseP[0]) && !Number.isNaN(mouseP[1])) {
        updateTooltip(d3.event, mouseP, false);
      }
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
    this.graphToContainer = new Map();
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

  set_legend_placement(placement) {
    this.graph.legend_placement = placement;
  }

  set_transform(transform) {
    this.graph.transform = transform;
  }

  set_transform_callback(callback) {
    this.graph.transform_update_callback = callback;
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

  xlims(lims) {
    this.graph.xlims(lims);
  }

  ylims(lims) {
    this.graph.ylims(lims);
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

  set_tooltip(tooltipBuilder) {
    this.graph.tooltipBuilder = tooltipBuilder;
  }

  set_width(width) {
    this.graph.width = width;
  }

  set_height(height) {
    this.graph.height = height;
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
    container = container ? nodify(container) : this.container;
    this.graphToContainer.set(this.graph, container);
    this.graph.attach(container);
    let graph = this.graph;
    this.reset();
    return graph;
  }

  clear(container) {
    container = container ? nodify(container) : this.container;
    container.innerHTML = '';
    for (let [graph, cont] of this.graphToContainer.entries()) {
      if (cont == container) {
        for (let nodeName in graph.nodes) {
          graph.nodes[nodeName].remove();
        }
        this.graphToContainer.delete(graph);
      }
    }
  }
}
