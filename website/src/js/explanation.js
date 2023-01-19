"use strict";

let js_params = bridge.transform_python_to_js_params(best_guess_parameters);
let sim = ftm.run_simulation(js_params);

///////////////////////////////////////////////////////////////////////////////
// Plotting stuff

let plt = new Plotter();

function plot_vlines(sim, line_color = 'black', graph = null) {
  graph = graph || plt;

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

function plot_variable(sim, variable, container) {
  let t = sim.timesteps;
  let v = sim.get_thread(variable.thread)

  let crop_year = sim.timeline_metrics['automation_gns_100%'] + 5;
  let end_idx = (sim.timesteps[sim.timesteps.length-1] >= crop_year) ? nj.argmax(nj.gte(t, crop_year)) : t.length;

  t = t.slice(0, end_idx);
  v = v.slice(0, end_idx);

  plt.set_width(518);
  plt.set_height(350);

  plt.plot(t, v);
  plt.show_grid(true);
  plt.set_margin({top: 5});

  plt.set_tooltip((x, ys) => {
    let y = ys[0];
    let content = `<span>Year: ${x.toFixed(1)} <br> ${variable.meaning}: ${y.toExponential(1)}</span>`;
    let node = html(content);
    MathJax.typeset([node]);
    return node;
  });

  plot_vlines(sim);
  plt.yscale(variable.yscale || 'log');

  let graph = plt.show(container);

  return graph;
}

///////////////////////////////////////////////////////////////////////////////
// Card rendering

let current_card = null;

let graph_icon = `
<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-graph-up" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
</svg>
`;

function render_card(card) {
  let template = `
    <div class="card">

      <div class="left section section-content">
        <h3 class="title"></h3>
        <div class="explanation"></div>
      </div>

      <div class="right">
        <div class="equations-box section">
          <div class="section-label">Key equations</div>
          <div class="equations section-content"></div>
        </div>

        <div class="variables-box section">
          <div class="section-label">Variables and parameters</div>
          <div class="variables section-content"></div>
        </div>
      </div>

    </div>
  `;

  let card_node = html(template);

  card_node.id = card.id;

  // Make the left section be the same height as the right one
  let left = card_node.querySelector('.left');
  let right = card_node.querySelector('.right');
  let resize_observer = new ResizeObserver(function() {
    left.style.height = `${right.getBoundingClientRect().height}px`;
  });
  resize_observer.observe(right);

  // Title
  card_node.querySelector('.title').innerHTML = card.title;

  // Explanations
  let abstract = card.abstract.split('\n').filter(s => s != '').map(s => '<p>' + s + '</p>').join('\n');
  let explanation = card.explanation;
  card_node.querySelector('.explanation').innerHTML = `<div class="abstract">${abstract}</div>` + '\n' + explanation;

  // Equations
  let equations = card.equations;
  if (!Array.isArray(equations)) {
    equations = [equations];
  }
  let equationsHtml = equations.map(s => '<p class="key-equation">\\[' + s + '\\]</p>').join('\n');
  card_node.querySelector('.equations').innerHTML = equationsHtml;

  // Variables
  let variables_table = html(`
    <table class="variable-table">
      <thead>
        <th>Variable</th>
        <th>Meaning</th>
        <th>Evolution in the best guess scenario</th>
      </thead>
      <tbody>
      </tbody>
    </table>
  `);

  for (let var_name of card.variables) {
    let variable = variables[var_name];

    let tr = html(`
      <tr>
        <td>${variable.repr || "\\(" + variable.name + "\\)"}</td>
        <td>${variable.meaning}</td>
        <td class="view-column"></td>
      </tr>
    `);

    if (variable.thread) {
      let view_button = html(`<div class="view-button-hitbox"><div class="view-button"><i>${graph_icon}</i></div></div>`);
      let view_column = tr.querySelector('.view-column');
      view_column.appendChild(view_button);
      let view_button_hitbox = tr.querySelector('.view-button-hitbox');

      let graph;

      tippy(view_button, {
        content: `<div class="plot-container"><div class="plot-title">${variable.meaning}</div><div class="plot-unit">${variable.unit ? "(" + variable.unit + ")" : ""}</div></div>`,
        triggerTarget: view_button_hitbox,
        trigger: 'click',
        duration: [10, 10],
        allowHTML: true,
        interactive: true,
        placement: 'right',
        appendTo: document.body,
        hideOnClick: false,
        theme: 'light-border',
        plugins: [hideOnEsc],
        maxWidth: '860px',
        onCreate: (instance) => {
          let plot_container = instance.popper.querySelector('.plot-container');
          graph = plot_variable(sim, variable, plot_container);
        },
        onShow: (instance) => {
          tippy.hideAll();
        },
        onHidden: (instance) => {
          graph.hide_tooltip();
        },
      });
    }

    let tbody = variables_table.querySelector('tbody');
    tbody.appendChild(tr);
  }

  card_node.querySelector('.variables').appendChild(html('<h4>Variables</h4>'));
  card_node.querySelector('.variables').appendChild(variables_table);

  // Parameters
  let parameters_table = html(`
    <table class="parameter-table">
      <thead>
        <th>Parameter</th>
        <th>Meaning</th>
        <th>Best guess value</th>
        <th>Justification for the best guess value</th>
      </thead>
      <tbody>
      </tbody>
    </table>
  `);

  function get_by_path(root, path) {
    if (typeof path == 'string') {
      // Convert "foo.bar[0].foobar" into "foo.bar.0.foobar"
      path = path.replace(/\[([0-9]*)\]/g, '.$1');

      let fields = path.split('.');
      let v = root;
      for (let field of fields) {
        v = v[field];
      }
      return v;
    } else {
      let v = path(sim);
      return v;
    }
  }

  for (let param_name of card.parameters) {
    let param = parameters[param_name];

    let value = get_by_path(sim.consts, param.constant);

    let tr = html(`
      <tr>
        <td>${param.repr || "\\(" + param.name + "\\)"}</td>
        <td>${param.meaning}</td>
        <td class="value-column"></td>
        <td>${param.justification}</td>
      </tr>
    `);

    let value_column = tr.querySelector('.value-column');

    let unit = param.unit || "";
    if (unit.length > 0 && unit[0] != '/') {
      unit = " " + unit;
    }

    if (Array.isArray(value)) {
      // Show a graph

      let view_button = html(`<div class="view-button-hitbox"><div class="view-button"><i>${graph_icon}</i></div></div>`);
      value_column.appendChild(view_button);
      let view_button_hitbox = tr.querySelector('.view-button-hitbox');

      let graph;

      tippy(view_button, {
        content: `<div class="plot-container"><div class="plot-title">${param.meaning}</div><div class="plot-unit">${param.unit ? "(" + param.unit + ")" : ""}</div></div>`,
        triggerTarget: view_button_hitbox,
        trigger: 'click',
        duration: [10, 10],
        allowHTML: true,
        interactive: true,
        placement: 'right',
        appendTo: document.body,
        hideOnClick: false,
        theme: 'light-border',
        plugins: [hideOnEsc],
        maxWidth: '860px',
        onCreate: (instance) => {
          let plot_container = instance.popper.querySelector('.plot-container');

          graph = plt.graph;
          plt.plot(param.graph.indices(sim), value);
          plt.yscale('log');
          plt.show_grid(true);
          plt.set_margin({top: 5});
          plt.set_width(518);
          plt.set_height(350);

          MathJax.typeset([plot_container]);

          plt.set_tooltip((x, ys) => {
            let y = ys[0];
            let content = `<span>${param.graph.tooltip(x, y)}</span>`;
            let node = html(content);
            MathJax.typeset([node]);
            return node;
          });

          plt.show(plot_container);
        },
        onShow: (instance) => {
          tippy.hideAll();
        },
        onHidden: (instance) => {
          graph.hide_tooltip();
        },
      });
    } else {
      if (typeof value == 'string') {
        value_column.innerHTML = value;
      } else {
        value_column.innerHTML = `<span class="no-break">${standard_format(value)}</span> ${unit}`;
      }
    }

    let tbody = parameters_table.querySelector('tbody');
    tbody.appendChild(tr);
  }

  card_node.querySelector('.variables').appendChild(html('<h4>Parameters</h4>'));
  card_node.querySelector('.variables').appendChild(parameters_table);

  setTimeout(function () {
    MathJax.typeset([card_node])
  }, 0);

  for (let node of card_node.querySelectorAll('[data-tooltip]')) {
    tippy(node, {
      content: node.dataset.tooltip,
      trigger: 'mouseenter click',
      duration: [10, 10],
      allowHTML: true,
      theme: 'light-border',
      plugins: [hideOnEsc],
      onCreate: (instance) => {
        MathJax.typeset([instance.popper]);
      },
    });
  }

  process_internal_links(card_node);

  let initialized = false;
  card.on_open = () => {
    if (!initialized) {
      setTimeout(function () {
        process_learn_more(card_node);
      }, 0);
      initialized = true;
    }
  };

  let box_ids = card.boxes || [`${card.id}-box`];
  let boxes = [];
  for (let box_id of box_ids) {
    let box = document.getElementById(box_id);
    boxes.push(box);
  }

  for (let box of boxes) {
    box.addEventListener('mouseenter', () => {
      for (let b of boxes) b.classList.add('hovering');
    });

    box.addEventListener('mouseleave', () => {
      for (let b of boxes) b.classList.remove('hovering');
    });
  }

  return card_node;
}

// Convert the "Learn more"s into expandable boxes
function process_learn_more(node) {
  for (let learn_more of node.querySelectorAll('.learn-more')) {
    let content_wrapper = html('<div class="learn-more-content"></div>');
    while (learn_more.firstChild) {
      content_wrapper.appendChild(learn_more.firstChild);
    }

    let header = html('<div class="learn-more-header">Learn more</div>');
    learn_more.appendChild(header);
    learn_more.appendChild(content_wrapper);

    content_wrapper.style.maxHeight = 0;
    learn_more.classList.add('closed');

    function update_max_height() {
      if (learn_more.classList.contains('closed')) {
        content_wrapper.style.maxHeight = 0;
      } else {
        let cur_max_height = content_wrapper.style.maxHeight;
        content_wrapper.style.maxHeight = '';
        let height = content_wrapper.getBoundingClientRect().height;
        content_wrapper.style.maxHeight = cur_max_height;
        setTimeout(() => {
          content_wrapper.style.maxHeight = `${height}px`;
        }, 0);
      }
    }

    let prev_width = null;
    let width_observer = new ResizeObserver(function() {
      let cur_width = learn_more.getBoundingClientRect().width;
      if (prev_width != cur_width) {
        prev_width = cur_width;
        update_max_height();
      }
    });
    width_observer.observe(learn_more);

    header.addEventListener('click', () => {
      learn_more.classList.toggle('closed');
      update_max_height();
    });
  }
}

function open_card(card) {
  close_cards();

  let node_container = document.querySelector('#card-container');
  node_container.appendChild(card.node);

  current_card = card;

  for (let box of document.querySelectorAll('svg .box')) {
    box.classList.remove('selected');
  }

  let box_ids = card.boxes || [`${card.id}-box`];
  for (let box_id of box_ids) {
    let box = document.getElementById(box_id);
    box.classList.add('selected');
  }

  window.history.replaceState({}, "", `#${card.id}`);

  card.on_open();
}

function close_cards() {
  let node_container = document.querySelector('#card-container');
  while (node_container.firstChild) node_container.removeChild(node_container.firstChild);

  if (current_card) {
    // Remove card ID from URL
    let url = window.location.toString();
    if (url.indexOf(`#${current_card.id}`) > 0) {
      let clean_url = url.substring(0, url.indexOf("#"));
      window.history.replaceState({}, "", clean_url);
    }
  }

  current_card = null;
  for (let box of document.querySelectorAll('svg .box')) {
    box.classList.remove('selected');
  }
}

// Render the cards
for (let card of cards) {
  let card_node = render_card(card);

  card.node = card_node;

  let box_ids = card.boxes || [`${card.id}-box`];
  for (let box_id of box_ids) {
    let box = document.getElementById(box_id);
    box.addEventListener('click', () => {
      if (card == current_card) {
        close_cards();
      } else {
        open_card(card);
      }
    });
  }
}

///////////////////////////////////////////////////////////////////////////////
// Appendices

let appendix_accordion = new handorgel(document.querySelector('.appendices-container'), {
  multiSelectable: false,
});

appendix_accordion.on('fold:opened', (fold) => {
  window.history.replaceState({}, "", `#${fold.button.id}`);
});

appendix_accordion.on('fold:closed', (fold) => {
  // Remove the appendix ID from the URL
  let url = window.location.toString();
  if (url.indexOf(`#${fold.button.id}`) > 0) {
    let clean_url = url.substring(0, url.indexOf("#"));
    window.history.replaceState({}, "", clean_url);
  }
});

// Close tooltips when clicking outside them
document.body.addEventListener('mousedown', (e) => {
  function is_inside_tooltip_or_view_button(node) {
    if (node == null) return false;
    if ('tippyRoot' in node.dataset) return true;
    if (node.classList.contains('view-button-hitbox')) return true;
    return is_inside_tooltip_or_view_button(node.parentElement);
  }

  if (!is_inside_tooltip_or_view_button(e.target)) {
    tippy.hideAll();
  }
});

document.querySelector('.appendices-container').classList.remove('invisible');

///////////////////////////////////////////////////////////////////////////////
// Deal with internal links to cards and appendices

function follow_internal_link(href) {
  // Is it inside a card?
  for (let card of cards) {
    let node = card.node.matches(href) ? card.node : card.node.querySelector(href);
    if (node) {
      // It is
      open_card(card);

      function is_in_left_section(node) {
        if (node == null) return false;
        if (node.classList.contains('left')) return true;
        return is_in_left_section(node.parentElement);
      }

      if (is_in_left_section(node)) {
        // Scroll inside the card
        let left = card.node.querySelector('.section.left');
        left.scrollTop = node.getBoundingClientRect().top;
      }

      card.node.scrollIntoView({
        behavior: 'smooth',
      });

      return;
    }
  }

  // Is it a fold?
  for (let fold of appendix_accordion.folds) {
    if (fold.button.id == href.slice(1)) {
      open_fold(fold);
      return;
    }
  }
}

function process_internal_links(node) {
  let internal_links = node.querySelectorAll('a[href^="#"]');
  for (let link of internal_links) {
    link.addEventListener("click", (e) => {
      follow_internal_link(link.getAttribute('href'));
      e.preventDefault();
    });
  }
}

function open_fold(fold) {
  function scroll() {
    fold.header.scrollIntoView({
      behavior: 'smooth',
    });
  }

  appendix_accordion.once('fold:opened', (fold) => {
    scroll();
  });

  if (fold.expanded) {
    scroll();
  } else {
    fold.open({transition: false});
  }
}

process_internal_links(document);

if (location.hash) {
  follow_internal_link(location.hash);
}
