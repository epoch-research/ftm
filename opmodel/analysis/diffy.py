import os
import re
import sys
import time
import importlib
from git import Repo
from types import MethodType
from xml.etree import ElementTree as et
from inspect import getframeinfo, stack

from . import log

##################################################################
# Main parameters of future function
project_ref_a = 'b:main'
params_url_a = 'https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=0'

project_ref_b = 'b:bioanchors_comparison'
params_url_b = 'https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=0'
##################################################################


max_steps = 10
eps = 1e-20
repo_url = 'https://github.com/epoch-research/opmodel'
repo_ssh_url = 'ssh://git@github.com/epoch-research/opmodel.git'
cache_dir = '_cache_'
output_dir = '_output_'

module_path = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.abspath(os.path.join(module_path, '..', '..', cache_dir))

output_dir = os.path.abspath(os.path.join(module_path, '..', '..', output_dir))
output_path = 'diffy.html'

class VarInfo:
  def __init__(self):
    self.order = []
    self.var_to_lineno = {}

  def add(self, var, lineno):
    if var in self.order:
      self.order.remove(var)
    self.order.append(var)

    self.var_to_lineno[var] = lineno

class ModelData:
  params_cache = {} # url -> dataframe

  def __init__(
      self,
      name = None,
      project_ref = None, # commit, branch (e.g, 'b:main') or path
      date = None,
      params_url = None,
      module = None,
      relative_module_path = '/core/opmodel.py',
      module_name = 'core.opmodel',
    ):

    self.project_ref = project_ref
    self.source_path = project_ref

    if self.is_commit(project_ref):
      self.commit = self.project_ref
      self.source_href = f'{repo_url}/tree/{self.commit}'
    elif self.is_branch(project_ref):
      self.branch = self.get_branch_from_ref(project_ref)
      self.source_href = f'{repo_url}/tree/{self.branch}'
    else:
      self.source_href = project_ref
      self.timestamp = time.time()

    self.relative_module_path = relative_module_path
    self.date = date
    self.name = name
    self.params_url = params_url
    self.module = module
    self.parameters = None

    self.module = self.get_sim_module()
    self.model = self.module.SimulateTakeOff

  def is_commit(self, ref):
    return re.match('^[0-9a-f]{8,40}$', ref)

  def is_branch(self, ref):
    return ref.startswith('b:')

  def get_branch_from_ref(self, ref):
    return ref[len('b:'):]

  def get_sim_module(self):
    if self.is_branch(self.project_ref):
      project_path = self.get_source_from_ref(self.branch)
    elif self.is_commit(self.project_ref):
      project_path = self.get_source_from_ref(self.commit)
    else:
      project_path = self.project_ref

    module_name = 'core.opmodel'
    file_path = project_path + self.relative_module_path

    sys.path.append(project_path)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    sys.path.remove(project_path)

    return module

  def get_source_from_ref(self, ref):
    repo_copy_path = os.path.join(cache_dir, ref)
    if not os.path.exists(repo_copy_path):
      os.makedirs(cache_dir, exist_ok=True)
      repo = Repo.clone_from(repo_ssh_url, repo_copy_path)
      repo.git.checkout(ref)

    repo = Repo(repo_copy_path)
    if not self.is_commit(ref):
      # it's a branch, then
      repo.git.checkout(ref)
      repo.git.pull(rebase = True)
    self.timestamp = repo.head.commit.committed_date

    return os.path.join(repo_copy_path, 'opmodel')

  def simulate(self):
    inputs = {}
    if self.params_url:
      inputs = {parameter : row['Best guess'] for parameter, row in self.get_parameters().iterrows()}

    # Run model
    self.inputs = inputs
    self.sim = self.model(**inputs)
    self.sim.run_simulation()
    self.n_steps = int(np.ceil((self.sim.t_end - self.sim.t_start)/self.sim.t_step))

    return self.sim

  def get_parameters(self):
    if self.parameters is None:
      if self.params_url not in ModelData.params_cache:
        parameters = pd.read_csv(self.params_url)
        parameters = parameters.set_index("Parameter")
        ModelData.params_cache[self.params_url] = parameters
      self.parameters = ModelData.params_cache[self.params_url].copy()

    return self.parameters

  def get_main_variables(self, step_index):
    sim = self.sim

    main_variables = {}

    if hasattr(sim, "rampup"):
      main_variables["rampup"] = sim.rampup[step_index]

    if hasattr(sim, "hardware_performance"):
      main_variables["hardware performance"] = np.log10(sim.hardware_performance[step_index])

    if hasattr(sim, "frac_gwp_compute"):
      main_variables["frac_gwp_compute"] = sim.frac_gwp_compute[step_index]

    if hasattr(sim, "hardware"):
      main_variables["hardware"] = np.log10(sim.hardware[step_index])

    if hasattr(sim, "software"):
      main_variables["software"] = sim.software[step_index]

    if hasattr(sim, "compute"):
      main_variables["compute"] = np.log10(sim.compute[step_index])

    if hasattr(sim, "labour"):
      main_variables["labour"] = sim.labour[step_index] / sim.labour[0]

    if hasattr(sim, "capital"):
      main_variables["capital"] = sim.capital[step_index] / sim.capital[0] * 0.025

    if hasattr(sim, "automatable_tasks_goods"):
      main_variables["automatable tasks goods"] = sim.automatable_tasks_goods[step_index]

    if hasattr(sim, "frac_automatable_tasks_goods"):
      main_variables["frac automatable tasks goods"] = sim.frac_automatable_tasks_goods[step_index]

    if hasattr(sim, "automatable_tasks_rnd"):
      main_variables["automatable tasks rnd"] = sim.automatable_tasks_rnd[step_index]

    if hasattr(sim, "frac_automatable_tasks_rnd"):
      main_variables["frac automatable tasks rnd"] = sim.frac_automatable_tasks_rnd[step_index]

    if hasattr(sim, "gwp"):
      main_variables["gwp"] = sim.gwp[step_index]

    if hasattr(sim, "frac_capital_hardware_rnd"):
      main_variables["frac_capital_rnd"] = sim.frac_capital_hardware_rnd[step_index]

    if hasattr(sim, "frac_labour_hardware_rnd"):
      main_variables["frac_labour_rnd"] = sim.frac_labour_hardware_rnd[step_index]

    if hasattr(sim, "frac_compute_hardware_rnd"):
      main_variables["frac_compute_rnd"] = sim.frac_compute_hardware_rnd[step_index]

    if hasattr(sim, "rnd_input_hardware"):
      main_variables["rnd input hardware"] = sim.rnd_input_hardware[step_index] / sim.rnd_input_hardware[0] * 0.003048307243707020

    if hasattr(sim, "cumulative_rnd_input_hardware"):
      main_variables["cumulative input hardware"] = sim.cumulative_rnd_input_hardware[step_index] / sim.rnd_input_hardware[0] * 0.003048307243707020

    if hasattr(sim, "rnd_input_hardware"):
      main_variables["ratio rnd input hardware"] = sim.rnd_input_hardware[0]**sim.rnd_parallelization_penalty / sim.cumulative_rnd_input_hardware[0]

    if hasattr(sim, "biggest_training_run"):
      main_variables["biggest_training_run"] = np.log10(sim.biggest_training_run[step_index])

    if hasattr(sim, "compute_share_goods"):
      main_variables["compute share goods"] = sim.compute_share_goods[step_index]

    return main_variables

  def get_all_variables(self, step_index):
    sim = self.sim

    variables = {}

    for attribute in sim.__dict__:
      value = getattr(sim, attribute)
      # Q: Might this be an "iteration variable"?
      if isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.n_steps:
        # A: Maybe
        variables[attribute] = value[step_index]

    return variables

  def get_static_variables(self):
    sim = self.sim

    variables_to_skip = set(['takeoff_metrics'])

    variables = {}

    for attribute in sim.__dict__:
      if attribute in variables_to_skip:
        continue

      value = getattr(sim, attribute)
      # Q: Might this be an "iteration variable"?
      if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != self.n_steps:
        # A: Maybe not
        variables[attribute] = value

    return variables


def get_var_info(module):
  var_info = VarInfo()

  import inspect
  import ast
  import astunparse

  ass_nodes = []

  def fill_attribute_names(nodes, var_info):
    if not isinstance(nodes, (list, tuple)):
      nodes = [nodes]

    for node in nodes:
      if isinstance(node, ast.Attribute):
        var_info.add(node.attr, node.lineno)
      elif isinstance(node, ast.Subscript):
        fill_attribute_names(node.value, var_info)
      elif isinstance(node, ast.Tuple):
        for x in node.elts:
          fill_attribute_names(x, var_info)

  class Instrumentor(ast.NodeVisitor):
    def visit_Assign(self, node):
      ass_nodes.append(node)
      fill_attribute_names(node.targets, var_info)
      self.generic_visit(node)

  source = inspect.getsource(module)
  tree = ast.parse(source)
  instrumentor = Instrumentor()
  instrumentor.visit(tree)

  return var_info

import pandas as pd
import numpy as np

def escape(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace("\"", "&quot;")
    return s

def get_max_change(diff_table):
  max_change = 0
  for var, row in diff_table.iterrows():
    diff = row['Diff (%)']
    if isinstance(diff, str):
      max_change = max(max_change, 100)
    else:
      max_change = max(max_change, np.max(diff))

  return max_change

def to_html(diff_table, as_string = False):
  table = et.Element('table', {'class': 'compare-table'})
  thead = et.Element('thead')
  tbody = et.Element('tbody')
  table.append(thead)
  table.append(tbody)

  no_changes = True

  # thead
  tr = et.Element('tr')
  thead.append(tr)

  tr.append(et.Element('td'))
  for column in diff_table.columns:
    tr.append(et.fromstring(f'<th>{column}</th>'))

  # tbody

  def float_formatter(x):
    formatted = '0' if (x == 0) else f'{x:e}' if ((eps < x and x < 1e-5) or x > 1e5) else str(x)

    # tmp-test
    if np.abs(x - 85267222316546) < 10:
      print(x, type(x))

    return formatted

  precision = 4

  def format(x):
    if isinstance(x, str): return x
    if isinstance(x, (int, float)):
      if x == 0: return '0'
      if x > 1e5 or x < 1e-5: np.format_float_scientific(x, precision = precision)
      if isinstance(x, float): return f'{x:.8}'
    return str(x)

  for var, row in diff_table.iterrows():
    original_precision = np.get_printoptions()['precision']
    try:
      np.set_printoptions(precision = precision)

      tr = et.Element('tr')
      tbody.append(tr)

      some_missing = False
      diff = row['Diff (%)']

      tr.append(et.fromstring(f'<th><div class="inner-td">{var}</div></th>'))

      for col, e in zip(diff_table.columns, row):
        td = et.fromstring(f'<td><div class="inner-td">{e if col == "lineno" else format(e)}</div></td>')

        if isinstance(e, str) and e == '(missing)':
          some_missing = True
          td.set('class', 'missing')

        tr.append(td)

      classes = []

      change_in_row = False

      if some_missing:
        classes.append('missing')
        change_in_row = True
      elif isinstance(diff, str):
        if diff == '(lengths differ)':
          change_in_row = True
      else:
        if np.max(diff) >= 1:
          change_in_row = True
        elif np.max(diff) > 0:
          change_in_row = True
          classes.append('small-difference')

      if change_in_row:
        classes.append('different')
        no_changes = False
      else:
        classes.append('no-changes')

      tr.set('class', ' '.join(classes))
    finally:
      # Restore the precision
      np.set_printoptions(precision = original_precision)

  classes = ['dataframe']
  if no_changes:
    classes.append('no-changes')
    tbody.append(et.fromstring(f'<tr class="no-changes-message" style="display: none"><td colspan="{len(diff_table.columns) + 1}">No changes</td></tr>'))
  table.set('class', ' '.join(classes))

  if as_string:
    et.indent(table, space="  ", level=0)
    return et.tostring(table, method='html').decode()
  else:
    return table

model_a = ModelData(
  name = 'Model a',
  project_ref = project_ref_a,
  params_url = params_url_a,
)

model_b = ModelData(
  name = 'Model b',
  project_ref = project_ref_b,
  params_url = params_url_b,
)

var_info = get_var_info(model_b.module)

model_a.simulate()
model_b.simulate()

html    = et.Element('html')
head    = et.Element('head')
body    = et.Element('body')
content = et.Element('div', {'class': 'main'})

html.append(head)
html.append(body)
body.append(content)

head.append(et.fromstring('<title>Diff checker</title>'))

head.append(et.fromstring('''
  <style>
    :root {
      --no-changes-color: black;
      --changes-color: red;
      --warn-color: hsl(39deg 100% 20%);
    }

    body {
      margin: 0;
      font-family: monospace;
    }

    tbody tr:not(.no-changes):not(.different):not(.no-changes-message) * {
      color: var(--warn-color) !important;
    }

    tbody tr.different * {
      color: var(--changes-color) !important;
    }

    tbody tr.different.small-difference * {
      color: var(--warn-color) !important;
    }

    .filtered-out {
      display: none;
    }

    tr.no-changes {
      color: var(--no-changes-color) !important;
    }

    section.only-show-differences tr.no-changes {
      display: none;
    }

    section.only-show-differences .no-changes-message {
      display: table-row !important;
      text-align: center;
    }

    tbody tr.no-changes-message td {
      text-align: center !important;
      color: green !important;
    }

    table.dataframe {
      border-collapse: collapse;
    }

    table.dataframe thead {
      border-bottom: 1px solid #aaa;
      vertical-align: bottom;
      background-color: #ddd;
    }

    table.dataframe td, table.dataframe th {
      text-align: right;
      padding: 0.2em 1.5em;
      max-height: 10em;
      max-width: 28em;
      border: 1px solid grey;
    }

    table.dataframe td, table.dataframe th {
      text-align: right;
      padding: 0.2em 1.5em;
    }

    table.dataframe tbody tr:nth-child(odd) {
      background-color: #eee;
    }

    table.dataframe tbody tr:hover {
      background-color: #ddd;
    }

    .inner-td {
      overflow-y: auto;
      max-height: 20em;
    }

    h3 {
      margin-top: 3em;
    }

    .main {
      margin: 8px;
    }

    .banner {
      background-color: #ddd;
      padding: 8px;
      text-align: center;
    }

    .ok { color: green; }
    .ko { color: red; }
    .bg-ok { background-color: hsl(120deg 80% 80%); }
    .bg-ko { background-color: hsl(0deg 80% 80%); }
    .bg-warn { background-color: hsl(55deg 80% 80%); }
  </style>
'''))

def get_delta(a, b):
  if isinstance(a, (list, tuple)): a = np.array(a)
  if isinstance(b, (list, tuple)): b = np.array(b)

  if isinstance(a, (float, int)): a = np.array([a])
  if isinstance(b, (float, int)): b = np.array([b])

  if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
    if len(a) != len(b):
      return '(lengths differ)'
    else:
      with np.errstate(divide = 'ignore'): # ignore divide by zero warnings
        diff = 100 * np.max(np.abs(np.divide(b - a, a, where = (a > eps) | (np.abs(b - a) > eps)))) # avoid zero by zero division
        return 0 if (diff < eps) else diff
  elif a == b:
    return 0

  return ''

def compare_dicts(dict_a, dict_b, name_a = 'Model a', name_b = 'Model b', add_line_numbers = True):
  keys = [k for k in dict_a.keys()] + [k for k in dict_b.keys() if k not in dict_a.keys()]

  keys = sorted(keys, key = lambda v: var_info.order.index(v) if v in var_info.order else -1)

  table = []
  for k in keys:
    row = {}
    row[name_a] = dict_a[k] if k in dict_a else '(missing)'
    row[name_b] = dict_b[k] if k in dict_b else '(missing)'
    row['Diff (%)'] = get_delta(row[name_a], row[name_b]) if (k in dict_a and k in dict_b) else ''
    if add_line_numbers:
      row['lineno'] = var_info.var_to_lineno[k] if k in var_info.var_to_lineno else ''
    table.append(row)

  return pd.DataFrame(table, index = keys)

def make_table_filter():
  return et.fromstring(f'''
    <p>
      Variable filter <input class="table-filter"></input>
    </p>
  ''')


#-----------------------
# Model info 
#-----------------------

current_section = et.Element('section', {'class': 'only-show-differences'})
content.append(current_section)

current_section.append(et.fromstring(f'''
  <h3>Models</h3>
'''))

for model in [model_a, model_b]:
  timestamp_span = f'<span data-timestamp="{model.timestamp * 1000}"></span>'

  pre_content = f'''
  {model.name}
    source: <a href="{model.source_href}">{model.source_path}</a> ({timestamp_span})'''

  if model.params_url:
    pre_content += f'\n    parameters: <a href="{escape(model.params_url)}">{escape(model.params_url)}</a>'
  else:
    pre_content += '\n    parameters: internal'

  current_section.append(et.fromstring(f'<pre>{pre_content}</pre>'))

#-----------------------
# Inputs 
#-----------------------

current_section = et.Element('section', {'class': 'only-show-differences'})
content.append(current_section)

current_section.append(et.fromstring(f'''
  <h3>Inputs</h3>
'''))

current_section.append(et.fromstring('<p>Only show differences <input class="only-show-differences" type="checkbox" checked="true"></input></p>'))
current_section.append(make_table_filter())

inputs_table = compare_dicts(model_a.inputs, model_b.inputs, add_line_numbers = False)
current_section.append(to_html(inputs_table))

#-----------------------
# Metrics 
#-----------------------

current_section = et.Element('section', {'class': 'only-show-differences'})
content.append(current_section)

current_section.append(et.fromstring(f'''
  <h3>Takeoff metrics</h3>
'''))

current_section.append(et.fromstring('<p>Only show differences <input class="only-show-differences" type="checkbox" checked="true"></input></p>'))
current_section.append(make_table_filter())

metrics_table = compare_dicts(model_a.sim.takeoff_metrics, model_b.sim.takeoff_metrics, add_line_numbers = False)
current_section.append(to_html(metrics_table))

#---------------------------
# Static values comparison 
#---------------------------

current_section = et.Element('section', {'class': 'only-show-differences'})
content.append(current_section)

current_section.append(et.fromstring(f'''
  <h3>Static variables</h3>
'''))

current_section.append(et.fromstring('<p>Only show differences <input class="only-show-differences" type="checkbox" checked="true"></input></p>'))
current_section.append(make_table_filter())

static_table = compare_dicts(model_a.get_static_variables(), model_b.get_static_variables())
current_section.append(to_html(static_table))

#---------------------------
# Step by step comparison 
#---------------------------

current_section = et.Element('section', {'class': 'only-show-differences'})
content.append(current_section)

current_section.append(et.fromstring(f'''
  <h3>Step by step comparison</h3>
'''))

current_section.append(et.fromstring('<p>Only show differences <input class="only-show-differences" type="checkbox" checked="true"></input></p>'))

step_count = min(model_a.n_steps, model_b.n_steps)

step_selector_container = et.fromstring('<p>Step </p>')
step_selector = et.Element('input', {'type': 'number', 'id': 'step-selector', 'step': '1', 'min': '0', 'max': str(step_count-1), 'value': '0'})
step_selector_container.append(step_selector)
first_change_span = et.Element('span')
step_selector_container.append(first_change_span)
if step_count > max_steps:
  step_selector_container.append(et.fromstring(f'<span>(available steps: first {max_steps} steps of {step_count})</span>'))
current_section.append(step_selector_container)

steps_data = et.Element('div', {'id': 'steps-data'})
steps_data.append(make_table_filter())

main_tables = []
all_tables = []

for step_index in range(step_count):
  main_table = compare_dicts(model_a.get_main_variables(step_index), model_b.get_main_variables(step_index))
  all_table = compare_dicts(model_a.get_all_variables(step_index), model_b.get_all_variables(step_index))

  main_tables.append(main_table)
  all_tables.append(all_table)

  if step_index < max_steps:
    steps_data.append(et.fromstring(f'''
    <div id="step-{step_index}" class="step-data" {'style="display: none"' if step_index != 0 else ''}>
      <div style='margin-top: 1em'><span style='font-weight: bold'>External variables</span>
        {to_html(main_table, as_string = True)}
      </div>

      <div style='margin-top: 1em'><span style='font-weight: bold'>All variables</span>
        {to_html(all_table, as_string = True)}
      </div>
    </div>
    '''))

current_section.append(steps_data)

#---------------------------
# Scripts
#---------------------------

# Handle the "only show differences" option
body.append(et.fromstring('''
  <script>
    for (let checkbox of document.querySelectorAll('input.only-show-differences')) {
      let section = checkbox.closest('section');

      onUpdate = () => {
        if (checkbox.checked) {
          section.classList.add('only-show-differences');
        } else {
          section.classList.remove('only-show-differences');
        }
      };

      checkbox.addEventListener('input', onUpdate);
      onUpdate();
    }
  </script>
'''))

# On step change, show the proper comparison table
body.append(et.fromstring('''
  <script>
    let stepSelector = document.getElementById('step-selector');
    let stepUpdate = () => {
      document.querySelectorAll('.step-data').forEach(x => x.style.display = 'none');

      let selectedStep = document.getElementById('step-' + stepSelector.value);
      if (selectedStep) {
        selectedStep.style.display = 'initial';
      }
    };

    stepSelector.addEventListener('input', stepUpdate);
    stepUpdate();
  </script>
'''))

# Scroll all elements of the same row at the same time
body.append(et.fromstring('''
  <script>
    for (let tr of document.querySelectorAll('tr:not(.no-changes-message)')) {
      let divs = tr.querySelectorAll('td div');
      if (divs.length > 1) {

        function setScroll(trigger) {
          for (let div of divs) {
            if (div != trigger) {
              div.scrollTop = trigger.scrollTop;
            }
          }
        }

        for (let div of divs) {
          div.addEventListener('scroll', e => setScroll(div));
        }
      }
    }
  </script>
'''))

# Filter table variables
body.append(et.fromstring('''
  <script>
    for (let filter of document.querySelectorAll('.table-filter')) {
      let tables = filter.parentElement.parentElement.querySelectorAll('table');

      filter.addEventListener('input', () => {
        for (let table of tables) {
          console.log(table);
          for (let tr of table.querySelectorAll('tbody tr')) {
            let div = tr.querySelector('th div');
            if (!div) continue;

            let varName = div.innerText;
            if (varName.match(filter.value)) {
              tr.classList.remove('filtered-out');
            } else {
              tr.classList.add('filtered-out');
            }
          }
        }
      })
    }
  </script>
'''))

# Convert timestamps to dates
# (add it before any body node)
body.insert(0, et.fromstring('''
  <script>
    function onNodeChange(records) {
      for (let record of records) {
        for (let node of record.addedNodes) {
          if (!node.dataset || !node.dataset.timestamp) continue;

          node.innerText = new Date(+node.dataset.timestamp)
            .toLocaleDateString('en-us', {
              year:"numeric",
              month:"short",
              day:"numeric",
              hour:'2-digit',
              minute:'2-digit',
              hour12:false,
              timeZoneName:'short'
          });
        }
      }
    }

    let observer = new MutationObserver(onNodeChange);
    observer.observe(document.body, { childList: true, subtree: true });
  </script>
'''))

#---------------------------
# Wrap up
#---------------------------

some_change = False

max_change = 0

max_change = max(max_change, get_max_change(metrics_table))
max_change = max(max_change, get_max_change(static_table))

first_change_step = None

for step_index in range(len(main_tables)):
  all_table = all_tables[step_index]
  main_table = main_tables[step_index]

  step_change = max(get_max_change(all_table), get_max_change(main_table))
  max_change = max(max_change, step_change)

  if step_change > 0 and first_change_step is None:
    first_change_step = step_index

if first_change_step is not None:
  step_selector.set('value', str(first_change_step))
  first_change_span.text = f'(first change at step {first_change_step})'

if max_change == 0:
  banner = et.fromstring(f'<div class="banner bg-ok">No changes</div>')
  print('No changes')
elif max_change < 1:
  banner = et.fromstring(f'<div class="banner bg-warn">There are small changes</div>')
  print('There are small changes')
else:
  banner = et.fromstring(f'<div class="banner bg-ko">There are changes</div>')
  print('There are changes')
body.insert(0, banner)

report_path = os.path.abspath(os.path.join(output_dir, output_path))

print(f'Details in the full report: {report_path}')

tree = et.ElementTree(html)
et.indent(tree, space="  ", level=0)

with open(report_path, 'w') as f:
  f.write('<!DOCTYPE html>')
  tree.write(f, encoding='unicode', method='html')
