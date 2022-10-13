import os
import re
import sys
import ast
import json
import time
import inspect
import traceback
import importlib
import numpy as np
import pandas as pd
from git import Repo
from git.exc import InvalidGitRepositoryError
from xml.etree import ElementTree as et

import argparse

from . import log, Report, get_option, init_cli_arguments, handle_cli_arguments
from ..core.dynamic_array import DynamicArray

EPS = 1e-40 # frankly, just some arbitrary number that feels low enough
DIFF_COL = 'Diff (%)'
CACHE_DIR = os.path.join(get_option('cache_dir'), 'diffy')

# This doesn't seem like a nice structure

class Model:
  def __init__(
      self, 

      model               = None,
      parameters          = None,
      module              = None,
      timestamp           = None,
      name                = None,
      source_href         = None,
      project_ref         = None,
      params_url          = None,
      original_params_url = None,
    ):

    self.model               = model
    self.parameters          = parameters
    self.module              = module
    self.timestamp           = timestamp
    self.name                = name
    self.source_href         = source_href
    self.project_ref         = project_ref
    self.params_url          = params_url
    self.original_params_url = original_params_url
    self.exception = None

  def get_var_to_lineno(self): return {}
  def get_takeoff_metrics(self): return {}
  def get_static_variables(self): return {}
  def get_main_dynamic_variables(self, step_index): return {}
  def get_internal_dynamic_variables(self, step_index): return {}
  def get_step_count(self): return 0

class JSModel(Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def simulate(self):
    inputs = {parameter : row['Best guess'] for parameter, row in self.parameters.iterrows()}
    if not 't_start' in inputs: inputs['t_start'] = 2022
    if not 't_end'   in inputs: inputs['t_end']   = 2100
    if not 't_step'  in inputs: inputs['t_step']  = 0.1
    self.inputs = inputs
    self.module.eval('''
      log = [];
      console = {log: function() {
        log.push(Array.from(arguments).map(x => JSON.stringify(x)).join(', '));
      }};
    ''')
    self.module.eval(f'simulation_result = ftm.run_simulation(bridge.transform_python_to_js_params({json.dumps(self.inputs)}))')
    print(self.module.execute("log.join('\\n')"))

  def get_var_to_lineno(self): return {}

  def get_static_variables(self): return {}

  def get_takeoff_metrics(self):
    variables = self.module.execute(f'bridge.get_takeoff_metrics(simulation_result)')
    return variables

  def get_main_dynamic_variables(self, step_index):
    variables = self.module.execute(f'bridge.get_external_variables(simulation_result, {step_index})')
    return variables

  def get_internal_dynamic_variables(self, step_index):
    variables = self.module.execute(f'bridge.get_internal_variables(simulation_result, {step_index})')
    return variables

  def get_step_count(self):
    count = self.module.execute('simulation_result.states.length')
    return count

class PythonModel(Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def simulate(self):
    log.info('Reading parameters...')

    # Get the parameters and remove those that are not part of this version of the model
    model_parameters = inspect.signature(self.model).parameters

    raw_inputs = self.parameters['Best guess'].to_dict()

    # Handle the parameters that have been renamed
    if not 'initial_buyable_hardware_performance' in raw_inputs:
      raw_inputs['initial_buyable_hardware_performance'] = raw_inputs['initial_hardware_performance']
    if not 'initial_hardware_performance' in raw_inputs:
      raw_inputs['initial_hardware_performance'] = raw_inputs['initial_buyable_hardware_performance']

    # Delete the extraneous parameters
    inputs = {k : v for k, v in raw_inputs.items() if k in model_parameters}

    self.inputs = inputs

    self.inputs['t_start'] = get_option('t_start', 2022)
    self.inputs['t_end']   = get_option('t_end',   2100)
    self.inputs['t_step']  = get_option('t_step',   0.1)

    try:
      log.info('Running the model...')
      self.sim = self.model(**inputs)
      self.sim.run_simulation()
      self.n_steps = int(np.ceil((self.sim.t_end - self.sim.t_start)/self.sim.t_step))
    except Exception as e:
      log.error('The model threw an exception:')
      log.indent()
      for line in traceback.format_exception(e):
        log.error(line.strip())
      log.deindent()
      self.exception = e

  def get_step_count(self):
    return self.n_steps

  def get_takeoff_metrics(self):
    return self.sim.takeoff_metrics

  def get_static_variables(self):
    variables = {}

    variables_to_skip = set(['takeoff_metrics', 'dynarrays'])

    for attribute in self.sim.__dict__:
      if attribute in variables_to_skip: continue

      value = getattr(self.sim, attribute)
      if isinstance(value, DynamicArray):
        value = value.data
      if not self.is_dynamic_variable(value):
        variables[attribute] = value

    return variables

  def get_main_dynamic_variables(self, step_index):
    sim = self.sim

    # Wrapping the variables in lambdas to prevent exceptions if they are missing in this version of the model
    variables = {
      "rampup":                       lambda: sim.rampup[step_index],
      "hardware performance":         lambda: np.log10(sim.hardware_performance[step_index]),
      "frac_gwp_compute":             lambda: sim.frac_gwp_compute[step_index],
      "hardware":                     lambda: np.log10(sim.hardware[step_index]),
      "software":                     lambda: sim.software[step_index],
      "compute":                      lambda: np.log10(sim.compute[step_index]),
      "labour":                       lambda: sim.labour[step_index] / sim.labour[0],
      "capital":                      lambda: sim.capital[step_index] / sim.capital[0] * 0.025,
      "automatable tasks goods":      lambda: sim.automatable_tasks_goods[step_index],
      "frac automatable tasks goods": lambda: sim.frac_automatable_tasks_goods[step_index],
      "automatable tasks rnd":        lambda: sim.automatable_tasks_rnd[step_index],
      "frac automatable tasks rnd":   lambda: sim.frac_automatable_tasks_rnd[step_index],
      "gwp":                          lambda: sim.gwp[step_index],
      "frac_capital_rnd":             lambda: sim.frac_capital_hardware_rnd[step_index],
      "frac_labour_rnd":              lambda: sim.frac_labour_hardware_rnd[step_index],
      "frac_compute_rnd":             lambda: sim.frac_compute_hardware_rnd[step_index],
      "rnd input hardware":           lambda: sim.rnd_input_hardware[step_index] / sim.rnd_input_hardware[0] * 0.003048307243707020,
      "cumulative input hardware":    lambda: sim.cumulative_rnd_input_hardware[step_index] / sim.rnd_input_hardware[0] * 0.003048307243707020,
      "ratio rnd input hardware":     lambda: sim.rnd_input_hardware[0]**sim.rnd_parallelization_penalty / sim.cumulative_rnd_input_hardware[0],
      "biggest_training_run":         lambda: np.log10(sim.biggest_training_run[step_index]),
      "compute share goods":          lambda: sim.compute_share_goods[step_index],
    }

    main_variables = {}

    for label, get_value in variables.items():
      try:
        main_variables[label] = get_value()
      except Exception as e:
        pass

    return main_variables

  def get_internal_dynamic_variables(self, step_index):
    variables = {}

    for attribute in self.sim.__dict__:
      value = getattr(self.sim, attribute)
      if isinstance(value, DynamicArray):
        value = value.data
      if self.is_dynamic_variable(value):
        variables[attribute] = value[step_index]

    return variables

  def is_dynamic_variable(self, value):
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.n_steps

  def get_var_to_lineno(self):
    var_to_lineno = {}

    ass_nodes = []

    class Instrumentor(ast.NodeVisitor):
      def visit_Assign(self, node):
        ass_nodes.append(node)
        self.traverse_nodes(node.targets)
        self.generic_visit(node)

      def traverse_nodes(self, nodes):
        if not isinstance(nodes, (list, tuple)):
          nodes = [nodes]

        for node in nodes:
          if isinstance(node, ast.Attribute):
            var_to_lineno[node.attr] = node.lineno
          elif isinstance(node, ast.Subscript):
            self.traverse_nodes(node.value)
          elif isinstance(node, ast.Tuple):
            for x in node.elts:
              self.traverse_nodes(x)

    source = inspect.getsource(self.module)
    tree = ast.parse(source)
    instrumentor = Instrumentor()
    instrumentor.visit(tree)

    return var_to_lineno

class ModelManager:
  params_cache = {} # memory cache for the model parameters (url --> dataframe)
  model_to_index = {}

  @staticmethod
  def load_model(
      name                        = None,
      project_ref                 = None, # commit, branch (e.g, 'b:main') or path
      params_url                  = None,
      relative_python_module_path = None
    ):

    original_params_url = params_url

    if not relative_python_module_path:
      relative_python_module_path = 'opmodel/core/opmodel.py'

    # Clean the param url if it's a Google sheet
    pattern = r'https://docs.google.com/spreadsheets/d/([a-zA-Z0-9-_]*)/.*\bgid\b=([0-9]*)?.*'
    m = re.match(pattern, params_url)
    if m:
      workbook_id = m.group(1)
      sheet_id = m.group(2)
      params_url = f'https://docs.google.com/spreadsheets/d/{workbook_id}/export?format=csv&gid={sheet_id}'

    parameters = ModelManager.get_parameters(params_url)

    # Keep track of the active models 
    model_index = 0
    while model_index in ModelManager.model_to_index.values():
      model_index += 1

    ref_type = ModelManager.get_ref_type(project_ref)

    if ref_type == 'git':
      git_ref = ModelManager.get_git_ref(project_ref)
      project_path, timestamp = ModelManager.load_source_from_git_ref(model_index, git_ref)
      source_href = f'{get_option("repo_web_url")}/tree/{git_ref}'
    else:
      project_path = project_ref
      timestamp = time.time()
      source_href = project_ref

    # Load the model
    if os.path.exists(os.path.join(project_path, relative_python_module_path)):
      # This is a Python model
      module = ModelManager.load_python_module(project_path, relative_python_module_path)
      model = PythonModel(
        model               = module.SimulateTakeOff,
        parameters          = parameters,
        module              = module,
        timestamp           = timestamp,
        name                = name,
        source_href         = source_href,
        project_ref         = project_ref,
        params_url          = params_url,
        original_params_url = original_params_url,
      )
    else:
      # This must be a JavaScript model
      module = ModelManager.load_js_module(project_path)
      model = JSModel(
        parameters          = parameters,
        module              = module,
        timestamp           = timestamp,
        name                = name,
        source_href         = source_href,
        project_ref         = project_ref,
        params_url          = params_url,
        original_params_url = original_params_url,
      )

    ModelManager.model_to_index[model] = model_index

    return model

  @staticmethod
  def unload_model(model):
    del ModelManager.model_to_index[model]

  @staticmethod
  def get_ref_type(ref):
    if ref.startswith('g:'):
      return 'git'
    return 'local-path'

  @staticmethod
  def get_git_ref(ref):
    return ref[len('g:'):]

  @staticmethod
  def load_source_from_git_ref(model_index, ref):
    repo = ModelManager.get_local_repo(model_index)

    # Hack
    # We are moving to main to avoid errors if the branch we were in disappears.
    repo.git.checkout('main')

    repo.git.reset('--hard')
    repo.git.pull()
    repo.git.checkout(ref)

    timestamp = repo.head.commit.committed_date
    return [repo.working_tree_dir, timestamp]

  @staticmethod
  def get_local_repo(model_index):
    path = os.path.join(CACHE_DIR, 'repos', f'repo_{model_index}')
    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)
      Repo.clone_from(get_option('repo_url'), path)

    try:
      repo = Repo(path)
    except InvalidGitRepositoryError as e:
      # OK, maybe the repo is corrupted. Let's try again .
      os.rmdir(path)
      os.makedirs(path, exist_ok=True)
      Repo.clone_from(get_option('repo_url'), path)
      repo = Repo(path)

    return repo

  @staticmethod
  def load_python_module(project_path, relative_module_path):
    module_name = '.'.join(relative_module_path[:-len('.py')].split('/'))

    file_path = os.path.join(project_path, relative_module_path)

    sys.path.append(project_path)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    sys.path.remove(project_path)

    return module

  @staticmethod
  def load_js_module(project_path):
    from py_mini_racer import MiniRacer

    scripts = [
      'nj.js',
      'ftm.js',
      'bridge.js',
    ]

    ctx = MiniRacer()
    for script in scripts:
      with open(os.path.join(project_path, script), 'r') as f:
        script = f.read()
        ctx.eval(script)
    return ctx

  @staticmethod
  def get_parameters(params_url):
    if params_url not in ModelManager.params_cache:
      parameters = pd.read_csv(params_url)
      parameters = parameters.set_index("Parameter id")
      ModelManager.params_cache[params_url] = parameters
    parameters = ModelManager.params_cache[params_url].copy()

    return parameters

def diffy(
    project_ref_a = 'g:main',
    params_url_a = None,
    relative_module_path_a = None,

    project_ref_b = '.',
    params_url_b = None,
    relative_module_path_b = None,

    max_steps = 10,

    ignore_missings = False,

    report_file_path = None,
    report_dir_path = None,
  ):

  #--------------------------------------------------------------------------
  # Initialization
  #--------------------------------------------------------------------------

  if params_url_a is None:
    params_url_a = get_option('param_table_url', 'https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=0')

  if params_url_b is None:
    params_url_b = get_option('param_table_url', 'https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=0')

  if report_file_path is None:
    report_file_path = 'diffy.html'

  model_a = ModelManager.load_model(
    name = 'Model a',
    project_ref = project_ref_a,
    params_url = params_url_a,
    relative_python_module_path = relative_module_path_a,
  )

  model_b = ModelManager.load_model(
    name = 'Model b',
    project_ref = project_ref_b,
    params_url = params_url_b,
    relative_python_module_path = relative_module_path_b,
  )

  for model in [model_a, model_b]:
    log.info(f'Simulating {model.name}')

    log.indent()
    model.simulate()
    log.deindent()

    log.info()

  var_to_lineno = model_b.get_var_to_lineno()
  if not var_to_lineno:
    var_to_lineno = model_a.get_var_to_lineno()

  report = Report(report_file_path = report_file_path, report_dir_path = report_dir_path)

  content = report.content

  report.add_title('Diff checker')

  report.head.append(et.fromstring('''
    <style>
      :root {
        --no-changes-color: black;
        --changes-color: red;
        --warn-color: hsl(39deg 100% 20%);
      }

      body {
        margin: 0;
        font-family: monospace;
        font-size: unset;
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

      tbody tr.different .tippy-content {
        color: white !important;
      }

      tbody tr.different .tippy-arrow {
        color: #333 !important;
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
        white-space: initial;
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
      }

      .inner-td {
        overflow-y: auto;
        max-height: 20em;
      }

      .ok { color: green; }
      .ko { color: red; }
      .bg-ok { background-color: hsl(120deg 80% 80%); }
      .bg-ko { background-color: hsl(0deg 80% 80%); }
      .bg-warn { background-color: hsl(55deg 80% 80%); }
    </style>
  '''))

  #--------------------------------------------------------------------------
  # Model info
  #--------------------------------------------------------------------------
  current_section = add_section(content, 'Models', add_filters = False)

  for model in [model_a, model_b]:
    timestamp_span = f'<span data-timestamp="{model.timestamp * 1000}"></span>'

    pre_lines = []
    pre_lines.append(f'{model.name}')
    pre_lines.append(f'    source: <a href="{model.source_href}">{model.project_ref}</a> ({timestamp_span})')

    if model.params_url:
      pre_lines.append(f'    parameters: <a href="{Report.escape(model.original_params_url)}">{Report.escape(model.original_params_url)}</a>')
    else:
      pre_lines.append('    parameters: internal')

    pre_content = '\n'.join(pre_lines)
    current_section.append(et.fromstring(f'<pre>{pre_content}</pre>'))

  #--------------------------------------------------------------------------
  # Inputs
  #--------------------------------------------------------------------------
  current_section = add_section(content, 'Parameters')

  inputs_table = compare_dicts(model_a.inputs, model_b.inputs, var_to_lineno, add_line_numbers = False, ignore_missings = ignore_missings)
  current_section.append(diff_table_to_html(inputs_table))

  if model_a.exception or model_b.exception:
    #--------------------------------------------------------------------------
    # Errors!
    #--------------------------------------------------------------------------
    current_section = add_section(content, 'Exceptions', add_filters = False)

    for model in [model_a, model_b]:
      if model.exception:
        current_section.append(et.fromstring(f'<p>An exception happened while executing {model.name}</p>'))
        current_section.append(et.fromstring(f'<pre>    {Report.escape("    ".join(traceback.format_exception(model.exception)))}</pre>'))

    log.info('> Exception executing the models')
    report.add_banner_message('Exception executing the models', ['bg-ko'])

  else: # everything's fine

    #--------------------------------------------------------------------------
    # Metrics
    #--------------------------------------------------------------------------
    current_section = add_section(content, 'Takeoff metrics')

    metrics_table = compare_dicts(model_a.get_takeoff_metrics(), model_b.get_takeoff_metrics(), var_to_lineno, add_line_numbers = False, ignore_missings = ignore_missings)
    current_section.append(diff_table_to_html(metrics_table,))

    #--------------------------------------------------------------------------
    # Static values comparison
    #--------------------------------------------------------------------------
    current_section = add_section(content, 'Static variables')

    static_table = compare_dicts(model_a.get_static_variables(), model_b.get_static_variables(), var_to_lineno, ignore_missings = ignore_missings)
    current_section.append(diff_table_to_html(static_table))

    #--------------------------------------------------------------------------
    # Step by step comparison
    #--------------------------------------------------------------------------
    current_section = add_section(content, 'Step by step comparison')

    # Step number selector
    step_count = min(model_a.get_step_count(), model_b.get_step_count())

    step_selector_container = et.fromstring('<p>Step </p>')
    current_section.append(step_selector_container)

    step_selector = et.Element('input', {'type': 'number', 'id': 'step-selector', 'step': '1', 'min': '0', 'max': str(step_count-1), 'value': '0'})
    step_selector_container.append(step_selector)

    first_change_span = et.Element('span')
    step_selector_container.append(first_change_span)

    if step_count > max_steps:
      step_selector_container.append(et.fromstring(f'<span>(available steps: first {max_steps} steps of {step_count})</span>'))

    # Add the tables
    external_tables = []
    internal_tables = []

    steps_data = et.Element('div', {'id': 'steps-data'})
    current_section.append(steps_data)

    for step_index in range(step_count):
      external_table = compare_dicts(model_a.get_main_dynamic_variables(step_index), model_b.get_main_dynamic_variables(step_index), var_to_lineno, ignore_missings = ignore_missings)
      internal_table = compare_dicts(model_a.get_internal_dynamic_variables(step_index), model_b.get_internal_dynamic_variables(step_index), var_to_lineno, ignore_missings = ignore_missings)

      external_tables.append(external_table)
      internal_tables.append(internal_table)

      if step_index < max_steps:
        steps_data.append(et.fromstring(f'''
        <div id="step-{step_index}" class="step-data" {'style="display: none"' if step_index != 0 else ''}>
          <div style='margin-top: 1em'><span style='font-weight: bold'>External variables</span>
            {diff_table_to_html(external_table, return_as_string = True)}
          </div>

          <div style='margin-top: 1em'><span style='font-weight: bold'>Internal variables</span>
            {diff_table_to_html(internal_table, return_as_string = True)}
          </div>
        </div>
        '''))

    #--------------------------------------------------------------------------
    # Wrap up
    #--------------------------------------------------------------------------
    max_change = 0
    max_change = max(max_change, get_max_change(metrics_table))
    max_change = max(max_change, get_max_change(static_table))

    first_change_step = None
    max_change_step = None
    max_step_change = None

    for step_index in range(len(external_tables)):
      external_table = external_tables[step_index]
      internal_table = internal_tables[step_index]

      step_change = max(get_max_change(internal_table), get_max_change(external_table))
      max_change = max(max_change, step_change)

      if step_change > 0 and (max_change_step is None or step_change > max_step_change):
        max_step_change = step_change
        max_change_step = step_index

      if step_change > 0 and first_change_step is None:
        first_change_step = step_index

    first_change_span.text = ''

    if first_change_step is not None:
      step_selector.set('value', str(first_change_step)) # show the first interesting step by default
      first_change_span.text += f'(first change at step {first_change_step})'

    if max_change_step is not None:
      first_change_span.text += f' (max change at step {max_change_step} ({max_step_change:e} %))'

    if max_change == 0:
      report.add_banner_message('No changes', ['bg-ok'])
      log.info('> No changes in the models')
    elif max_change < 1:
      report.add_banner_message('There are small changes', ['bg-warn'])
      log.info('> There are small changes in the models')
    else:
      report.add_banner_message('There are changes', ['bg-ko'])
      log.info('> There are changes in the models')

  add_javascript_to_report(report)

  report_path = report.write()

  log.info(f'Details in the full report: {report_path}')

  ModelManager.unload_model(model_a)
  ModelManager.unload_model(model_b)

def get_max_change(diff_table):
  max_change = max([100 if isinstance(x, str) else abs(x) for x in diff_table[DIFF_COL]], default = 0)
  return max_change

def diff_table_to_html(diff_table, return_as_string = False):
  table = et.Element('table', {'class': 'compare-table'})
  thead = et.Element('thead')
  tbody = et.Element('tbody')
  table.append(thead)
  table.append(tbody)

  no_changes = True

  #
  # thead
  #

  tr = et.Element('tr')
  thead.append(tr)

  tr.append(et.Element('td'))
  for column in diff_table.columns:
    tr.append(et.fromstring(f'<th>{column}</th>'))

  #
  # tbody
  #

  precision = 4

  def float_formatter(x):
    formatted = '0' if (x == 0) else f'{x:e}' if (x < 1e-5 or x > 1e5) else str(x)
    return formatted

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

      some_missing = False

      tr = et.Element('tr')
      tr.append(et.fromstring(f'<th>{var}</th>'))

      for col, e in zip(diff_table.columns, row):
        td = et.fromstring(f'<td>{e if col == "lineno" else format(e)}</td>')
        if isinstance(e, str) and e == '(missing)':
          some_missing = True
          td.set('class', 'missing')
        tr.append(td)

      tbody.append(tr)

      classes = []
      diff = row[DIFF_COL]
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
    tbody.append(et.fromstring(
      f'<tr class="no-changes-message" style="display: none"><td colspan="{len(diff_table.columns) + 1}">No changes</td></tr>'
    ))

  table.set('class', ' '.join(classes))

  if return_as_string:
    et.indent(table, space="  ", level=0)
    return et.tostring(table, method='html').decode()
  else:
    return table

def compare_dicts(dict_a, dict_b, var_to_lineno, name_a = 'Model a', name_b = 'Model b', add_line_numbers = True, ignore_missings = False):
  def get_diff(a, b):
    """
    Returns the maximum relative difference (or a string)
    """

    if isinstance(a, bool): a = 1 if a else 0
    if isinstance(b, bool): b = 1 if b else 0
    if isinstance(a, (list, tuple)): a = np.array(a)
    if isinstance(b, (list, tuple)): b = np.array(b)
    if isinstance(a, (float, int)): a = np.array([a])
    if isinstance(b, (float, int)): b = np.array([b])

    diff = ''

    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
      if len(a) != len(b):
        diff = '(lengths differ)'
      else:
        with np.errstate(divide = 'ignore'): # ignore division by zero warnings
          diff = 100 * np.max(np.abs(np.divide(b - a, a, where = (a > EPS) | (np.abs(b - a) > EPS)))) # avoid zero by zero division
          diff = 0 if (diff < EPS) else diff
    elif a == b:
      diff = 0

    return diff

  table = []

  # Variables sorted in the order in which they are set in the model b module
  keys = [k for k in dict_a.keys()] + [k for k in dict_b.keys() if k not in dict_a.keys()]
  keys = sorted(keys, key = lambda v: var_to_lineno[v] if v in var_to_lineno else -1)

  filtered_keys = []
  for k in keys:
    if ignore_missings and k not in dict_a or k not in dict_b:
      continue
    row = {}
    row[name_a] = dict_a[k] if k in dict_a else '(missing)'
    row[name_b] = dict_b[k] if k in dict_b else '(missing)'
    row[DIFF_COL] = get_diff(row[name_a], row[name_b]) if (k in dict_a and k in dict_b) else ''
    if add_line_numbers:
      row['lineno'] = var_to_lineno[k] if k in var_to_lineno else ''
    table.append(row)
    filtered_keys.append(k)

  columns = [name_a, name_b, DIFF_COL]
  if add_line_numbers:
    columns.append('lineno')
  return pd.DataFrame(table, index = filtered_keys, columns = columns)

def add_section(parent_element, title, add_filters = True):
  section = et.Element('section', {'class': 'only-show-differences'})
  parent_element.append(section)

  section.append(et.fromstring(f'<h3>{title}</h3>'))

  if add_filters:
    section.append(et.fromstring('<p>Only show differences <input class="only-show-differences" type="checkbox" checked="true"></input></p>'))
    section.append(et.fromstring(f'<p>Variable filter <input class="table-filter"></input></p>'))

  return section

def add_javascript_to_report(report):
  body = report.body

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

  # When changing the step number, show the comparison table for that step
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
            for (let tr of table.querySelectorAll('tbody tr')) {
              let div = tr.querySelector('th');
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

      {
        let observer = new MutationObserver(onNodeChange);
        observer.observe(document.body, { childList: true, subtree: true });
      }
    </script>
  '''))


if __name__ == '__main__':
  parser = init_cli_arguments()

  parser.add_argument(
    "-a",
    "--ref-a",
    default=None,
    help="Reference to model a",
  )

  parser.add_argument(
    "-b",
    "--ref-b",
    default=None,
    help="Reference to model a",
  )

  parser.add_argument(
    "--params-a",
    default=None,
    help="Parameters URL or local path for model a",
  )

  parser.add_argument(
    "--params-b",
    default=None,
    help="Parameters URL or local path for model b",
  )

  parser.add_argument(
    "--module-path-a",
    default=None,
    help="Relative Python module path for model a",
  )

  parser.add_argument(
    "--module-path-b",
    default=None,
    help="Relative Python module path for model b",
  )

  parser.add_argument(
    "--params",
    default=None,
    help="Parameters URL or local path for both models",
  )

  parser.add_argument(
    "-s",
    "--max-steps",
    type=int,
    default=10,
    help="Outputs to the report at most this number of steps",
  )

  parser.add_argument(
    "--ignore-missings",
    action='store_true',
  )

  args = handle_cli_arguments(parser)

  diff_args = {}

  if args.ref_a: diff_args['project_ref_a'] = args.ref_a
  if args.ref_b: diff_args['project_ref_b'] = args.ref_b

  if args.params:
    diff_args['params_url_a'] = args.params
    diff_args['params_url_b'] = args.params
  if args.params_a: diff_args['params_url_a'] = args.params_a
  if args.params_b: diff_args['params_url_b'] = args.params_b

  diff_args['max_steps'] = args.max_steps
  diff_args['ignore_missings'] = args.ignore_missings

  if args.module_path_a: diff_args['relative_module_path_a'] = args.module_path_a
  if args.module_path_b: diff_args['relative_module_path_b'] = args.module_path_b

  diffy(**diff_args)
