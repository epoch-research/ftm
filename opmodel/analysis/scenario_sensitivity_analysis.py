"""
Performs both Monte Carlo and sensitivity analyses to generate a joint report
Mimicking this sheet: https://docs.google.com/spreadsheets/d/1RGAUbO2LiLtB-IFlArtjs-IMxZvwbJHXEvORBWgQNDE/
"""

from . import log
from . import *
from ..core.scenarios import ScenarioRunner, get_parameter_table

from .timelines_analysis import timelines_analysis
from .sensitivity_analysis import sensitivity_analysis

def scenario_sensitivity_analysis(report_file_path=None, report_dir_path=None):
  #############################################################################
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # For testing!!
  import os
  import pickle
  module_path = os.path.dirname(os.path.realpath(__file__))
  cache_dir_path = os.path.join(module_path, '..', '..', '_cache_')
  cache_filename = os.path.abspath(os.path.join(cache_dir_path, 'foobar'))

  if not os.path.exists(cache_filename):
    scenarios = ScenarioRunner()
    scenarios.simulate_all_scenarios()
    with open(cache_filename, 'wb') as f:
      pickle.dump(scenarios, f) # , pickle.HIGHEST_PROTOCOL)

  with open(cache_filename, 'rb') as f:
    scenarios = pickle.load(f)

  cache_filename_2 = os.path.abspath(os.path.join(cache_dir_path, 'baq'))

  if not os.path.exists(cache_filename_2):
    sensitivity_results = sensitivity_analysis()
    with open(cache_filename_2, 'wb') as f:
      pickle.dump(sensitivity_results, f) # , pickle.HIGHEST_PROTOCOL)

  with open(cache_filename_2, 'rb') as f:
    sensitivity_results = pickle.load(f)
  # end testing
  #############################################################################

  parameter_table = get_parameter_table()

  if report_file_path is None:
    report_file_path = 'sensitivity_analysis_assumptions_and_results.html'

  report = Report(report_file_path = report_file_path, report_dir_path = report_dir_path)

  html = []

  report.add_html('''
    <style>
      .scenario-analysis-outputs * {
        background-color: white;
      }

      table tr:not(.unhoverable):hover * {
        background-color: #ddd;
      }

      .scenario-analysis-outputs .separator {
        height: 10px;
      }

      .left-align {
        text-align: left !important;
      }

    </style>
  ''', report.head)

  # ---------------------------------------------------------------------------
  # Scenario analysis outputs
  # ---------------------------------------------------------------------------

  report.add_header("Scenario analysis outputs", level = 3)

  table = []
  table.append(f'<table class="dataframe scenario-analysis-outputs">')

  table.append(f'<tr class="unhoverable">')
  table.append(f'<th class="left-align">Metric</th>')
  for metric in scenarios.metrics:
    table.append(f'<th>{metric}</th>')
  table.append(f'</tr>')

  table.append(f'<tr class="unhoverable">')
  table.append(f'<th class="left-align">Meaning of metric</th>')
  for metric in scenarios.metrics:
    table.append(f'<td>{metric}</td>')
  table.append(f'</tr>')

  table.append(f'<tr class="unhoverable">')
  table.append(f'<th class="left-align">Scenarios</th>')
  table.append(f'</tr>')

  for scenario_group in scenarios.groups:
    if scenario_group.name != 'normal':
      table.append(f'<tr class="unhoverable separator"></tr>')

      table.append(f'<tr class="unhoverable">')
      table.append(f'<th class="left-align"><b>{scenario_group.name}</b></th>')
      for metric in scenarios.metrics:
        table.append(f'<td></td>')
      table.append(f'</tr>')

    for scenario in scenario_group:
      table.append(f'<tr>')
      table.append(f'<td class="left-align">{scenario.name}</td>')
      for metric in scenario.metrics:
        table.append(f'<td>{metric.value}</td>')
      table.append(f'</tr>')

  table.append(f'</table>')

  report.add_html_lines(table)

  # ---------------------------------------------------------------------------
  # Scenario analysis assumptions 
  # ---------------------------------------------------------------------------

  report.add_header("Scenario analysis assumptions", level = 3)

  table = []

  table.append(f'<table class="dataframe">')

  table.append(f'<thead>')

  table.append(f'<tr>')
  table.append(f'<th>Parameter</th>')
  table.append(f'<th>Meaning</th>')
  for scenario_group in scenarios.groups:
    if scenario_group.name == 'normal':
      table.append(f'<th><b>Scenarios</b></th>')
    else:
      table.append(f'<td><em>{scenario_group.name}</em></td>')
    for i in range(len(scenario_group) - 1):
      table.append(f'<td></td>')
  table.append(f'</tr>')

  table.append(f'<tr>')
  table.append(f'<td></td>')
  table.append(f'<td></td>')
  for scenario_group in scenarios.groups:
    for scenario in scenario_group:
      table.append(f'<td>{scenario.name}</td>')
  table.append(f'</tr>')

  table.append(f'</thead>')

  table.append(f'<tbody>')
  for parameter, row in parameter_table.iterrows():
    table.append(f'<tr>')
    table.append(f'<th>{parameter}</th>')
    table.append(f'<td style="white-space: normal; text-align: left;">{report.escape(row["Meaning"] if not pd.isnull(row["Meaning"]) else "")}</td>')
    for scenario_group in scenarios.groups:
      for scenario in scenario_group:
        table.append(f'<td>{scenario.params[parameter]}</td>')
    table.append(f'</tr>')
  table.append(f'</tbody>')

  table.append(f'</table>')

  report.add_html_lines(table)

  # ---------------------------------------------------------------------------
  # Param importance outputs
  # ---------------------------------------------------------------------------

  report.add_header("Param importance outputs", level = 3)

  table = []

  table.append(f'<table class="dataframe">')

  table.append(f'<tr>')
  table.append(f'<th>Metric</th>')
  for metric in scenarios.metrics:
    table.append(f'<th>{metric}</th>')
  table.append(f'</tr>')

  table.append(f'<tr>')
  table.append(f'<th>Meaning of metric</th>')
  for metric in scenarios.metrics:
    table.append(f'<td>{metric}</td>')
  table.append(f'</tr>')

  table.append(f'<tr>')
  table.append(f'</tr>')

  table.append(f'<tr>')
  table.append(f'<th>Parameter</th>')
  for metric in scenarios.metrics:
    table.append(f'<td></td>')
  table.append(f'</tr>')

  for parameter in parameter_table.index:
    if parameter not in sensitivity_results.table.index:
      continue

    table.append(f'<tr>')
    table.append(f'<td>{parameter}</td>')
    for metric in scenarios.metrics:
      col = sensitivity_results.table[metric]
      table.append(f'<td>{col[parameter]}</td>')
    table.append(f'</tr>')

  table.append(f'</table>')

  print("\n".join(table))
  report.add_html_lines(table)

  # ---------------------------------------------------------------------------
  # Param importance + Monte Carlo assumptions
  # ---------------------------------------------------------------------------

  report.add_header("Param importance + Monte Carlo assumptions", level = 3)

  table = []

  normal_scenario_group = None
  for group in scenarios.groups:
    if group.name == 'normal':
      normal_scenario_group = group
      break

  table.append(f'<table class="dataframe">')

  table.append(f'<thead>')

  table.append(f'<tr>')
  table.append(f'<th>Parameter</th>')
  table.append(f'<th>Meaning</th>')
  table.append(f'<th><b>Scenarios</b></th>')
  for scenario in range(len(normal_scenario_group) -1):
    table.append(f'<td></td>')
  table.append(f'<th></th>')
  table.append(f'<th></th>')
  table.append(f'</tr>')

  table.append(f'<tr>')
  table.append(f'<td></td>')
  table.append(f'<td></td>')
  for scenario in normal_scenario_group:
    table.append(f'<td>{scenario.name}</td>')
  table.append(f'</tr>')

  table.append(f'</thead>')

  table.append(f'<tbody>')
  for parameter, row in parameter_table.iterrows():
    table.append(f'<tr>')
    table.append(f'<th>{parameter}</th>')
    table.append(f'<td>{report.escape(row["Meaning"] if not pd.isnull(row["Meaning"]) else "")}</td>')
    for scenario in normal_scenario_group:
      table.append(f'<td>{scenario.params[parameter]}</td>')
    table.append(f'</tr>')

  table.append(f'</tbody>')
  table.append(f'</table>')

  report.add_html_lines(table)

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

if __name__ == '__main__':
  parser = init_cli_arguments()
  parser.add_argument(
    "-n",
    "--n-trials",
    type=int,
    default=100,
  )
  args = parser.parse_args()
  scenario_sensitivity_analysis(report_file_path=args.output_file, report_dir_path=args.output_dir)

