import json
from xml.etree import ElementTree as et

from . import log
from . import *
from .exploration_analysis import add_scenario_group_to_report, plot_compute_increase
from ..core.scenarios import ScenarioRunner, get_parameter_table

class TimelinesAnalysisResults:
  pass

def write_timelines_analysis_report(report_file_path=None, report_dir_path=None, report=None):
  if report_file_path is None:
    report_file_path = 'timelines_analysis.html'

  results = timelines_analysis()

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  #
  # Metrics
  #

  report.add_header("Metrics", level = 3)

  table = []

  for group in results.scenario_groups:
    for scenario in group:
      row = {**{'timeline' : group.name, 'scenario' : scenario.name}, **scenario.model.takeoff_metrics}
      for metric in ['rampup_start', 'agi_year', 'doubling_times']:
        row[metric] = getattr(scenario.model, metric)
      table.append(row)
  table = pd.DataFrame(table)

  report.add_data_frame(table, show_index = False)


  #
  # Graphs
  #

  report.add_header("Compute increase over time", level = 3)

  # Hackily add legend as an independent separate figure
  plot_compute_increase(results.scenario_groups[0], title = group.name, show_legend = False)
  ax = plt.gca()
  legend_fig = plt.figure()
  plt.figlegend(*ax.get_legend_handles_labels())
  report.add_figure(legend_fig)

  graph_container = report.add_html('<div style="display: flex; overflow-x: auto;"></div>')

  for i, group in enumerate(results.scenario_groups):
    plot_compute_increase(group, title = group.name, show_legend = False)
    report.add_figure(parent = graph_container)

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  #
  # Model summaries and inputs
  #

  # Summary

  report.add_header("Model summary", level = 3)

  scenario_options = "\n".join([
    f'<option value="{group.name} - {scenario.name}">{group.name} - {scenario.name}</option>'
    for group in results.scenario_groups
    for scenario in group
  ])

  report.add_html(f'''
    <p>
      Scenario <br></br>
      <select id="scenario-1-selector">
        {scenario_options}
      </select>
    </p>
  ''')

  report.add_html(f'''
    <p>
      Compare to <br></br>
      <select id="scenario-2-selector">
        <option value="null">(none)</option>
        {scenario_options}
      </select>
    </p>
  ''')

  table_container = et.Element('div', {'class': 'dataframe-container'})
  table = et.Element('table', {'border': '1', 'class': 'dataframe', 'id': 'timelines-summary-table'})
  tbody = et.Element('tbody')
  thead = et.Element('thead')
  tr = et.Element('tr', {'style': 'text-align: right;'})

  scenario = results.scenario_groups[0][0]
  tr.append(et.fromstring(f'<th>timeline</th>'))
  tr.append(et.fromstring(f'<th>scenario</th>'))
  for metric in scenario.model.get_summary_table().columns:
    tr.append(et.fromstring(f'<th>{metric}</th>'))

  thead.append(tr)
  table.append(thead)
  table.append(tbody)
  table_container.append(table)

  container = report.add_html('<div style="overflow-x: auto;"></div>')
  container.append(table_container)

  summaries = {}
  for group in results.scenario_groups:
    summaries_group = {}
    for scenario in group:
      summary_table = scenario.model.get_summary_table()
      summaries_group[scenario.name] = summary_table.to_dict()
      row_count = len(summary_table)
    summaries[group.name] = summaries_group
  summaries_json = json.dumps(summaries)

  # Inputs

  report.add_header("Inputs", level = 3)

  report.add_html(f'''
    <p>
      Scenario <br></br>
      <select id="scenario-1-selector-inputs">
        {scenario_options}
      </select>
    </p>
  ''')

  report.add_html(f'''
    <p>
      Compare to <br></br>
      <select id="scenario-2-selector-inputs">
        <option value="null">(none)</option>
        {scenario_options}
      </select>
    </p>
  ''')

  table_container = et.Element('div', {'class': 'dataframe-container'})
  table = et.Element('table', {'border': '1', 'class': 'dataframe', 'id': 'timelines-inputs-table'})
  tbody = et.Element('tbody')
  thead = et.Element('thead')
  table.append(thead)
  table.append(tbody)
  table_container.append(table)

  container = report.add_html('<div style="overflow-x: auto;"></div>')
  container.append(table_container)

  inputs = {}
  for group in results.scenario_groups:
    group_inputs = {}
    for scenario in group:
      group_inputs[scenario.name] = scenario.params
    inputs[group.name] = group_inputs
  inputs_json = json.dumps(inputs)


  #
  # Add JavaScript
  #

  script = et.Element('script')
  script.text = '''
    let summaries = ''' + summaries_json + ''';
    let inputs = ''' + inputs_json + ''';
    let rowCount = ''' + str(row_count) + ''';

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

      for (let k in col) {
        x = col[k];
        if (typeof x != 'string' && !Number.isNaN(x)) {
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

      let formatted = {};
      for (let k in col) {
        x = col[k];
        if (typeof x == 'string' || Number.isNaN(x)) {
          formatted[k] = x;
        } else {
          if (someExponential) {
            let f = col[k].toExponential(maxExpFracDigits);
            let [coefficient, exponent] = f.split('e');
            let sign = exponent.charAt(0);
            formatted[k] = `${coefficient}e${sign}${exponent.substring(1).padStart(maxExpDigits, '0')}`;
          } else {
            formatted[k] = col[k].toFixed(maxFracDigits);
          }
        }
      }

      return formatted;
    }
    function renderSummary(scenarios) {
      let summaryTable = document.getElementById('timelines-summary-table');
      let tbody = summaryTable.querySelector('tbody')

      let html = '';

      for (let scenarioIndex = 0; scenarioIndex < scenarios.length; scenarioIndex++) {
        if (scenarioIndex > 0) {
          // Vertical spacing
          html += '<tr style="height: 15px"></tr>';

          // Keep even rows even and odd rows odd
          html += '<tr style="display: none"></tr>';
        }

        let scenarioId = scenarios[scenarioIndex];

        let timeline = scenarioId[0];
        let name = scenarioId[1];
        let summary = summaries[timeline][name];

        let formattedCols = {};
        for (let col in summary) {
          formattedCols[col] = formatColumn(summary[col]);
        }

        for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
          html += '<tr>';
          html += `<td>${timeline}</td>`;
          html += `<td>${name}</td>`;
          for (let col in summary) {
            html += `<td>${formattedCols[col][rowIndex]}</td>`;
          }
          html += '</tr>';
        }
      }

      tbody.innerHTML = html;

      reloadMeaningTooltips();
      reloadParamBgColors();
    }

    function renderInputs(scenarios) {
      let inputsTable = document.getElementById('timelines-inputs-table');
      let tbody = inputsTable.querySelector('tbody')
      let thead = inputsTable.querySelector('thead')

      let theadHtml = '';
      let html = '';

      theadHtml += '<tr><th></th>';

      let formattedCols = [];
      for (let scenarioIndex = 0; scenarioIndex < scenarios.length; scenarioIndex++) {
        let scenarioId = scenarios[scenarioIndex];

        let timeline = scenarioId[0];
        let name = scenarioId[1];
        let input = inputs[timeline][name];

        let formattedCol = formatColumn(input);
        formattedCols.push(formattedCol);

        theadHtml += `<th>${timeline} - ${name}</th>`;
      }

      for (let param in formattedCols[0]) {
        html += '<tr>';
        html += `<th>${param}</th>`;
        for (let col of formattedCols) {
          html += `<td>${col[param]}</td>`;
        }
        html += '</tr>';
      }

      theadHtml += '</tr>';
      tbody.innerHTML = html;
      thead.innerHTML = theadHtml;

      reloadMeaningTooltips();
      reloadParamBgColors();
    }

    let scenario1Selector = document.getElementById('scenario-1-selector');
    let scenario2Selector = document.getElementById('scenario-2-selector');
    let scenario1InputSelector = document.getElementById('scenario-1-selector-inputs');
    let scenario2InputSelector = document.getElementById('scenario-2-selector-inputs');

    scenario1Selector.addEventListener('input', updateSummaries);
    scenario2Selector.addEventListener('input', updateSummaries);
    scenario1InputSelector.addEventListener('input', updateInputs);
    scenario2InputSelector.addEventListener('input', updateInputs);

    function updateSummaries() {
      let scenario1Id = scenario1Selector.value.split(' - ');
      let scenario2Id = (scenario2Selector.value == 'null') ? null : scenario2Selector.value.split(' - ');

      let scenarios = [scenario1Id];
      if (scenario2Id != null) scenarios.push(scenario2Id);

      renderSummary(scenarios);
    }

    function updateInputs() {
      let scenario1Id = scenario1InputSelector.value.split(' - ');
      let scenario2Id = (scenario2InputSelector.value == 'null') ? null : scenario2InputSelector.value.split(' - ');

      let scenarios = [scenario1Id];
      if (scenario2Id != null) scenarios.push(scenario2Id);

      renderInputs(scenarios);
    }

    updateSummaries();
    updateInputs();
  '''
  report.body.append(script)



def timelines_analysis(report_file_path=None, report_dir_path=None):
  #############################################################################
  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # TODO Remove this block and uncomment the lines below it
  # TEST
  import os
  import pickle
  from . import CACHE_DIR
  cache_filename = os.path.join(CACHE_DIR, 'timelines_analysis.pickle')

  if not os.path.exists(cache_filename):
    scenarios = ScenarioRunner()
    scenarios.simulate_all_scenarios()
    with open(cache_filename, 'wb') as f:
      pickle.dump(scenarios, f) # , pickle.HIGHEST_PROTOCOL)

  with open(cache_filename, 'rb') as f:
    scenarios = pickle.load(f)
  # end testing
  #############################################################################

  # TODO uncomment the lines below
  #scenarios = ScenarioRunner()
  #scenarios.simulate_all_scenarios()

  results = TimelinesAnalysisResults()
  results.scenario_groups = [group for group in scenarios.groups if group.name != 'normal']

  return results

if __name__ == '__main__':
  parser = init_cli_arguments()
  args = parser.parse_args()
  write_timelines_analysis_report(report_file_path=args.output_file, report_dir_path=args.output_dir)
