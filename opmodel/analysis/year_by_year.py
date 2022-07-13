"""
Explore year by year.
"""

from . import log
from . import *

import math
from xml.etree import ElementTree as et

def explore_year(report_file_path=None, report_dir_path=None):
  if report_file_path is None:
    report_file_path = 'year_by_year.html'

  # Retrieve parameter table
  log.info('Retrieving parameters...')
  parameter_table = get_parameter_table()
  best_guess_parameters = {parameter : row['Best guess'] for parameter, row in parameter_table.iterrows()}

  # Run model
  log.info('Running simulation...')
  model = SimulateTakeOff(**best_guess_parameters, t_step=1)
  model.run_simulation()

  log.info('Writing report...')
  report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  #############################################################################
  # Year exploration

  report.add_header("Year by year analysis", level = 3)

  # Toggle mode (year by year or all years)
  report.content.append(et.fromstring(f'''
    <p>
      Mode
      <select id="mode-selector">
        <option value="all-years">Show all years at once</option>
        <option value="single-year">Show single year</option>
      </select>
    </p>
  '''))

  #-------------------------------------
  # Year by year
  #-------------------------------------

  year_by_year_container = et.Element('div', {'id': 'year-by-year-container'})
  report.content.append(year_by_year_container)

  # Year selector
  year_by_year_container.append(et.fromstring(f'''
    <p>
      Year
      <input type="number" id="year-selector" step="1" min="{model.t_start}" max="{(math.floor(model.t_end)-1)}" value="{model.t_start}"></input>
    </p>
  '''))

  all_years_table = []
  years = range(math.ceil(model.t_start), math.floor(model.t_end))
  for year in years:
    index = model.time_to_index(year)
    row = {
      'rampup':                       1 if model.rampup[index] else 0,
      'hardware performance':         np.log10(model.hardware_performance[index]),
      'frac_gwp_compute':             model.frac_gwp_compute[index],
      'hardware':                     np.log10(model.hardware[index]),
      'software':                     model.software[index],
      'compute':                      np.log10(model.compute[index]),
      'labour':                       model.labour[index] / model.labour[0],
      'capital':                      model.capital[index] / model.capital[0] * 0.025,
      'automatable tasks goods':      model.automatable_tasks_goods[index],
      'frac automatable tasks goods': model.frac_automatable_tasks_goods[index],
      'automatable tasks rnd':        model.automatable_tasks_rnd[index],
      'frac automatable tasks rnd':   model.frac_automatable_tasks_rnd[index],
      'gwp':                          model.gwp[index],
      'frac_capital_rnd':             model.frac_capital_hardware_rnd[index],
      'frac_labour_rnd':              model.frac_labour_hardware_rnd[index],
      'frac_compute_rnd':             model.frac_compute_hardware_rnd[index],
      'rnd input hardware':           model.rnd_input_hardware[index] / model.rnd_input_hardware[0] * 0.003048307243707020,
      'cumulative input hardware':    model.cumulative_rnd_input_hardware[index] / model.rnd_input_hardware[0] * 0.003048307243707020,
      'ratio rnd input hardware':     model.rnd_input_hardware[0]**model.rnd_parallelization_penalty / model.cumulative_rnd_input_hardware[0],
      'biggest_training_run':         np.log10(model.biggest_training_run[index]),
      'compute share goods':          model.compute_share_goods[index],
    }
    all_years_table.append(row.copy())

    # Yikes!
    row['gwp'] = f'{row["gwp"]:e}'
    row['rampup'] = 'Y' if row['rampup'] else 'N'

    table = report.add_data_frame(row, parent = year_by_year_container)
    table.set('id', f'year-{year}')
    table.set('class', f'{table.get("class")} year-data')

  #-------------------------------------
  # All years at once
  #-------------------------------------

  all_years_container = et.Element('div', {'id': 'all-years-container'})
  report.content.append(all_years_container)

  all_years_container.append(et.fromstring('<p>Show funny cell bars <input id="show-cell-bars" type="checkbox"></input></p>'))

  df = pd.DataFrame(all_years_table, index = years).style \
    .format({'gwp': '{:e}', 'rampup': lambda x: 'Y' if x else 'N'}) \
    .bar(color = '#ddd')
  report.add_data_frame(df, parent = all_years_container)


  #############################################################################
  # Wrap up

  #-------------------------------------
  # Inputs
  #-------------------------------------

  report.add_header("Inputs", level = 3)
  report.add_data_frame(best_guess_parameters, index = 'Best guess')

  #-------------------------------------
  # Javascript
  #-------------------------------------

  # Mode selector
  report.body.append(et.fromstring('''
    <script>
      let modeSelector = document.getElementById('mode-selector');
      let yearByYearContainer = document.getElementById('year-by-year-container');
      let allYearsContainer = document.getElementById('all-years-container');

      let modeUpdate = () => {
        yearByYearContainer.style.display = (modeSelector.value == 'single-year') ? 'initial' : 'none';
        allYearsContainer.style.display   = (modeSelector.value == 'all-years')   ? 'initial' : 'none';
      };

      modeSelector.addEventListener('input', modeUpdate);
      modeUpdate();
    </script>
  '''))

  # When changing the year number, show the comparison table for that year
  report.body.append(et.fromstring('''
    <script>
      let yearSelector = document.getElementById('year-selector');
      let yearUpdate = () => {
        document.querySelectorAll('.year-data').forEach(x => x.style.display = 'none');

        let selectedStep = document.getElementById('year-' + yearSelector.value);
        if (selectedStep) {
          selectedStep.style.display = 'initial';
        }
      };

      yearSelector.addEventListener('input', yearUpdate);
      yearUpdate();
    </script>
  '''))

  # Show cell bars
  report.body.append(et.fromstring('''
    <script>
      let checkbox = document.getElementById('show-cell-bars');

      onCheck = () => {
        if (checkbox.checked) {
          document.body.classList.remove('hide-cell-bars');
        } else {
          document.body.classList.add('hide-cell-bars');
        }
      };

      checkbox.addEventListener('input', onCheck);
      onCheck();
    </script>
  '''))

  report.head.append(et.fromstring('''
    <style>
      body.hide-cell-bars .dataframe-container td {
        background: initial !important;
      }
    </style>
  '''))

  #-------------------------------------
  # Write and done
  #-------------------------------------

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

if __name__ == '__main__':
  parser = init_cli_arguments()
  args = handle_cli_arguments(parser)
  explore_year(report_file_path=args.output_file, report_dir_path=args.output_dir)

