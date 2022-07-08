from . import log
from . import *

from .exploration_analysis import add_scenario_group_to_report
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

  for group in results.scenario_groups:
    report.add_header(group.name, level = 2)
    add_scenario_group_to_report(group, report)
    report.add_vspace()

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

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
