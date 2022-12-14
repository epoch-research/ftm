from . import log
from . import *

from .report import Report
from .exploration_analysis import explore
from .sensitivity_analysis import write_sensitivity_analysis_report
from .timelines_analysis import write_timelines_analysis_report
from .mc_analysis import write_mc_analysis_report, McAnalysisResults

def megareport(report_file_path=None, report_dir_path=None, quick_test_mode=False, report=None,
    mc_trials=None, mc_input_results_filename=None, variance_reduction_params={}, variance_reduction_method=False):
  if mc_trials is None: mc_trials = 100
  if report_file_path is None:
    report_file_path = 'megareport.html'

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  log.info('Generating timelines analysis tab')
  log.indent()
  report.begin_tab_group()
  report.begin_tab('Timelines analysis', 'timelines_analysis')
  write_timelines_analysis_report(report = report)
  log.deindent()
  log.info()

  log.info('Generating parameter importance analysis tab')
  log.indent()
  report.begin_tab('Parameter importance analysis', 'parameter_importance_analysis')
  write_sensitivity_analysis_report(report = report, quick_test_mode = quick_test_mode,
      variance_reduction_params=variance_reduction_params, method=variance_reduction_method)
  log.deindent()
  log.info()

  log.info('Generating Monte Carlo analysis tab')
  log.indent()
  report.begin_tab('Monte Carlo analysis', 'mc_analysis')
  if quick_test_mode:
    write_mc_analysis_report(report = report, n_trials = 4 if mc_trials is None else mc_trials, input_results_filename = mc_input_results_filename)
  else:
    write_mc_analysis_report(report = report, n_trials = mc_trials, input_results_filename = mc_input_results_filename)
  log.deindent()
  log.info()

  report.end_tab_group()

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  parser = init_cli_arguments()

  parser.add_argument(
    "-q",
    "--quick-test-mode",
    action='store_true',
  )

  parser.add_argument(
    "-n",
    "--mc-trials",
    type=int,
  )

  parser.add_argument(
    "-c",
    "--rank-correlations-url",
    type=str,
    default=None,
  )

  parser.add_argument(
    "--input-results-file",
    help = 'Read the MC results from this file (pickle) instead of regenerating them',
  )

  parser.add_argument(
    "--variance-reduction-restore-dir",
  )

  parser.add_argument(
    "-m",
    "--variance-reduction-method",
    default = 'one_at_a_time',
    choices = [
      'one_at_a_time',
      'variance_reduction_on_margin',
      'shapley_values',
      'combined',
    ]
  )

  args = handle_cli_arguments(parser)

  variance_reduction_params = {
      'restore_dir': args.variance_reduction_restore_dir
  }

  set_option('rank_correlations_sheet_url', args.rank_correlations_url)

  megareport(
    report_file_path=args.output_file, report_dir_path=args.output_dir,
    quick_test_mode=args.quick_test_mode, mc_trials=args.mc_trials,
    mc_input_results_filename=args.input_results_file,
    variance_reduction_params=variance_reduction_params,
    variance_reduction_method=args.variance_reduction_method,
  )
