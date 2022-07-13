from . import log
from . import *

from ..report.report import Report
from .exploration_analysis import explore
from .sensitivity_analysis import write_sensitivity_analysis_report
from .timelines_analysis import write_timelines_analysis_report
from .mc_analysis import write_mc_analysis_report

def megareport(report_file_path=None, report_dir_path=None, quick_test_mode=False, report=None):
  if report_file_path is None:
    report_file_path = 'megareport.html'

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  log.info('Generating timelines analysis tab')
  log.indent()
  report.begin_tab_group()
  report.begin_tab('Timelines analysis')
  write_timelines_analysis_report(report = report)
  log.deindent()
  log.info()

  log.info('Generating sensitivity analysis tab')
  log.indent()
  report.begin_tab('Sensitivity analysis')
  write_sensitivity_analysis_report(report = report, quick_test_mode = quick_test_mode)
  log.deindent()
  log.info()

  log.info('Generating Monte Carlo analysis tab')
  log.indent()
  report.begin_tab('Monte Carlo analysis')
  if quick_test_mode:
    write_mc_analysis_report(report = report, n_trials = 4)
  else:
    write_mc_analysis_report(report = report)
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
  args = handle_cli_arguments(parser)
  megareport(report_file_path=args.output_file, report_dir_path=args.output_dir, quick_test_mode=args.quick_test_mode)
