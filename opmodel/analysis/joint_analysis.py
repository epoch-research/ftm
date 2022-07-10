from . import log
from . import *

from ..report.report import Report
from .exploration_analysis import explore
from .sensitivity_analysis import write_sensitivity_analysis_report
from .timelines_analysis import write_timelines_analysis_report
from .mc_analysis import write_mc_analysis_report

def joint_analysis(report_file_path=None, report_dir_path=None, report=None):
  if report_file_path is None:
    report_file_path = 'joint_report.html'

  new_report = report is None
  if new_report:
    report = Report(report_file_path=report_file_path, report_dir_path=report_dir_path)

  log.indent()
  report.begin_tab_group()
  report.begin_tab('Timelines analysis')
  write_timelines_analysis_report(report = report)
  log.deindent()

  log.indent()
  report.begin_tab('Sensitivity analysis')
  write_sensitivity_analysis_report(report = report)
  log.deindent()

  log.indent()
  report.begin_tab('Monte Carlo analysis')
  write_mc_analysis_report(report = report)
  log.deindent()

  report.end_tab_group()

  if new_report:
    report_path = report.write()
    log.info(f'Report stored in {report_path}')

  log.info('Done')

if __name__ == '__main__':
  parser = init_cli_arguments()
  args = parser.parse_args()
  joint_analysis(report_file_path=args.output_file, report_dir_path=args.output_dir)
