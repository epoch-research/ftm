from . import log
from . import *

from .exploration_analysis import explore

def timelines_analysis(report_file_path=None, report_dir_path=None):
  if report_file_path is None:
    report_file_path = 'timelines_analysis.html'

  timelines_parameters = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv&gid=2101919125')
  timelines_parameters = timelines_parameters.set_index(timelines_parameters.columns[0])

  parameter_table = pd.read_csv('https://docs.google.com/spreadsheets/d/1r-WxW4JeNoi_gCMc5y2iTlJQnan_LLCF5s_V4ZDDMkI/export?format=csv#gid=0')
  parameter_table = parameter_table.set_index("Parameter")

  report = Report(report_file_path = report_file_path, report_dir_path = report_dir_path)

  for timeline in timelines_parameters:
    parameters = timelines_parameters[timeline]

    log.info('----------------------------------------------------')
    log.info(timeline)

    report.add_header(timeline)

    parameter_table.loc['full_automation_requirements_training', 'Conservative'] =  parameters['Full automation requirements']
    parameter_table.loc['full_automation_requirements_training', 'Best guess']   =  parameters['Full automation requirements']
    parameter_table.loc['full_automation_requirements_training', 'Aggressive']   =  parameters['Full automation requirements']

    parameter_table.loc['flop_gap_training', 'Conservative'] = parameters['Long FLOP gap']
    parameter_table.loc['flop_gap_training', 'Best guess']   = parameters['Med FLOP gap']
    parameter_table.loc['flop_gap_training', 'Aggressive']   = parameters['Short FLOP gap']

    log.indent()
    explore(exploration_target = 'compare', parameter_table = parameter_table, report = report)
    log.deindent()

    report.add_vspace()

    log.info('')

  report_path = report.write()
  log.info(f'Report stored in {report_path}')

if __name__ == '__main__':
  parser = init_cli_arguments()
  args = parser.parse_args()
  timelines_analysis(report_file_path=args.output_file, report_dir_path=args.output_dir)
