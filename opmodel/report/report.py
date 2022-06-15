import sys
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
from xml.etree import ElementTree as et
import os

DEFAULT_REPORT_DIRECTORY = '_output_'     # Relative to the root of the repository
DEFAULT_REPORT_FILE      = 'report.html'  # Relative to the DEFAULT_REPORT_DIRECTORY

class Report:
  def __init__(self, report_file_path=None, report_dir_path=None, add_csv_copy_button=False):
    self.report_file_path = report_file_path or DEFAULT_REPORT_FILE
    self.report_dir_path  = report_dir_path  or Report.default_report_path()

    self.add_csv_copy_button = add_csv_copy_button

    self.html    = et.Element('html')
    self.head    = et.Element('head')
    self.body    = et.Element('body')
    self.content = et.Element('div', {'class': 'main'})

    # General styling
    self.head.append(et.fromstring('''
      <style>
        .main {
          display: grid;
          grid-template-columns: [full-start] minmax(4vw,auto) [wide-start] minmax(auto,140px) [main-start] min(640px,calc(100% - 8vw)) [main-end] minmax(auto,140px) [wide-end] minmax(4vw,auto) [full-end];
          padding: 10px;
        }

        .main > * {
          grid-column: full-start/full-end;
        }

        .main > .figure-container {
          grid-column: full-start/full-end;
        }

        .figure-container img {
          max-width: 90vw;
        }

        table, td, tr, th {
          border: none;
        }

        table.dataframe {
          width: 100%;
          margin-left: auto;
          margin-right: auto;
          border-collapse: collapse;
          white-space: nowrap;
        }

        table.dataframe thead {
          border-bottom: 1px solid #aaa;
          vertical-align: bottom;
          background-color: #ddd;
        }

        table.dataframe td, table.dataframe th {
          text-align: right;
          padding: 0.2em 1.5em;
          text-wrap: nowrap;
        }

        table.dataframe tbody tr:nth-child(odd) {
          background-color: #eee;
        }

        table.dataframe tbody tr:hover {
          background-color: #ddd;
        }

        .copy-button {
          display: inline-block;
          background-color: #444;
          color: white;
          margin-top: 4px;
          font-size: 0.9rem;
          padding: 4px;
          border-radius: 5px;
        }

        .copy-button:hover {
          cursor: pointer;
        }

        .copy-button-container {
          text-align: right;
        }

        #image-modal img {
          height: 90%;
          max-width: 100%;
          cursor: move;
        }

        img.figure:hover {
          cursor: zoom-in;
        }

        *:focus {
          outline: none;
        }
      </style>
    '''))

    # Micromodal styling
    self.head.append(et.fromstring('''
      <style>
        /**************************
          Basic Modal Styles
        **************************/

        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0,0,0,0.75);
          display: flex;
          justify-content: center;
          align-items: center;
          z-index:10;
        }

        .modal-container {
          background-color: #fff;
          padding: 30px;
          height: 100%;
          border-radius: 4px;
          overflow-y: auto;
          box-sizing: border-box;
        }

        @supports (display: flex) {
          .modal-container {
            height: initial;
            max-height: 80vh;
          }
        }

        .modal-header {
          position: relative;
          display: block;
          height: 30px;
          margin-bottom: 20px;
        }

        @supports (display: flex) {
          .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
            height:initial;
            margin-bottom: 0px;
          }
        }

        .modal-title {
          position: absolute;
          top: 20px;
          left: 20px;
          margin-top: 0;
          margin-bottom: 0;
          font-weight: 600;
          font-size: 1.25rem;
          line-height: 1.25;
          color: #00449e;
          box-sizing: border-box;
        }

        .modal-close {
          position: absolute;
          top: 20px;
          right: 20px;
          background: transparent;
          border: 0;
          cursor: pointer;
          margin: 0px;
          padding: 0px;
        }

        @supports (display: flex) {
          .modal-title {
            position: static;
          }

          .modal-close {
            position: static;
          }
        }

        .modal-header .modal-close:before { content: "\2715"; }

        .modal-content {
          margin-top: 10px;
          margin-bottom: 10px;
          color: rgba(0,0,0,.8);
        }

        @supports (display: flex) {
          .modal-content {
            margin-top: 2rem;
            margin-bottom: 2rem;
            line-height: 1.5;
          }
        }

        .modal-btn {
          font-size: .875rem;
          padding-left: 1rem;
          padding-right: 1rem;
          padding-top: .5rem;
          padding-bottom: .5rem;
          background-color: #e6e6e6;
          color: rgba(0,0,0,.8);
          border-radius: .25rem;
          border-style: none;
          border-width: 0;
          cursor: pointer;
          -webkit-appearance: button;
          text-transform: none;
          overflow: visible;
          line-height: 1.15;
          margin: 0;
          will-change: transform;
          -moz-osx-font-smoothing: grayscale;
          -webkit-backface-visibility: hidden;
          backface-visibility: hidden;
          -webkit-transform: translateZ(0);
          transform: translateZ(0);
          transition: -webkit-transform .25s ease-out;
          transition: transform .25s ease-out;
          transition: transform .25s ease-out,-webkit-transform .25s ease-out;
        }

        .modal-btn-primary {
          background-color: #00449e;
          color: #fff;
        }

        /**************************
          Demo Animation Style
        **************************/

        @keyframes mmfadeIn {
          from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes mmfadeOut {
          from { opacity: 1; }
            to { opacity: 0; }
        }

        .micromodal-slide {
          display: none;
        }

        .micromodal-slide.is-open {
          display: block;
        }

        .micromodal-slide[aria-hidden="false"] .modal-overlay {
          animation: mmfadeIn .3s cubic-bezier(0.0, 0.0, 0.2, 1);
        }

        .micromodal-slide[aria-hidden="true"] .modal-overlay {
          animation: mmfadeOut .3s cubic-bezier(0.0, 0.0, 0.2, 1);
        }

        .micromodal-slide .modal-container,
        .micromodal-slide .modal-overlay {
          will-change: transform;
        }

        /**************************
          Custom styles for individual modals
        **************************/

        .modal-container button {
          outline: none;
          cursor: pointer !important;
        }

        .modal-container h2.modal-title {
          color: #595959;
        }

        .modal-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .modal-title {
          margin-top: 0;
          margin-bottom: 0;
          font-weight: 600;
          font-size: 1.25rem;
          line-height: 1.25;
          color: #00449e;
          box-sizing: border-box;
        }

        .modal-close {
          font-size: 24px;
        }

        .modal-content {
          margin-top: 2rem;
          margin-bottom: 2rem;
          line-height: 1.5;
          color: rgba(0,0,0,.8);
        }

        .modal-btn {
          padding: 10px 15px;
          background-color: #e6e6e6;
          border-radius: 4px;
          -webkit-appearance: none;
        }

        /**************************
          Mobile custom styles for individual modals
        **************************/

        @media only screen and (min-device-width : 320px) and (max-device-width : 480px) {
          .modal-container {
            width: 90% !important;
            min-width: 90% !important;
          }
          
          @supports (display: flex) {
            .modal-container {
              width: 90% !important;
              min-width: 90% !important;
            }
          }
        }
      </style>
    '''))

    self.head.append(et.fromstring('<script defer="true" src="https://cdn.jsdelivr.net/npm/clipboard@2.0.10/dist/clipboard.min.js"></script>'))
    self.head.append(et.fromstring('<script defer="true" src="https://unpkg.com/micromodal/dist/micromodal.min.js"></script>'))
    self.head.append(et.fromstring('<script defer="true" src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>'))

    self.head.append(et.fromstring(f'''
      <script type="text/javascript" charset="utf-8">
        let addCsvCopyButton = {'true' if add_csv_copy_button else 'false'};
      </script>
    '''));

    self.head.append(et.fromstring('''
      <script type="text/javascript" charset="utf-8">
        window.onload = () => {
          if (addCsvCopyButton) {
            let clipboard = new ClipboardJS('.copy-button');
          }

          let panzoomInstance;
          for (let img of document.querySelectorAll('img')) {
            img.addEventListener('click', () => {
              MicroModal.show('image-modal', {
                onShow: () => {
                  if (!panzoomInstance) {
                    panzoomInstance = panzoom(document.querySelector('#image-modal img'), {smoothScroll: false});
                  }
                  document.querySelector('#image-modal img').src = img.src;
                },
                onClose: () => {
                  panzoomInstance.dispose();
                  panzoomInstance = null;
                },
              });
            });
          }
        };
      </script>
    '''))

    self.body.append(et.fromstring('''
      <div class="modal micromodal-slide" id="image-modal" aria-hidden="true">
        <div class="modal-overlay" tabindex="-1" data-micromodal-close='true'>
          <div class="modal-content-content">
            <img src="" />
          </div>
        </div>
      </div>
    '''))

    self.html.append(self.head)
    self.html.append(self.body)
    self.body.append(self.content)

  def add_header(self, title, level=1):
    element = et.Element(f'h{max(1, level)}')
    element.text = title
    self.content.append(element)

  def add_paragraph(self, paragraph):
    element = et.Element('p')
    element.text = paragraph
    self.content.append(element)

  def add_all_figures(self):
    for fignum in plt.get_fignums():
      self.add_figure(plt.figure(fignum))

  def add_figure(self, figure = None):
    if not figure: figure = plt.gcf()

    image = BytesIO()
    figure.savefig(image, format='svg', bbox_inches='tight')
    base64_image = base64.b64encode(image.getvalue()).decode('utf-8')

    container = et.Element('div', {'class': 'figure-container'})
    img = et.Element('img', {'class': 'figure', 'src': f'data:image/svg+xml;base64,{base64_image}'})
    container.append(img)

    self.content.append(container)
    plt.close(figure)

  def add_data_frame(self, df):
    container = et.Element('div', {'class': 'table-container'})
    dataframe_wrapper = et.Element('div', {'class': 'dataframe-container'})

    table = et.fromstring(df.to_html())
    dataframe_wrapper.append(table)
    container.append(dataframe_wrapper)

    if self.add_csv_copy_button:
      copy_button_container = et.Element('div', {'class': 'copy-button-container'})
      copy_button_container.append(et.fromstring(f'<div class="copy-button" data-clipboard-text="{df.to_csv()}">Copy CSV</div>'))
      container.append(copy_button_container)

    self.content.append(container)

  def write(self):
    if os.path.isabs(self.report_file_path):
      report_abs_path = self.report_file_path
    else:
      # If the report path is relative, store it in the reports directory
      os.makedirs(self.report_dir_path, exist_ok=True)
      report_abs_path = os.path.join(self.report_dir_path, self.report_file_path)

    tree = et.ElementTree(self.html)
    et.indent(tree, space="  ", level=0)
    tree.write(report_abs_path, encoding='unicode', method='html')

    return report_abs_path

  def default_report_path():
    module_path = os.path.dirname(os.path.realpath(__file__))
    report_dir_path = os.path.abspath(os.path.join(module_path, '..', '..', DEFAULT_REPORT_DIRECTORY))
    return report_dir_path


if __name__ == '__main__':
  import numpy as np

  report = Report(add_csv_copy_button=False, report_file_path='report-test.html')

  report.add_header("Title")
  report.add_paragraph("Today I've learned many things.")
  report.add_paragraph("Lfkasjflsa aafksl fklsajf lskajf lkasjf klkadsj fldkasj flkadsj fld sjlafjdaslfj adslfjdalsfj fladsjfds.")

  # Some plot
  report.add_header("Foobar", level = 4)
  plt.plot([1, 2, 3, 4], [2, 4, 9, 16], 'o')
  report.add_figure()
  report.add_paragraph("Lfkasjflsa aafksl fklsajf lskajf lkasjf klkadsj fldkasj flkadsj fld sjlafjdaslfj adslfjdalsfj fladsjfds.")

  # Some data frame
  num = np.array([[ 0.17899619, 0.17899619,  0.17899619, 0.33093259,  0.2076353,   0.06130814, 0.33093259,  0.2076353,   0.06130814,],
                  [ 0.20392888, 0.20392888,  0.20392888, 0.42653105,  0.33325891,  0.10473969, 0.42653105,  0.33325891,  0.10473969,],
                  [ 0.17038247, 0.17038247,  0.17038247, 0.19081956,  0.10119709,  0.09032416, 0.19081956,  0.10119709,  0.09032416,],
                  [-0.10606583,-0.10606583, -0.10606583,-0.13680513, -0.13129103, -0.03684349,-0.13680513, -0.13129103, -0.03684349,],
                  [ 0.20319428, 0.20319428,  0.20319428, 0.28340985,  0.20994867,  0.11728491, 0.28340985,  0.20994867,  0.11728491,],
                  [ 0.04396872, 0.04396872,  0.04396872, 0.23703525,  0.09359683,  0.11486036, 0.23703525,  0.09359683,  0.11486036,],
                  [ 0.27801304, 0.27801304,  0.27801304,-0.05769304, -0.06202813,  0.04722761,-0.05769304, -0.06202813,  0.04722761,],])
  days = ['5 days', '10 days', '20 days', '60 days', '120 days', '240 days', 'a', 'b', 'c']
  prices = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'CRM', 'EXPE', 'FB']
  df = pd.DataFrame(num, index=prices, columns=days)

  report.add_data_frame(df)

  # More plot
  report.add_header("Foobar", level = 4)
  plt.plot([1, 2, 3, 4], [8, 8, 8, 8], '--')
  report.add_figure()

  plt.close()
  report.add_header("Several figures", level = 4)

  plt.figure(figsize=(14, 8), dpi=80)
  plt.title(f"Figure 1")
  plt.plot([1, 2, 3, 4], [1, 4, 9, 16], '--')
  plt.plot([10, 20, 3, 4], [1, 8, 4, 2], '--')

  plt.figure(figsize=(14, 8), dpi=80)
  plt.title(f"Figure 2")
  plt.plot([1, 2, 3, 4], [1, 4, 9, 16], '--')
  plt.plot([10, 20, 3, 4], [1, 8, 4, 2], '--')

  report.add_all_figures()

  report_path = report.write()
  print(f'Report written to {report_path}')
