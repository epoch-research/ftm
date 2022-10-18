import os
import re
import time
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.formats.style
from xml.etree import ElementTree as et

DEFAULT_REPORT_DIRECTORY = '_output_'     # Relative to the root of the repository
DEFAULT_REPORT_FILE      = 'report.html'  # Relative to the DEFAULT_REPORT_DIRECTORY

from ..core.utils import *

# Tabs code taking from https://inspirationalpixels.com/creating-tabs-with-html-css-and-jquery/

class Report:
  def __init__(self, report_file_path=None, report_dir_path=None, add_csv_copy_button=False):
    self.report_file_path = report_file_path or DEFAULT_REPORT_FILE
    self.report_dir_path  = report_dir_path  or Report.default_report_path()

    self.make_tables_scrollable = True

    self.add_csv_copy_button = add_csv_copy_button

    self.param_names = get_param_names()
    self.metric_names = get_metric_names()
    self.most_important_metrics = get_most_important_metrics()
    self.most_important_parameters = get_most_important_parameters()

    self.tab_groups = []
    self.tab_groups_parents = []
    self.current_tab = None

    self.generated_ids = []

    self.html    = et.Element('html')
    self.head    = et.Element('head')
    self.body    = et.Element('body')
    self.content = et.Element('div', {'class': 'main'})

    # General styling
    self.head.append(et.fromstring('''
      <style>
        body {
          font-family: Arial, sans-serif;
          font-size: 15px;
        }

        .info-icon {
          margin-left: 0.5em;
        }

        .super-info-icon {
          position: relative;
          font-size: 0.8em;
          margin-left: 0;
          bottom: 0.5em;
        }

        .tippy-content {
          text-align: left;
        }

        .banner {
          background-color: #ddd;
          padding: 8px;
          text-align: center;
        }

        table {
          font-size: 1em;
        }

        .main {
          padding: 1em;
          padding-top: 0;
        }

        .main > * {
          grid-column: full-start/full-end;
        }

        .main > .figure-container {
          grid-column: full-start/full-end;
        }

        .figure-container img {
          max-width: calc(min(100%, 1200px));
          max-height: 550px;
        }

        table, td, tr, th {
          border: none;
        }

        table.dataframe {
          border-collapse: collapse;
          white-space: nowrap;
        }

        .table-container {
          margin-bottom: 1em;
        }

        table.dataframe thead {
          border-bottom: 1px solid #aaa;
          vertical-align: bottom;
          background-color: #ddd;
        }

        table.dataframe tfoot {
          background-color: #ddd;
        }

        table.dataframe td, table.dataframe th {
          text-align: right;
          padding: 0.2em 1.5em;
        }

        table.dataframe tbody tr:nth-child(odd) {
          background-color: #eee;
        }

        .show-only-important table.dataframe th:not(.important),
        .show-only-important table.dataframe td:not(.important)
        {
          display: none;
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

        [data-modal-trigger] {
          cursor: pointer;
        }

        .dataframe-modal .modal-content-content {
          overflow-y: auto;
          max-height: 90vh;
          max-width: calc(100vw - 100px);
          background-color: white;
        }

        #image-modal img {
          height: 90vh;
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

    # Tabs
    # See https://inspirationalpixels.com/creating-tabs-with-html-css-and-jquery/#step-css
    self.head.append(et.fromstring('''
      <style>
        :root {
          --border-radius: 7px;
        }

        /*----- Tabs -----*/
        .tabs {
          width:100%;
          display:inline-block;
        }

        .tab-links {
          padding-left: 0;
          height: 37px;
          border-bottom: 1px solid grey;
          border-left: 1px solid grey;
          border: none;
          overflow-y: hidden;
          margin-bottom: -2px;
          margin-top: 0;
        }

        /*----- Tab Links -----*/
        /* Clearfix */
        .tab-links:after {
          display:block;
          clear:both;
          content:'';
        }

        .tab-links li {
          float:left;
          list-style:none;
          border: 1px solid grey;
          border-left: none;
          background:#e7e7e7;
          border-top: 2px solid #333;
        }

        .tab-links li:first-child {
          border-top-left-radius: var(--border-radius);
          border-left: 2px solid #333;
        }

        .tab-links li:last-child {
          border-top-right-radius: var(--border-radius);
          border-right: 2px solid #333;
        }

        .tab-links span {
          padding:9px 15px;
          margin-right: 1px;
          display:inline-block;
          font-size:16px;
          font-weight:600;
          text-decoration: none;
          color:black;
          cursor: pointer;
        }

        .tab-links li:hover {
          background:#aaa;
        }

        .tab-links span:hover {
          text-decoration:none;
        }

        .tab-links li.active {
          background:white;
          border-bottom: 1px solid white;
        }

        /*----- Content of Tabs -----*/
        .tab-content {
          padding: 1em;
          background:#fff;
          overflow-x: auto;
          border-radius: var(--border-radius);
          border: 2px solid #333;
          border-top-left-radius: 0;
        }

        .tab {
          display:none;
        }

        .tab.active {
          display:block;
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

    self.head.append(et.fromstring('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.9.0/font/bootstrap-icons.css"></link>'))
    self.head.append(et.fromstring('<link rel="stylesheet" href="https://unpkg.com/tippy.js@6/themes/light-border.css" />'))
    self.head.append(et.fromstring('<script defer="true" src="https://cdn.jsdelivr.net/npm/clipboard@2.0.10/dist/clipboard.min.js"></script>'))
    self.head.append(et.fromstring('<script defer="true" src="https://unpkg.com/micromodal/dist/micromodal.min.js"></script>'))
    self.head.append(et.fromstring('<script defer="true" src="https://unpkg.com/panzoom@9.4.0/dist/panzoom.min.js"></script>'))
    self.head.append(et.fromstring('<script src="https://unpkg.com/@popperjs/core@2"></script>'))
    self.head.append(et.fromstring('<script src="https://unpkg.com/tippy.js@6"></script>'))
    self.head.append(et.fromstring('<script src="https://code.jquery.com/jquery-3.6.0.slim.min.js"></script>'))

    self.head.append(et.fromstring(f'''
      <script>
        let addCsvCopyButton = {'true' if add_csv_copy_button else 'false'};
      </script>
    '''));

    # Tippy plugins
    self.head.append(et.fromstring('''
      <script>
        // See https://atomiks.github.io/tippyjs/v6/plugins/#hideonesc
        const hideOnEsc = {
          name: 'hideOnEsc',
          defaultValue: true,
          fn({hide}) {
            function onKeyDown(event) {
              if (event.keyCode === 27) {
                hide();
              }
            }

            return {
              onShow() {
                document.addEventListener('keydown', onKeyDown);
              },
              onHide() {
                document.removeEventListener('keydown', onKeyDown);
              },
            };
          },
        };
      </script>
    '''));

    self.head.append(et.fromstring('''
      <script>
        window.addEventListener('load', () => {
          if (addCsvCopyButton) {
            let clipboard = new ClipboardJS('.copy-button');
          }

          for (let trigger of document.querySelectorAll('[data-modal-trigger]')) {
            trigger.addEventListener('click', () => {
              let modalId = trigger.dataset.modalTrigger;
              MicroModal.show(modalId);
            });
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
        });
      </script>
    '''))

    # Convert all timestamps to dates
    self.body.append(et.fromstring('''
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

        let observer = new MutationObserver(onNodeChange);
        observer.observe(document.body, { childList: true, subtree: true });
      </script>
    '''))

    # Importance selectors
    self.body.append(et.fromstring('''
      <script>
        function onImportanceSelect(selector) {
          // This sucks
          let tableContainer = selector.parentElement.nextElementSibling;

          if (selector.value == 'all') {
            tableContainer.classList.remove('show-only-important');
          } else {
            tableContainer.classList.add('show-only-important');
          }
        }
      </script>
    '''))

    # Tooltips explaining the params and metrics on hover
    script = et.Element('script')
    self.body.append(script);
    script.text = '''
        let paramNotes = ''' + json.dumps(get_parameters_meanings()) + ''';
        let paramJustifications = ''' + json.dumps(get_parameter_justifications()) + ''';
        let backgroundColors = ''' + json.dumps(get_parameters_colors()) + ''';
        let metricNotes = ''' + json.dumps(get_metrics_meanings()) + ''';

        function injectMeaningTooltip(node, meaning) {
          if (node._meaningInjected) return;

          if (meaning) {
            let icon = document.createElement('i');
            icon.classList.add('bi', 'bi-info-circle', 'info-icon');
            node.append(icon);

            tippy(icon, {
              content: meaning,
              allowHTML: true,
              interactive: true,
              placement: (node.parentElement.parentElement.tagName == 'THEAD') ? 'top' : 'right',
              appendTo: document.body,
              plugins: [hideOnEsc],
              theme: 'light-border',
            });

            node._meaningInjected = true;
          } else {
            // TODO Temporary, remove later
            if (node.parentElement.firstElementChild == node) {
              // When the ths run vertically, add extra padding to account for the missing icon
              node.style.paddingRight = '3em';
            }
          }
        }

        function injectMeaningTooltips() {
          for (let node of document.querySelectorAll('[data-param-id]')) {
            let paramId = node.dataset.paramId;
            let content = paramNotes[paramId];
            if (node.dataset.showJustification && paramId in paramJustifications && paramJustifications[paramId].length > 0) {
              content += '<br><br><span style="font-weight: bold">Justification for the best guess value:</span> ' + paramJustifications[paramId];
            }
            injectMeaningTooltip(node, content);
          }

          for (let node of document.querySelectorAll('[data-metric-id]')) {
            let content = metricNotes[node.dataset.metricId];
            if ('meaningSuffix' in node.dataset) {
              content += node.dataset.meaningSuffix;
            }
            injectMeaningTooltip(node, content);
          }
        }

        function injectParamBgColors() {
          document.querySelectorAll('[data-param-id]').forEach(node => {
            let color = backgroundColors[node.dataset.paramId];

            if (color) {
              node.style.backgroundColor = color;
            }
          });
        }

        window.addEventListener('load', () => {
          injectMeaningTooltips();
          injectParamBgColors();
        });
    '''

    # Figure modal
    self.body.append(et.fromstring('''
      <div class="modal micromodal-slide" id="image-modal" aria-hidden="true">
        <div class="modal-overlay" tabindex="-1" data-micromodal-close="true">
          <div class="modal-content-content">
            <img src="" />
          </div>
        </div>
      </div>
    '''))

    # Generation date
    self.content.append(et.fromstring(f'''
      <p style="color: grey">
        Generated at
        <span data-timestamp="{int(time.time() * 1000)}" class="page-generation-time" style="color: grey"></span>
      </p>
    '''))

    self.html.append(self.head)
    self.html.append(self.body)
    self.body.append(self.content)

    # Tabs
    # See https://inspirationalpixels.com/creating-tabs-with-html-css-and-jquery/#step-jquery
    self.body.append(et.fromstring('''
      <script>
        jQuery('.tabs .tab-links span').on('click', function(e) {
          var currentAttrValue = jQuery(this).data('href');

          // Show/Hide Tabs
          jQuery(`.tabs [data-id="${currentAttrValue.slice(1)}"]`).show().siblings().hide();

          // Change/remove current tab to active
          jQuery(this).parent('li').addClass('active').siblings().removeClass('active');

          history.replaceState(null, null, currentAttrValue);

          e.preventDefault();
        });
      </script>
    '''))

    self.body.append(et.fromstring('''
      <script>
        let tab;

        if (location.hash) {
          let tabName = location.hash.slice(1).split('-')[0];
          tab = document.querySelector(`.tab[data-id="${tabName}"]`);
        }

        // By default, activate the first tab
        if (!tab) {
          tab = document.querySelector('.tab');
        }

        //... if any
        if (tab) {
          let link = document.querySelector(`span[data-href="#${tab.dataset.id}"]`);

          tab.classList.add("active");
          link.parentElement.classList.add("active");
        }
      </script>
    '''))

    # Tooltips
    self.body.append(et.fromstring('''
      <script>
        for (let tooltipContainer of document.querySelectorAll(`[data-tooltip-content]`)) {
          tippy(tooltipContainer, {
            content: tooltipContainer.dataset.tooltipContent,
            allowHTML: true,
            interactive: true,
            trigger: tooltipContainer.dataset.tooltipTriggers,
            appendTo: document.body,
            theme: 'light-border',
            plugins: [hideOnEsc],
            onMount(instance) {
              if (tooltipContainer.dataset.tooltipOnMount) {
                eval(tooltipContainer.dataset.tooltipOnMount);
              }
            }
          });
        }
      </script>
    '''))

    self.default_parent = self.content

  def add_title(self, title):
    self.head.append(et.fromstring(f'<title>{title}</title>'))

  def add_header(self, title, level=1, parent=None):
    if parent is None: parent = self.default_parent

    element = et.Element(f'h{max(1, level)}', {'id': self.convert_to_id(title)})
    element.text = title
    parent.append(element)

  def add_element(self, element, parent=None):
    if parent is None: parent = self.default_parent
    parent.append(element)
    return element

  def add_html(self, html, parent=None):
    if parent is None: parent = self.default_parent

    element = et.fromstring(html)
    parent.append(element)
    return element

  def add_html_lines(self, html_lines, parent=None):
    self.add_html("\n".join(html_lines))

  def add_paragraph(self, paragraph, parent=None):
    return self.add_html(f'<p>{paragraph}</p>')

  def add_vspace(self, px=100, parent=None):
    if parent is None: parent = self.default_parent

    element = et.fromstring(f'<div style="height:{px}px"></div>')
    parent.append(element)

  def add_all_figures(self, parent=None):
    for fignum in plt.get_fignums():
      self.add_figure(plt.figure(fignum), parent)

  def add_figure(self, figure = None, parent = None):
    if parent is None: parent = self.default_parent

    if not figure: figure = plt.gcf()

    image = BytesIO()
    figure.savefig(image, format='svg', bbox_inches='tight')
    base64_image = base64.b64encode(image.getvalue()).decode('utf-8')

    container = et.Element('div', {'class': 'figure-container'})
    img = et.Element('img', {'class': 'figure', 'src': f'data:image/svg+xml;base64,{base64_image}'})
    container.append(img)

    parent.append(container)
    plt.close(figure)

    return container

  def add_data_frame_modal(self, df, modal_id, index = None, show_index = True, show_index_header = False, use_render = False, parent=None, show_justifications=False, **to_html_args):
    if parent is None: parent = self.default_parent

    modal             = et.Element('div', {'class': 'modal micromodal-slide dataframe-modal', 'id': modal_id, 'aria-hidden': 'true'})
    content_container = et.Element('div', {'class': 'modal-overlay', 'tabindex': '-1', 'data-micromodal-close': 'true'})
    content           = et.Element('div', {'class': 'modal-content-content'})

    self.add_data_frame(df, index, show_index, show_index_header, use_render, parent = content, **to_html_args)

    modal.append(content_container)
    content_container.append(content)
    parent.append(modal)

  def add_data_frame(
      self, df, index = None, show_index = True, show_index_header = False, use_render = False, parent=None, show_justifications=False,
      show_importance_selector=False, importance_layout = 'horizontal', important_rows_to_keep=[], important_columns_to_keep=[0], label = 'xxx', # TODO Document this
      nan_format = lambda **args: 'NaN',
      **to_html_args
    ):
    if parent is None: parent = self.default_parent

    if isinstance(df, dict):
      if not isinstance(index, (list, tuple)): index = [index]
      df = pd.DataFrame(df, index = index).transpose()

    if isinstance(df, pd.io.formats.style.Styler):
      df.set_table_attributes('class="dataframe"')

    html = df.to_html(index = show_index, index_names = show_index_header, **to_html_args)
    html = html.replace('&nbsp;', '') # hack!

    dataframe_wrapper = et.fromstring(f'<div class="dataframe-container">{html}</div>')

    self.process_table(dataframe_wrapper, show_justifications, nan_format)

    container = et.Element('div', {'class': 'table-container'})
    container.append(dataframe_wrapper)

    if self.make_tables_scrollable:
      container.set('style', 'overflow-x: auto')

    if self.add_csv_copy_button:
      copy_button_container = et.Element('div', {'class': 'copy-button-container'})
      copy_button_container.append(et.fromstring(f'<div class="copy-button" data-clipboard-text="{df.to_csv()}">Copy CSV</div>'))
      container.append(copy_button_container)

    parent.append(container)

    if show_importance_selector:
      self.add_importance_selector(container, importance_layout, important_rows_to_keep, important_columns_to_keep, label = label, parent = parent)

    return container

  def process_table(self, table, show_justifications = False, nan_format = lambda **args: 'NaN'):
    # Add data attributes
    for th in table.findall('.//th'):
      if th.text in self.param_names:
        th.attrib['data-param-id'] = th.text
        if show_justifications:
          th.attrib['data-show-justification'] = "true"
      if th.text in self.metric_names:
        th.attrib['data-metric-id'] = th.text

    if get_option('human_names', False):
      # Convert parameter and metric ids into human names
      # (in a hacky way)
      for th in table.findall('.//th'):
        if th.text in self.param_names:
          human_name = self.param_names[th.text]
          th.text = human_name

        if th.text in self.metric_names:
          human_name = self.metric_names[th.text]
          th.text = human_name

    def format_nans(row, col, index_r, index_c, cell):
      if cell.text == 'NaN':
        cell.text = nan_format(row = row, col = col, index_r = index_r, index_c = index_c, cell = cell)

    self.apply_to_table(table, format_nans)

  def add_importance_selector(self, table_container,
      layout = 'horizontal', important_rows_to_keep=[], important_columns_to_keep=[], keep_cell = None, # TODO Document all this
      label = 'xxx', parent = None,
    ):

    table = table_container.find('.//table')
    thead = table.find('.//thead')
    tbody = table.find('.//tbody')

    if layout == 'vertical':
      important_rows_to_keep.append(0)
    else:
      important_columns_to_keep.append(0)

    self.add_class(table_container, 'show-only-important')

    if keep_cell is None:
      keep_cell = lambda row, col, index_r, index_c, cell: \
          (row in important_rows_to_keep) or (index_r in self.most_important_parameters) \
          if layout == 'vertical' else \
          (col in important_columns_to_keep) or (index_c in self.most_important_metrics)

    def add_important_class(row, col, index_r, index_c, cell):
      if keep_cell(row, col, index_r, index_c, cell):
        self.add_class(cell, 'important')

    self.apply_to_table(table, add_important_class)

    # Create the selector
    selector = self.create_importance_selector(label)

    # Add the selector
    self.insert_before(parent, table_container, selector)

  def create_importance_selector(self, label):
    selector = et.fromstring(f'''
      <p class="importance-selector">
        Show
        <select onchange="onImportanceSelect(this)">
          <option value="important">the most important {label}</option>
          <option value="all">all {label}</option>
        </select>
      </p>
    ''')
    return selector

  def apply_to_table(self, table, func):
    indices_row = {}
    indices_col = {}

    for row, tr in enumerate(table.findall('.//tr')):
      element_id = ''
      if len(tr):
        el = tr[0]
        element_id = el.attrib.get('data-metric-id', el.attrib.get('data-param-id', el.text or ''))
      indices_row[row] = element_id

    for col, el in enumerate(table.find('.//tr')):
      element_id = el.attrib.get('data-metric-id', el.attrib.get('data-param-id', el.text or ''))
      indices_col[col] = element_id

    # Mark the important cells
    for row, tr in enumerate(table.findall('.//tr')):
      for col, cell in enumerate(tr):
        index_r = indices_row[row]
        index_c = indices_col[col]
        func(row, col, index_r, index_c, cell)

  def add_banner_message(self, message, classes = []):
    element = et.fromstring(f'<div>{message}</div>')
    element.set('class', 'banner ' + ' '.join(classes))
    self.body.insert(0, element)

  def generate_tooltip_html(self, content, on_mount = '', triggers = '', classes = ''):
    if classes: classes = ' ' + classes
    return f'''<i class="bi-info-circle super-info-icon{classes}" data-tooltip-triggers="{triggers}" data-tooltip-on-mount="{on_mount}" data-tooltip-content="{Report.escape(content)}"></i>'''

  def generate_tooltip(self, content, on_mount = '', triggers = '', classes = ''):
    return et.fromstring(generate_tooltip_html(content, on_mount))

  # ---------------------------------------------------------------------------
  # Tabs
  # ---------------------------------------------------------------------------

  def begin_tab_group(self, parent = None):
    if parent is None: parent = self.default_parent

    self.current_tab = None
    self.tab_groups.append([])
    self.tab_groups_parents.append(parent)

  def end_tab_group(self):
    group = self.tab_groups.pop()
    parent = self.tab_groups_parents.pop()

    tab_group_el = et.Element('div', {'class': 'tabs'})

    tab_links = et.Element('ul', {'class': 'tab-links'})
    for i, tab in enumerate(group):
      li = et.fromstring(f'<li><span data-href="#{tab.id}">{tab.name}</span></li>')
      tab_links.append(li)
    tab_group_el.append(tab_links)

    tab_content = et.Element('div', {'class': 'tab-content'})
    tab_group_el.append(tab_content)
    for i, tab in enumerate(group):
      tab.node.set('class', 'tab')
      tab.node.set('data-id', tab.id)
      tab_content.append(tab.node)

    parent.append(tab_group_el)
    self.default_parent = parent
    self.current_tab = None

  def begin_tab(self, name, id = None):
    node = et.Element('div', {'class': 'tab'})

    self.current_tab = Tab(name, node, id)
    self.tab_groups[-1].append(self.current_tab)
    self.default_parent = node


  # ---------------------------------------------------------------------------
  # Writing
  # ---------------------------------------------------------------------------

  def write(self):
    if os.path.isabs(self.report_file_path):
      report_abs_path = self.report_file_path
    else:
      # If the report path is relative, store it in the reports directory
      os.makedirs(self.report_dir_path, exist_ok=True)
      report_abs_path = os.path.join(self.report_dir_path, self.report_file_path)

    tree = et.ElementTree(self.html)
    et.indent(tree, space="  ", level=0)
    with open(report_abs_path, 'w') as f:
      f.write('<!DOCTYPE html>')
      tree.write(f, encoding='unicode', method='html')

    return report_abs_path

  def default_report_path():
    module_path = os.path.dirname(os.path.realpath(__file__))
    report_dir_path = os.path.abspath(os.path.join(module_path, '..', '..', DEFAULT_REPORT_DIRECTORY))
    return report_dir_path

  # ---------------------------------------------------------------------------
  # Utils
  # ---------------------------------------------------------------------------

  def insert_before(self, parent, reference, element):
    if parent is None: parent = self.default_parent
    parent.insert(list(parent).index(reference), element)

  def add_class(self, el, clazz):
    if 'class' not in el.attrib:
      el.attrib['class'] = ''
    el.attrib['class'] = (el.attrib['class'] + ' ' + clazz).strip()

  def convert_to_id(self, s):
    s = re.sub(' +', '-', s.lower())

    if self.current_tab:
      s = f'{self.current_tab.id}-{s}'

    if s in self.generated_ids:
      n = 2
      while f'{s}-n' in self.generated_ids:
        n += 1
      s = f'{s}-n'

    self.generated_ids.append(s)

    return s

  @staticmethod
  def escape(s):
      s = s.replace("&", "&amp;")
      s = s.replace("<", "&lt;")
      s = s.replace(">", "&gt;")
      s = s.replace("\"", "&quot;")
      return s

class Tab:
  id = 0

  def __init__(self, name, node, id = None):
    self.name = name
    self.node = node

    if id is None:
      self.id = f'tab_{Tab.id}'
      Tab.id += 1
    else:
      self.id = id

if __name__ == '__main__':
  import numpy as np

  report = Report(add_csv_copy_button=False, report_file_path='report-test.html')

  report.add_header("Title")
  report.add_paragraph("Today I've learned many things.")
  report.add_paragraph("Lfkasjflsa aafksl fklsajf lskajf lkasjf klkadsj fldkasj flkadsj fld sjlafjdaslfj adslfjdalsfj fladsjfds.")
  report.add_paragraph("""I'm a paragraph with <span style='color: red'>inner elements</span>.""")

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
