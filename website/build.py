#!/usr/bin/env python

import argparse
import base64
import re
import shutil
import os

SRC_DIR = 'src'
BUILD_DIR = 'build'

os.makedirs(BUILD_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('command', nargs = '?')
args = parser.parse_args()

def build():
  print('Building')

  os.makedirs(BUILD_DIR, exist_ok=True)

  def cp(filename):
    shutil.copyfile(os.path.join(SRC_DIR, filename), os.path.join(BUILD_DIR, filename))

  def cpdir(name):
    shutil.rmtree(os.path.join(BUILD_DIR, name), ignore_errors = True)
    shutil.copytree(os.path.join(SRC_DIR, name), os.path.join(BUILD_DIR, name))

  def read(filename):
    with open(os.path.join(SRC_DIR, filename), 'r') as f:
      return f.read()

  def write(content, filename):
    with open(os.path.join(BUILD_DIR, filename), 'w') as f:
      return f.write(content)

  cp('favicon.svg')
  cp('spinner.svg')

  cpdir('css')
  cpdir('js')
  cpdir('img')
  cpdir('libs')

  header = read('header.html')
  footer = read('footer.html')

  def process_header(header, page):
    header = header.replace('{{{{playground-link-active}}}}', ' active' if (page == 'playground') else '')
    header = header.replace('{{{{reports-link-active}}}}', ' active' if (page == 'reports') else '')
    header = header.replace('{{{{description-link-active}}}}', ' active' if (page == 'description') else '')
    header = header.replace('{{{{about-link-active}}}}', ' active' if (page == 'about') else '')
    return header

  playground = read('playground.html')
  playground = f'''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Playground</title>
    {process_header(header, 'playground')}
    {playground}
    {footer}
  '''

  write(playground, 'playground.html')
  write(playground, 'index.html')

  reports = read('reports.html')
  reports = extract_figures(reports, 'img/reports')
  reports = f'''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Reports</title>
    {process_header(header, 'reports')}
    {reports}
    {footer}
  '''

  write(reports, 'reports.html')

  description = read('description.html')
  description = f'''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Description</title>
    {process_header(header, 'description')}
    {description}
    {footer}
  '''

  write(description, 'description.html')

  about = read('about.html')
  about = f'''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>About</title>
    {process_header(header, 'about')}
    {about}
    {footer}
  '''

  write(about, 'about.html')

  four_oh_four = read('404.html')
  four_oh_four = f'''
    <!DOCTYPE html>
    <meta charset="utf-8">
    <title>Ooops!</title>
    {process_header(header, 'four_oh_four')}
    {four_oh_four}
    {footer}
  '''

  write(four_oh_four, '404.html')

  print('Done')

def extract_figures(html, figures_dir):
  # Move base64-embedded figures into separate files so the page itself stays small
  os.makedirs(os.path.join(BUILD_DIR, figures_dir), exist_ok=True)
  count = 0

  def replace(match):
    nonlocal count
    count += 1
    filename = f'{figures_dir}/figure-{count:02}.svg'
    with open(os.path.join(BUILD_DIR, filename), 'wb') as f:
      f.write(base64.b64decode(match.group(2)))
    return f'<img{match.group(1)}src="{filename}" loading="lazy"{match.group(3)}>'

  return re.sub(r'<img([^>]*?)src="data:image/svg\+xml;base64,([^"]+)"([^>]*)>', replace, html)

def clean():
  print('Cleaning')
  os.rmdir(BUILD_DIR)

if args.command == 'clean':
  clean()
else:
  build()
