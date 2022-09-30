#!/usr/bin/env python

import argparse
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

  header = read('header.html')

  playground = read('playground.html')
  playground = f'''
    <!DOCTYPE html>
    <title>Playground</title>
    {header}
    {playground}
  '''

  write(playground, 'playground.html')

  megareport = read('megareport.html')
  megareport = f'''
    <!DOCTYPE html>
    <title>Megareport</title>
    {header}
    {megareport}
  '''

  write(megareport, 'megareport.html')

  four_oh_four = read('404.html')
  four_oh_four = f'''
    <!DOCTYPE html>
    <title>Ooops</title>
    {header}
    {four_oh_four}
  '''

  write(four_oh_four, '404.html')

def clean():
  print('Cleaning')
  os.rmdir(BUILD_DIR)

if args.command == 'clean':
  clean()
else:
  build()
