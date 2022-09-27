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

  def read(filename):
    with open(os.path.join(SRC_DIR, filename), 'r') as f:
      return f.read()

  def write(content, filename):
    with open(os.path.join(BUILD_DIR, filename), 'w') as f:
      return f.write(content)

  shutil.rmtree(os.path.join(BUILD_DIR, 'assets'))
  shutil.copytree(os.path.join(SRC_DIR, 'assets'), os.path.join(BUILD_DIR, 'assets'))

  shutil.rmtree(os.path.join(BUILD_DIR, 'ftm'))
  shutil.copytree(os.path.join(SRC_DIR, 'ftm'), os.path.join(BUILD_DIR, 'ftm'))

  header = read('header.html')

  playground = read('playground.html')
  playground = f'''
    {header}
    {playground}
  '''

  write(playground, 'playground.html')

  megareport = read('megareport.html')
  megareport = f'''
    {header}
    {megareport}
  '''

  write(megareport, 'megareport.html')

def clean():
  print('Cleaning')
  os.rmdir(BUILD_DIR)

if args.command == 'clean':
  clean()
else:
  build()
