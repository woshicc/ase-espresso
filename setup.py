
# -*- coding: utf-8 -*-

"""
    Setup file for the ase-espresso package.
"""

from distutils.core import setup, Extension
from distutils import ccompiler
import shutil
import os

MAIN_PACKAGE = "espresso"
DESCRIPTION = "Python API for the Quantum Espresso software"
LICENSE = "GPLv3"
URL = "https://github.com/lmmentel/ase-espresso"
AUTHOR = "Lukasz Mentel"
EMAIL = "lmmentel@gmail.com"
VERSION = '0.1.2'

CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Topic :: Scientific/Engineering :: Chemistry',
               'Topic :: Scientific/Engineering :: Physics']

SCRIPTS = ['scripts/pwlog2traj']

def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()

extensions = [Extension('espfilter', sources=['c-src/espfilter.c']),
              Extension('cubecutperiodic', sources=['c-src/cubecutperiodic.c'])]

def setup_package():

    setup(name=MAIN_PACKAGE,
          version=VERSION,
          url=URL,
          description=DESCRIPTION,
          author=AUTHOR,
          author_email=EMAIL,
          license=LICENSE,
          long_description=readme(),
          classifiers=CLASSIFIERS,
          packages=['espresso'],
          scripts=SCRIPTS,
          ext_modules=extensions,
    )

if __name__ == "__main__":
    setup_package()
