
# -*- coding: utf-8 -*-

"""
    Setup file for the ase-espresso package.
"""

from setuptools import setup

MAIN_PACKAGE = "aseqe"
DESCRIPTION = "Python API for the Quantum Espresso software"
LICENSE = "GPLv3"
URL = "https://github.com/lmmentel/ase-espresso"
AUTHOR = "Lukasz Mentel"
EMAIL = "lmmentel@gmail.com"
VERSION = '0.3.4'
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
               'Natural Language :: English',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Chemistry',
               'Topic :: Scientific/Engineering :: Physics']
DEPENDENCIES = ['ase',
                'future',
                'numpy',
                'path.py',
                'pexpect',
                'python-hostlist',
                'six']
KEYWORDS = 'chemistry physics quantum mechanics solid state'


def readme():
    'Return the contents of the README.md file.'
    with open('README.rst') as freadme:
        return freadme.read()


def setup_package():

    setup(name=MAIN_PACKAGE,
          version=VERSION,
          url=URL,
          description=DESCRIPTION,
          author=AUTHOR,
          author_email=EMAIL,
          license=LICENSE,
          keywords=KEYWORDS,
          long_description=readme(),
          classifiers=CLASSIFIERS,
          packages=['espresso'],
          install_requires=DEPENDENCIES,
          )


if __name__ == "__main__":
    setup_package()
