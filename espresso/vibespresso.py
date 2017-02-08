# -*- coding: utf-8 -*-

# ****************************************************************************
# Original work Copyright (C) 2013-2015 SUNCAT
# Modified work Copyright 2015-2017 Lukasz Mentel
#
# This file is distributed under the terms of the
# GNU General Public License. See the file 'COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************

from __future__ import print_function, absolute_import

from builtins import object

import numpy as np

from ase.calculators.calculator import FileIOCalculator
from .espresso import Espresso

__version__ = '0.3.1'


class Vibespresso(FileIOCalculator, object):
    """
    Special espresso calculator, which expects the first calculation to
    be performed for a structure without displacements. All subsequent
    calculations are then initialized with the Kohn-Sham potential of
    the first calculation to speed up vibrational calculations.
    """
    def __init__(self, outdirprefix='out', **kwargs):
        """
        In addition to the parameters of a standard espresso calculator,
        outdirprefix (default: 'out') can be specified, which will be the
        prefix of the output of the calculations for different displacements
        """

        self.arg = kwargs.copy()
        self.outdirprefix = outdirprefix
        self.counter = 0
        self.equilibriumdensity = outdirprefix + '_equi.tgz'
        self.firststep = True
        self.ready = False

        self.atoms = None

    def update(self, atoms):
        if self.atoms is not None:
            x = atoms.positions - self.atoms.positions
            if np.max(x) > 1.0e-13 or np.min(x) < -1.0e-13:
                self.ready = False
        else:
            self.atoms = atoms.copy()
        self.runcalc(atoms)
        if atoms is not None:
            self.atoms = atoms.copy()

    def runcalc(self, atoms):
        if not self.ready:
            self.arg['outdir'] = self.outdirprefix + '_%04d' % self.counter
            self.counter += 1
            if self.firststep:
                self.esp = Espresso(**self.arg)
                self.esp.set_atoms(atoms)
                self.esp.get_potential_energy(atoms)
                self.esp.save_chg(self.equilibriumdensity)
                self.firststep = False
            else:
                self.arg['startingpot'] = 'file'
                self.esp = Espresso(**self.arg)
                self.esp.set_atoms(atoms)
                self.esp.initialize(atoms)
                self.esp.load_chg(self.equilibriumdensity)
                self.esp.get_potential_energy(atoms)
                self.esp.stop()
            self.ready = True

    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        if force_consistent:
            return self.esp.energy_free
        else:
            return self.esp.energy_zero

    def get_forces(self, atoms):
        self.update(atoms)
        return self.esp.forces

    def get_name(self):
        return 'VibEspresso'

    def get_version(self):
        return __version__
