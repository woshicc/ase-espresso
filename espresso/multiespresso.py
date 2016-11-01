# -*- coding: utf-8 -*-

# ****************************************************************************
# Copyright (C) 2013 SUNCAT
# This file is distributed under the terms of the
# GNU General Public License. See the file `COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************

from __future__ import print_function, absolute_import

import sys
from .espresso import Espresso
from .siteconfig import SiteConfig

__version__ = '0.2.0'


class NEBEspresso(object):
    '''
    Special calculator running multiple Espresso calculators in parallel.
    Useful for e.g. nudged elastic band calculations.

    Args:
        images : list of ase.Atoms
            Images along the reaction path as a list of ase.Atoms objects
        outprefix : str, default=`neb`
            Prefix of the output directories for images
        log : str, default='neb_master.log'
            Name of the log file
    '''

    def __init__(self, images, outprefix='neb', log='neb_master.log',
                 **kwargs):
        '''
        Set the necessary parameters
        '''

        qe_args = kwargs.copy()
        qe_args['single_calculator'] = False

        self.images = images
        self.nimages = len(self.images)

        self.outprefix = outprefix
        self.log = log
        self.calculators = []

        self.site = site

    @property
    def site(self):
        return self._site

    @site.setter
    def site(self, value):
        if value is None:
            self._site = SiteConfig.check_scheduler()
        else:
            self._site = value



        # Initialize lists of cpu subsets if needed
        if procrange is None:
            self.proclist = False
        else:
            self.proclist = True
            procs = self.site.procs + []
            procs.sort()
            nprocs = len(procs)
            self.myncpus = nprocs / numcalcs
            i1 = self.myncpus * procrange
            self.mycpus = self.localtmp + '/myprocs%{0:0>4d}.txt'.format(procrange)
            with open(self.mycpus, 'w') as fcpu:
                for i in range(i1, i1 + self.myncpus):
                    fcpu.write(procs[i])


    def _assign_calcs(self):

        self.done = [False] * self.images


        for i in range(self.nimages):
            qe_args['outdir'] = '{0:s}_{1:04d}'.format(self.outprefix, i)
            qe_args['procrange'] = i
            esp = Espresso(**qe_args)
            self.calculators.append(esp)


    def wait_for_total_energies(self):

        s = open(self.txt, 'a')
        for i in range(self.ncalc):
            self.calculators[i].init_only(self.images[i])
            self.done[i] = False
        notdone = True
        while notdone:
            notdone = False
            for i in range(self.ncalc):
                if self.calculators[i].recalculate:
                    if not self.done[i]:
                        a = self.calculators[i].cerr.readline()
                        notdone |= (a != '' and a[:17] != '!    total energy')
                        if a[:13] == '     stopping':
                            raise RuntimeError('problem with calculator #{}'.format(i))
                        elif a[:20] == '     convergence NOT':
                            raise RuntimeError('calculator #{} did not converge'.format(i))
                        elif a[1:17] != '    total energy':
                            sys.stderr.write(a)
                        else:
                            if a[0] != '!':
                                self.done[i] = False
                                print('current free energy (calc. %3d; in scf cycle) :' % i, a.split()[-2], 'Ry', file=s)
                            else:
                                self.done[i] = True
                                print('current free energy (calc. %3d; ionic step) :  ' % i, a.split()[-2], 'Ry', file=s)
                            s.flush()
        print('', file=s)
        s.close()

    def set_images(self, images):

        if len(images) != self.ncalc:
            raise ValueError("number of images ({0}) doesn't match number of calculators ({1})".format(len(images), self.ncalc))

        for i in range(self.ncalc):
            images[i].set_calculator(self.calculators[i])
        self.images = images

    def set_neb(self, neb):
        self.set_images(neb.images[1:len(neb.images) - 1])
        self.neb = neb
        self.neb.neb_orig_forces = self.neb.get_forces
        self.neb.get_forces = self.nebforce

    def nebforce(self):
        self.wait_for_total_energies()
        return self.neb.neb_orig_forces()

    def get_world(self):
        return self.calculators[0].get_world()
