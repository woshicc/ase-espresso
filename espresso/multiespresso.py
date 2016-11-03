# -*- coding: utf-8 -*-

# ****************************************************************************
# Copyright (C) 2013 SUNCAT
# This file is distributed under the terms of the
# GNU General Public License. See the file `COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************

from __future__ import print_function, absolute_import

from io import open
import copy
import sys
from .espresso import Espresso
from .siteconfig import SiteConfig

__version__ = '0.2.0'


def splitinto(l, k):
    '''
    Split a list into `k` sublists of roughly equal size

    Args:
        l : list
            List to be split
        k : int
            number of sublists
    '''

    if len(l) % k == 0:
        n = len(l) // k
    elif len(l) % k != 0 and len(l) > k:
        n = len(l) // k + 1

    if len(l) % n == 0:
        splits = len(l) // n
    elif len(l) % n != 0 and len(l) > n:
        splits = len(l) // n + 1
    else:
        splits = 1

    return [l[n * i:n * i + n] for i in range(splits)]


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

    def __init__(self, images, outprefix='neb', masterlog='neb_master.log',
                 site=None, **kwargs):
        '''
        Set the necessary parameters
        '''

        self.calc_args = kwargs

        self.images = images
        self.nimages = len(self.images)

        self.outprefix = outprefix
        self.masterlog = masterlog

        self.jobs = []

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

    def _create_calculators(self):
        'Create the calculator instances'

        imageprocs = splitinto(self.site.proclist, self.nimages)

        for i, (image, procs) in enumerate(zip(self.images, imageprocs)):

            site = copy.deepcopy(self.site)

            site.proclsit = procs
            site.nprocs = len(site.proclist)
            site.usehostfile = True

            calc_args = self.calc_args.copy()
            calc_args['outdir'] = '{0:s}_{1:04d}'.format(self.outprefix, i)
            calc_args['site'] = site

            self.jobs.append({'image': image,
                              'calc': Espresso(**calc_args),
                              'done': False,
                              'procs': procs})

    def _set_images(self):

        for job in self.jobs:
            job['image'].set_calculator(job['calc'])

    def write_hostfiles(self):

        for job in self.jobs:

            job['calc'].initialize(job['image'])

        self.mycpus = self.localtmp + '/myprocs%{0:0>4d}.txt'.format(procrange)
        with open(self.mycpus, 'w') as fcpu:
            for i in range(i1, i1 + self.myncpus):
                fcpu.write(procs[i])

    def wait_for_total_energies(self):

        mlog = open(self.masterlog, 'ab')

        for job in self.jobs:
            job['calc'].initialize(job['image'])
            job['done'] = False

        running = True
        while running:

            running = False

            for job in self.jobs:

                if job['calc'].recalculate:

                    if not job['done']:
                        
                        a = self.calculators[i].cerr.readline()
                        running |= (a != '' and a[:17] != '!    total energy')
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
                            mlog.flush()
        print('', file=mlog)
        mlog.close()

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
