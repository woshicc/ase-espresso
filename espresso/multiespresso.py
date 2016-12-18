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
from .espresso import Espresso,iEspresso
from .siteconfig import SiteConfig

__version__ = '0.2.0'


def splitinto(l, n):
    '''
    Split a list into `n` sublists of roughly equal size

    Args:
        l : list
            List to be split
        n : int
            number of sublists
    '''

    q, r = divmod(len(l), n)
    indices = [q * i + min(i, r) for i in range(n + 1)]
    return [l[indices[i]:indices[i + 1]] for i in range(n)]


class NEBEspresso(object):
    '''
    Special calculator running multiple Espresso calculators in parallel.
    Useful for e.g. nudged elastic band calculations.

    Args:
        neb : ase.neb.NEB
            The nudged elastic band object to associate the calculator with
        outprefix : str, default=`neb`
            Prefix of the output directories for images
        masterlog : str, default='neb_master.log'
            Name of the log file
        site : SiteConfig
            SiteConfig object
    '''

    def __init__(self, neb, outprefix='neb',site=None, **kwargs):
        '''
        Set the necessary parameters
        '''

        self.calc_args = kwargs

        self._set_neb(neb)

        self.outprefix = outprefix

        self.jobs = []

        self.site = site

        self._create_calculators()
        self._associate_calculators()

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

            site.proclist = procs
            site.nprocs = len(site.proclist)
            site.usehostfile = True

            calc_args = self.calc_args.copy()
            calc_args['outdir'] = '{0:s}_{1:04d}'.format(self.outprefix, i)
            calc_args['site'] = site

            self.jobs.append({'image': image,
                              'calc': iEspresso(**calc_args),
                              })

    def _associate_calculators(self):

        for job in self.jobs:
            job['image'].set_calculator(job['calc'])

    def wait_for_total_energies(self):

        for job in self.jobs:
            job['calc'].calculate(job['image'])

    def _set_neb(self, neb):

        self.images = neb.images[1:len(neb.images) - 1]
        self.nimages = len(self.images)
        self.neb = neb
        self.neb.neb_orig_forces = self.neb.get_forces
        self.neb.get_forces = self.nebforce

    def nebforce(self):

        self.wait_for_total_energies()
        return self.neb.neb_orig_forces()

    def get_world(self):

        return self.calculators[0].get_world()
