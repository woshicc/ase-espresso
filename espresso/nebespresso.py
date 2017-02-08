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

from io import open
import copy
from .espresso import Espresso, iEspresso
from .siteconfig import SiteConfig
import threading

__version__ = '0.3.1'


def splitinto(l, n):
    '''
    Split a list into `n` sublists of roughly equal size

    Args:
        l (`list`) :
            list to be split
        n (int) :
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
        neb (`ase.neb.NEB`) :
            The nudged elastic band object to associate the calculator with
        outprefix (str) :
            Prefix of the output directories for images, defaults to `neb`
        masterlog (str)
            Name of the log file, defaults to 'neb_master.log'
        site (`siteconfig.SiteConfig`)
            Site configuration object
    '''

    def __init__(self, neb, outprefix='neb', site=None, **kwargs):
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
        threads = []
        for job in self.jobs:
            t = threading.Thread(target=job['calc'].calculate,args=(job['image'],))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def _set_neb(self, neb):

        self.images = neb.images[1:len(neb.images) - 1]
        self.nimages = len(self.images)
        self.neb = neb
        self.neb.neb_orig_forces = self.neb.get_forces
        self.neb.get_forces = self.nebforce

    def nebforce(self):
        'Wait for the calcualtions to finish and the the NEB force'
        self.wait_for_total_energies()
        return self.neb.neb_orig_forces()

    def get_world(self):

        return self.calculators[0].get_world()
