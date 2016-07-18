# -*- coding: utf-8 -*-

# ****************************************************************************
# Copyright (C) 2013 SUNCAT
# This file is distributed under the terms of the
# GNU General Public License. See the file `COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
# ****************************************************************************

'subroutines for creation of subdirectories & clean-up'

from __future__ import print_function

import os
import tempfile
import string
import random
import subprocess
from path import Path

__version__ = '0.1.2'


def mklocaltmp(workdir, site):
    '''
    Create a temporary local directory for the job

    Args:
        workdir : str
        site : espresso.siteconfig.SiteConfig
    '''

    if workdir is None or len(workdir) == 0:
        suffix = '_' + ''.join(random.choice(string.uppercase + string.digits) for _ in range(6))
        tempdir = Path(tempfile.mkdtemp(prefix='qe' + str(site.jobid), suffix=suffix, dir=site.submitdir))
    else:
        tempdir = Path(os.path.join(site.submitdir, workdir))
        tempdir.makedirs_p()
    return tempdir.abspath()


def mkscratch(localtmp, site):
    '''
    Create a scratch dir on each node

    Args:
        localtmp : str
        site : espresso.siteconfig.SiteConfig
    '''

    suffix = '_' + ''.join(random.choice(string.uppercase + string.digits) for _ in range(6))
    tempdir = Path(tempfile.mkdtemp(prefix='qe' + site.jobid, suffix=suffix, dir=site.scratch))

    if site.batchmode:
        cwd = os.getcwd()
        localtmp.chdir()
        subprocess.call(site.perHostMpiExec + ['mkdir', '-p', str(tempdir)])
        os.chdir(cwd)
    return tempdir.abspath()
