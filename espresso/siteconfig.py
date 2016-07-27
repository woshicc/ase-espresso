# -*- coding: utf-8 -*-

from __future__ import division

import contextlib
import os
import sys
import tempfile
import functools
from subprocess import call

import hostlist
from path import Path

__version__ = '0.1.2'


def preserve_cwd(function):
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        cwd = os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            os.chdir(cwd)
    return decorator


@contextlib.contextmanager
def working_directory(path):
    '''
    A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    '''

    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class SiteConfig(object):

    '''
    Site configuration holding details about the execution environment
    with methods for retrieving the details from systems variables and
    creating directories

    Args:
        scheduler : str
            Name of the scheduler, curretly supports only 'SLURM' and
            'PBS'/'TORQUE'
        scratchenv : str
            Name of the envoronmental variable that defines the scratch path
    '''

    def __init__(self, scheduler, scratchenv='SCRATCH'):

        self.scheduler = scheduler
        self.scratchenv = scratchenv
        self.localtmp = None
        self.global_scratch = None
        self.user_scratch = None
        self.submitdir = None
        self.jobid = None
        self.set_variables()

    def set_variables(self):
        '''
        Resolve the site attributes based on the scheduler used
        '''

        if self.scheduler is None:
            self.set_interactive()
        elif self.scheduler.lower() == 'slurm':
            self.set_slurm_env()
        elif self.scheduler.lower() in ['pbs', 'torque']:
            self.set_pbs_env()

    @classmethod
    def check_scheduler(cls):
        '''
        Check if either SLURM or PBS/TORQUE are running
        '''

        scheduler = None

        # check id SLURM is installed and running
        with open(os.devnull, 'w') as devnull:
            exitcode = call('scontrol version', shell=True, stderr=devnull)
            if exitcode == 0:
                scheduler = 'SLURM'

        # check if PBS/TORQUE is installed and running
        with open(os.devnull, 'w') as devnull:
            exitcode = call('ps aux | grep pbs | grep -v grep', shell=True,
                            stderr=devnull)
            if exitcode == 0:
                scheduler = 'PBS'

        return cls(scheduler)

    def set_interactive(self):
        '''
        Set the attributes necessary for interactive runs

        - `batchmode` is False
        - `jobid` is set to the PID
        - 'global_scratch' checks for scratch under `self.scratchenv` if it is
          not defined used current directory
        '''

        self.batchmode = False
        self.submitdir = Path(os.path.dirname(os.path.realpath(sys.argv[0])))
        self.jobid = os.getpid()

        if os.getenv(self.scratchenv) is not None:
            self.global_scratch = Path(os.getenv(self.scratchenv))
        else:
            self.global_scratch = self.submitdir

    def set_global_scratch(self):
        'Set the global scratch attribute'

        scratch = os.getenv(self.scratchenv)

        if scratch is None:
            raise OSError('variable ${} is undefied'.format(scratch))
        else:
            if os.path.exists(scratch):
                self.global_scratch = Path(scratch)
            else:
                raise OSError('scratch directory <{}> defined with ${} does not exist'.format(
                    scratch, self.scratchenv))

    def set_slurm_env(self):
        '''
        Set the attributes necessary to run the job based on the
        enviromental variables associated with SLURM scheduler
        '''

        self.batchmode = True

        env = {'jobid': 'SLURM_JOB_ID',
               'submitdir': 'SUBMITDIR'}

        for name, var in env.items():
            value = os.getenv(var)
            if value is None:
                raise OSError('variable ${} is undefined'.format(var))
            else:
                setattr(self, name, value)

        self.set_global_scratch()

        self.nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
        self.tpn = int(os.getenv('SLURM_TASKS_PER_NODE').split('(')[0])
        jobnodelist = hostlist.expand_hostlist(os.getenv('SLURM_JOB_NODELIST'))

        self.hosts = [jobnodelist[i // self.tpn] for i in range(len(jobnodelist) * self.tpn)]
        self.nprocs = len(self.hosts)

        self.perHostMpiExec = ['mpirun', '-host', ','.join(jobnodelist),
                               '-np', '{0:d}'.format(self.nnodes)]
        self.perProcMpiExec = 'mpirun -wdir {0:s} {1:s}'
        self.perSpecProcMpiExec = 'mpirun -machinefile {0:s} -np {1:d} -wdir {2:s} {3:s}'

    def set_pbs_env(self):
        '''
        Set the attributes necessary to run the job based on the
        enviromental variables associated with PBS/TORQUE scheduler
        '''

        self.batchmode = True

        env = {'jobid': 'PBS_JOBID',
               'submitdir': 'PBS_O_WORKDIR'}

        for name, var in env.items():
            value = os.getenv(var)
            if value is None:
                raise OSError('variable ${} undefined'.format(var))
            else:
                setattr(self, name, value)

        self.set_global_scratch()

        nodefile = os.getenv('PBS_NODEFILE')
        with open(nodefile, 'r') as nf:
            self.hosts = [x.strip() for x in nf.readlines()]

        self.nprocs = len(self.hosts)
        uniqnodes = sorted(set(self.hosts))

        self.perHostMpiExec = ['mpirun', '-host', ','.join(uniqnodes),
                               '-np', '{0:d}'.format(len(uniqnodes))]

        self.perProcMpiExec = 'mpiexec -machinefile {nf:s} -np {np:s}'.format(
            nf=nodefile, np=str(self.nprocs)) + ' -wdir {0:s} {1:s}'
        self.perSpecProcMpiExec = 'mpiexec -machinefile {0:s} -np {1:d} -wdir {2:s} {3:s}'

    # methods for running espresso

    def do_perProcMpiExec(self, workdir, program):

        return os.popen2(self.perProcMpiExec.format(workdir, program))

    def do_perProcMpiExec_outputonly(self, workdir, program):

        return os.popen(self.perProcMpiExec.format(workdir, program), 'r')

    def runonly_perProcMpiExec(self, workdir, program):

        call(self.perProcMpiExec.format(workdir, program))

    def do_perSpecProcMpiExec(self, machinefile, nproc, workdir, program):

        return os.popen3(self.perSpecProcMpiExec.format(machinefile, nproc,
                                                        workdir, program))

    def make_localtmp(self, workdir):
        '''
        Create a temporary local directory for the job

        Args:
            workdir : str
                Name of the working directory for the run
        '''

        if workdir is None or len(workdir) == 0:
            prefix = '_'.join(['qe', str(os.getuid()), str(self.jobid)])
            self.localtmp = Path(tempfile.mkdtemp(prefix=prefix, suffix='_tmp',
                                                  dir=self.submitdir))
        else:
            self.localtmp = Path(tempfile.mkdtemp(prefix=workdir,
                                                  dir=self.submitdir))

        return self.localtmp.abspath()

    def make_scratch(self):
        '''
        Create a user scratch dir on each node (in the global scratch area)
        in batchmode or a single local scratch directory otherwise
        '''

        prefix = '_'.join(['qe', str(os.getuid()), str(self.jobid)])
        self.user_scratch = Path(tempfile.mkdtemp(prefix=prefix,
                                                  suffix='_scratch',
                                                  dir=self.global_scratch))

        with working_directory(str(self.localtmp)):
            if self.batchmode:
                exitcode = call(self.perHostMpiExec +
                                ['mkdir', '-p', str(self.user_scratch)])
            else:
                self.user_scratch.makedirs_p()

        return self.user_scratch.abspath()
