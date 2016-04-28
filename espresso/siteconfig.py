# -*- coding: utf-8 -*-

__version__ = '0.1.2'

import os, sys
from subprocess import check_output, call, CalledProcessError

class SiteConfig(object):

    def __init__(self, scheduler):

        self.scheduler = scheduler
        self.set_variables()

    def set_variables(self):

        if self.scheduler is None:
            self.set_interactive()
        elif self.scheduler.lower() == 'slurm':
            self.set_slurm_env()
        elif self.scheduler.lower() in ['pbs', 'torque']:
            self.set_pbs_env()

    def set_interactive(self):
        'Set the variables necessary for interactive runs'

        self.batchmode = False
        self.submitdir = os.path.dirname(os.path.realpath(sys.argv[0]))
        self.jobid = os.gepid()
        if os.getenv('SCRATCH') is not None:
            self.scratch = os.getenv('SCRATCH')
        elif os.getenv('TMPDIR') is not None:
            self.scratch = os.getenv('TMPDIR')
        else:
            self.scratch = os.path.join(self.submitdir, 'tmp')

    def set_slurm_env(self):
        'Get enviromental variables associated with SLURM scheduler'

        self.jobid = os.getenv('SLURM_JOB_ID')
        self.batchmode = self.jobid is not None

        if self.batchmode:
            self.scratch = os.getenv('SCRATCH')
            self.nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
            self.submitdir = os.getenv('SUBMITDIR')
            self.tpn = int(os.getenv('SLURM_TASKS_PER_NODE').split('(')[0])
            jobnodelist = os.getenv('SLURM_JOB_NODELIST')
            output = check_output(['scontrol', 'show', 'hostnames', jobnodelist])
            nodeslist = output.split('\n')[:-1]
            self.procs = [nodeslist[i//self.tpn] for i in range(len(nodeslist)*self.tpn)]
            self.nprocs = len(self.procs)

            self.perHostMpiExec = 'mpirun -host ' + ','.join(nodeslist)+' -np {0:d}'.format(self.nnodes)
            self.perProcMpiExec = 'mpirun -wdir {0:s} {1:s}'
            self.perSpecProcMpiExec = 'mpirun -machinefile {0:s} -np {1:d} -wdir {2:s} {3:s}'

    def set_pbs_env(self):
        'Get enviromental variables associated with PBS/TORQUE scheduler'

        self.jobid = os.getenv('PBS_JOBID')
        self.batchmode = self.jobid is not None

        if self.batchmode:
            self.scratch = os.getenv('SCRATCH')
            if not os.path.exists(self.scratch):
                self.scratch = os.path.join('/tmp', os.getenv('USER'))

            self.submitdir = os.getenv('PBS_O_WORKDIR')

            nodefile = os.getenv('PBS_NODEFILE')
            with open(nodefile, 'r') as nf:
                self.procs = [x.strip() for x in nf.readlines()]

            self.nprocs = len(self.procs)
            uniqnodes = sorted(set(self.procs))

            uniqnodefile = os.path.realpath(os.path.join(self.scratch, 'uniqnodefile'))
            with open(uniqnodefile, 'w') as unf:
                for node in uniqnodes:
                    unf.write(node)

            self.perHostMpiExec = ['mpiexec', '-machinefile',  uniqnodefile, '-np', str(len(uniqnodes))]
            self.perProcMpiExec = 'mpiexec -machinefile {nf:s} -np {np:s}'.format(nf=nodefile, np=str(self.nprocs)) + ' -wdir {0:s} {1:s}'
            self.perSpecProcMpiExec = 'mpiexec -machinefile {0:s} -np {1:d} -wdir {2:s} {3:s}'

    @classmethod
    def check_scheduler(cls):

        scheduler = None

        # check id SLURM is installed and running
        try:
            out = check_output('scontrol version', shell=True)
            scheduler = 'SLURM'
        except CalledProcessError:
            pass

        # check if PBS/TORQUE is installed and running
        try:
            out = check_output('ps aux | grep pbs | grep -v grep', shell=True)
            scheduler = 'PBS'
        except CalledProcessError:
            pass

        return cls(scheduler)

    # methods for running espresso

    def do_perProcMpiExec(self, workdir, program):

        return os.popen2(self.perProcMpiExec.format(workdir, program))

    def do_perProcMpiExec_outputonly(self, workdir, program):

        return os.popen(self.perProcMpiExec.format(workdir, program), 'r')

    def runonly_perProcMpiExec(self, workdir, program):

        call(self.perProcMpiExec.format(workdir, program))

    def do_perSpecProcMpiExec(self, machinefile, nproc, workdir, program):

        return os.popen3(self.perSpecProcMpiExec.format(machinefile, nproc, workdir, program))
