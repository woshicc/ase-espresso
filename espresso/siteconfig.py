import os
from subprocess import check_output, call, Popen

class SiteConfig(object):

    def __init__(self, scheduler):

        self.scheduler = scheduler
        self.set_variables()

    def set_variables(self):

        if self.scheduler.lower() == 'slurm':
            self.get_slurm_env()
        elif self.scheduler.lower() in ['pbs', 'torque']:
            self.get_pbs_env()
        else:
            raise NotImplemented('{} support not implemented yet'.format(self.scheduler))

    def get_slurm_env(self):

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

            self.perHostMpiExec = 'mpirun -host '+','.join(nodeslist)+' -np {0:d}'.format(self.nnodes)
            self.perProcMpiExec = 'mpirun -wdir {0:s} {1:s}'
            self.perSpecProcMpiExec = 'mpirun -machinefile {0:s} -np {1:d} -wdir {2:s} {3:s}'

    def get_pbs_env(self):

        raise NotImplemented('PBS support not implemented yet')

    # methods for running espresso

    def do_perProcMpiExec(self, workdir, program):

        return os.popen2(self.perProcMpiExec.format(workdir, program))

    def do_perProcMpiExec_outputonly(self, workdir, program):

        return os.popen(self.perProcMpiExec.format(workdir, program), 'r')

    def runonly_perProcMpiExec(self, workdir, program):

        call(self.perProcMpiExec.format(workdir, program))

    def do_perSpecProcMpiExec(self, machinefile, nproc, workdir, program):

        return os.popen3(self.perSpecProcMpiExec.format(machinefile, nproc, workdir, program))
