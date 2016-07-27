
# test adapted from Quantum Espresso PWSCF v.5.3.0 (svn rev. 11974)
#   : PW/examples/example02

from __future__ import print_function

import numpy as np

from ase import Atoms
from ase.units import Rydberg, Bohr
from ase.optimize import BFGS
from espresso import Espresso, iEspresso


def test_relax_co_qe_bfgs():

    co = Atoms('CO', [[2.256 * Bohr, 0.0, 0.0], [0.0, 0.0, 0.0]])
    co.set_cell(np.ones(3) * 12.0 * Bohr)

    calc = Espresso(pw=24.0 * Rydberg,
                    dw=144.0 * Rydberg,
                    kpts='gamma',
                    xc='PBE',
                    calculation='relax',
                    ion_dynamics='bfgs',
                    spinpol=False,
                    outdir='qe_bfgs')

    co.set_calculator(calc)
    calc.calculate(co)

    print('qe bfgs:')
    print('energy: ', co.get_potential_energy())
    print('positions: ', co.positions)
    print('forces: ', co.get_forces())


def test_relax_co_ase_bfgs():

    co = Atoms('CO', [[2.256 * Bohr, 0.0, 0.0], [0.0, 0.0, 0.0]])
    co.set_cell(np.ones(3) * 12.0 * Bohr)

    calc = Espresso(pw=24.0 * Rydberg,
                    dw=144.0 * Rydberg,
                    kpts='gamma',
                    xc='PBE',
                    calculation='scf',
                    ion_dynamics=None,
                    spinpol=False,
                    outdir='qe_ase_bfgs')

    co.set_calculator(calc)

    minimizer = BFGS(co, logfile='minimizer.log', trajectory='relaxed.traj')
    minimizer.run(fmax=0.01)

    print('ase(scf) bfgs:')
    print('energy: ', co.get_potential_energy())
    print('positions: ', co.positions)
    print('forces: ', co.get_forces())


def test_relax_co_ase_interactive_bfgs():

    co = Atoms('CO', [[2.256 * Bohr, 0.0, 0.0], [0.0, 0.0, 0.0]])
    co.set_cell(np.ones(3) * 12.0 * Bohr)

    calc = iEspresso(pw=24.0 * Rydberg,
                     dw=144.0 * Rydberg,
                     kpts='gamma',
                     xc='PBE',
                     calculation='relax',
                     ion_dynamics='ase3',
                     spinpol=False,
                     outdir='qe_ase_interactive_bfgs')

    co.set_calculator(calc)

    minimizer = BFGS(co, logfile='minimizer.log', trajectory='relaxed.traj')
    minimizer.run(fmax=0.01)

    print('ase(interactive) bfgs:')
    print('energy: ', co.get_potential_energy())
    print('positions: ', co.positions)
    print('forces: ', co.get_forces())

test_relax_co_qe_bfgs()
test_relax_co_ase_bfgs()
test_relax_co_ase_interactive_bfgs()
