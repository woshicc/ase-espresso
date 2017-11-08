
# test adapted from Quantum Espresso PWSCF v.5.3.0 (svn rev. 11974)
#   : PW/examples/example02

from __future__ import print_function

import numpy as np

from ase import Atoms
from ase.units import Rydberg, Bohr
from ase.optimize import BFGS
from espresso import Espresso, iEspresso


def test_relax_co_qe_bfgs(tmpdir):

    tmpdir.chdir()

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


def test_relax_co_ase_bfgs(tmpdir):

    tmpdir.chdir()

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


def test_relax_co_ase_interactive_bfgs(tmpdir):

    tmpdir.chdir()

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

    ref_ene = -616.9017404193379

    ref_pos = np.array([[1.19233393e+00, 0.0e+00, 0.00000000e+00],
                        [1.48996586e-03, 0.00000000e+00, 0.00000000e+00]])

    ref_for = np.array([[0.00162288, 0.0, 0.0],
                        [-0.00162288, 0.0, 0.0]])

    assert np.allclose(co.get_potential_energy(), ref_ene)
    #assert np.allclose(co.positions, ref_pos)
    #assert np.allclose(co.get_forces(), ref_for)

    print(co.positions)
    print(co.get_forces())
