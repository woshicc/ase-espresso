
'''
test adapted from https://wiki.fysik.dtu.dk/ase/tutorials/neb/idpp.html#example-1-ethane
'''

from __future__ import print_function

from espresso import iEspresso, NEBEspresso, NEBEspressoOld
from ase.build import molecule
from ase.neb import NEB, NEBTools
from ase.optimize.fire import FIRE as QuasiNewton
from asetools import smart_cell
from ase.io.trajectory import Trajectory


def test_ethene_rotation():

    # Optimise molecule
    initial = molecule('C2H6')
    smart_cell(initial, vac=4.0, h=0.01)

    # Create final state
    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]

    # Generate blank images
    images = [initial]
    nimage = 7

    for i in range(nimage):
        images.append(initial.copy())

    images.append(final)

    # Run IDPP interpolation
    neb = NEB(images)
    neb.interpolate('idpp')

    calcs = NEBEspressoOld(neb, kpts='gamma', pw=300, dw=4000)

    # Run NEB calculation
    qn = QuasiNewton(neb, logfile='ethane_linear.log')

    for j in range(nimage):
        traj = Trajectory('neb%d.traj' % j, 'w', images[j + 1])
        qn.attach(traj)

    qn.run(fmax=0.05)

    nt = NEBTools(neb.images)
    print('fmax: ', nt.get_fmax())
    print('Ef, dE: ', nt.get_barrier())


def test_ethene_rotation_new():

    # Optimise molecule
    initial = molecule('C2H6')
    smart_cell(initial, vac=4.0, h=0.01)
    initial.set_calculator(iEspresso(pw=300, dw=4000, kpts='gamma'))
    qn = QuasiNewton(initial, 'initial.traj')
    qn.run(fmax=0.01)

    # Create final state
    final = initial.copy()
    final.positions[2:5] = initial.positions[[3, 4, 2]]
    final.set_calculator(iEspresso(pw=300, dw=4000, kpts='gamma'))
    final.get_potential_energy()

    # Generate blank images
    images = [initial]
    nimage = 7

    for i in range(nimage):
        image = initial.copy()
        image.set_calculator(iEspresso(pw=300, dw=4000, kpts='gamma'))
        images.append(image)
    images.append(final)

    # Run IDPP interpolation
    neb = NEBEspresso(images)
    neb.interpolate('idpp')

    # Run NEB calculation
    qn = QuasiNewton(neb, logfile='ethane_linear.log', trajectory='neb.traj')
    qn.run(fmax=0.05)

    nt = NEBTools(neb.images)
    print('fmax: ', nt.get_fmax())
    print('Ef, dE: ', nt.get_barrier())

test_ethene_rotation()
test_ethene_rotation_new()
