from espresso import NEBEspresso,iEspresso
from ase.build import molecule
from ase.neb import NEB
from ase.optimize.fire import FIRE as QuasiNewton
from asetools import smart_cell
from ase.io.trajectory import Trajectory
from ase.io import write,read

# Optimise molecule
initial = molecule('C2H6')
smart_cell(initial,vac=4.0,h=0.01)
write('initial.traj',initial)

# Create final state
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]
write('final.traj',final)

# Generate blank images
images = [initial]
nimage = 7

for i in range(nimage):
    images.append(initial.copy())

images.append(final)

# Run IDPP interpolation
neb = NEB(images)
neb.interpolate()

calcs = NEBEspresso(neb,kpts='gamma',pw=300,dw=4000)

# Run NEB calculation
qn = QuasiNewton(neb, logfile='ethane_linear.log')

for j in range(nimage):
  traj = Trajectory('neb%d.traj' % j, 'w', images[j+1])
  qn.attach(traj)

qn.run(fmax=0.05)
