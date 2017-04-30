NEBEspresso
===========

NEBEspresso is a special calculator that allows to parallelize the calcualtions
over images i.e. each image calculation utilized a a set of CPUs that are assigned to
it from the pool of all available CPU for the NEB calcualtion.


Ethene methylation over H-SAPO-5
--------------------------------


.. code-block:: python

   import ase.io
   from espresso import iEspresso, NEBEspresso
   from ase.optimizers import LBFGS

   initial = ase.io.read('SAPO-5-CH3--ethene.traj')
   final = ase.io.read('H-SAPO-5--propene.traj')

   images = [initial]
   for _ in range(5):
       image = initial.copy()
       image.set_calculator(iEspresso(pw=600, dw=7000, calculation='relax', ion_dynamics='ase3'))
       images.append(image)
   images.append(final)

   neb = NEBEspresso(images, ouuprefix='neb')
   neb.interpolate()

   optimizer = LBFGS(neb, trajectory='neb.traj', logfile='optimizer.log')
   optimizer.run(fmax=0.05)
