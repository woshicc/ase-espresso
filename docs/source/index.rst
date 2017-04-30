.. ase-espresso documentation master file, created by
   sphinx-quickstart on Wed Nov 25 10:09:02 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ase-espresso's documentation!
========================================

`ase-espresso`_ is a python interface for `Quantum Espresso`_ using the `Atomic Simulation Environment (ASE)`_.

The main purpose of the `ase-espresso`_ interface is to allow for
python-controlled ionic updates (e.g. ase-based structural relaxation) and to
provide post-processed `Quantum Espresso`_ output (e.g. charge densities, DOS) as
numpy_ arrays. While the `ase-espresso`_ interface can be used to create input
files for `Quantum Espresso`_ only, there are alternative python interfaces for
input file generation (or for running static calculations ionic step by ionic
step):

- `ase_qe_intrfce <http://www.qe-forge.org/gf/project/ase_qe_intrfce/>`_,
- `PWscfInput <http://physics.ucf.edu/%7Edle/blog.php?id=2>`_,
- `qecalc <https://pypi.python.org/pypi/qecalc/0.3.0>`_.


.. _ase-espresso: https://github.com/vossjo/ase-espresso
.. _Quantum Espresso: http://www.quantum-espresso.org/
.. _Atomic Simulation Environment (ASE): (https://wiki.fysik.dtu.dk/ase/
.. _numpy: http://www.numpy.org/

Contents:

.. toctree::
   :maxdepth: 2

   Installation <install>
   NEBEspresso <nebespresso>
   API Reference <api>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
