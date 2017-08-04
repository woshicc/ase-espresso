============
ase-espresso
============

|Documentation|

`ase-espresso <https://github.com/lmmentel/ase-espresso>`__ provides a
Python interface compatible with `Atomic Simulation Environment
(ASE) <https://wiki.fysik.dtu.dk/ase/index.html>`__ for managing
calculations with the `Quantum
Espresso <http://www.quantum-espresso.org/>`__ code.

This is a fork from `vossjo <https://github.com/vossjo/ase-espresso>`__
that offers a lot of improvements over the original version, the most
important ones include:

-  the files were restructured into a python package
-  a ``setup.py`` file was introduced to allow installation through
   `pip <https://pip.pypa.io/en/stable/>`__ or
   `setuptools <https://pypi.python.org/pypi/setuptools>`__
-  configuration for the documentation is provided through
   `sphinx <http://www.sphinx-doc.org/en/stable/>`__ and a lot of
   docstrings were updated
-  the ``site.cfg`` is obsolete now, and no additional configuration is
   required, the functionality is replaced by a new ``SiteConfig`` class
   that dynamically gathers information about the execution environment
-  the old ``espresso`` class is now split into two: ``Espresso``
   preserving the standard functionality and ``iEspresso`` responsible
   for dynamic/interactive jobs with a custom version of pw.x
-  changes were made to establish python 3.x compatibility
-  the ``Espresso`` class were restructured according to
   `ase <https://wiki.fysik.dtu.dk/ase/index.html>`__ guidelines
   regarding calculator objects to support full compatibility with
   `ase <https://wiki.fysik.dtu.dk/ase/index.html>`__
-  most of the system calls are now handled by
   `pexpect <https://pexpect.readthedocs.io/en/stable>`__ and
   `subprocess <https://docs.python.org/2/library/subprocess.html>`__
   instead of the  ``os.system``, ``os.popen()``, ``os.popen2()``,
   ``os.popen3()``
-  tests were added
-  code style and readability were improved

Installation
============

Dependencies
------------

-  `Atomic Simulation Environment
   (ASE) <https://wiki.fysik.dtu.dk/ase/index.html>`__
-  `numpy <http://www.numpy.org/>`__
-  `pexpect <https://pexpect.readthedocs.io/en/stable>`__
-  `future <http://python-future.org/>`__
-  `path.py <https://github.com/jaraco/path.py>`__
-  `python-hostlist <https://www.nsc.liu.se/~kent/python-hostlist/>`__

The recommended installation method is with
`pip <https://pip.pypa.io/en/stable/>`__. The current version can be
installed directly from
`github <https://github.com/lmmentel/ase-espresso>`__:

.. code:: bash

    pip install https://github.com/lmmentel/ase-espresso/archive/master.zip

or cloned first

.. code:: bash

    git clone https://github.com/lmmentel/ase-espresso.git

and installed via

.. code:: bash

    pip install ./ase-espresso

Documentation
-------------

The documentation is hosted on
`ase-espresso.readthedocs.io <http://ase-espresso.readthedocs.io/en/latest/>`__.

You can also generate the documentation locally using
`sphinx <http://www.sphinx-doc.org/en/stable/>`__ by going to the
``docs`` directory and typing:

.. code:: bash

    make html

The built documentation can be viewed in a any browser

.. code:: bash

    firefox build/html/index.html

.. |Documentation| image:: https://readthedocs.org/projects/ase-espresso/badge/?version=latest
   :target: http://ase-espresso.readthedocs.io/en/latest/?badge=latest
