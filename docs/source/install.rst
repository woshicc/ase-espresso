Dependencies
============

- ASE_
- `numpy <http://www.numpy.org/>`_
- `pexpect <https://pexpect.readthedocs.io/en/stable/>`_
- `future <http://python-future.org/>`_
- `path.py <http://pythonhosted.org/path.py/>`_


Installation
============

.. warning::

   The instructions below are only valid for the branch lmmfixes_. For
   instructions regarding the version from vossjo_ please refer to the wiki_.

The recommended installation method is with pip_. The current
branch (lmmfixes_) can be installed directly from github_:

.. code-block:: bash

   pip install https://github.com/lmmentel/ase-espresso/archive/lmmfixes.zip

or cloned first

.. code-block:: bash

   git clone https://github.com/lmmentel/ase-espresso.git -b lmmfixes

and installed via

.. code-block:: bash

   pip install ./ase-espresso


You can verify that your installation was successful by opening a python console
and trying to import :py:class:`Espresso <espresso.espresso.Espresso>`::

   >>> from espresso import Espresso


Configuration
=============

To run properly `ase-espresso`_ requires that the `Quantum Espresso`_ code is
properly compiled and the executables are available to the shell. You can to that
by extending the ``PATH`` variable with the location of your `Quantum Espresso`_ 

.. code-block:: bash

   export PATH=$PATH:/path/to/your/quantum-espresso/executables

Another thing that is required is setting the environmental vaiable with the path
to the directory containing pseudopotentials

.. code-block:: bash

   export ESP_PSP_PATH=/path/to/pseudo/pseudopotentials


.. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
.. _github: https:github.com
.. _lmmfixes: https://github.com/lmmentel/ase-espresso/tree/lmmfixes
.. _pip: https://pip.pypa.io/en/stable/
.. _vossjo: https://github.com/vossjo/ase-espresso
.. _wiki: https://github.com/vossjo/ase-espresso/wiki
.. _ase-espresso: https://github.com/vossjo/ase-espresso
.. _Quantum Espresso: http://www.quantum-espresso.org/
