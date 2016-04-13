============
ase-espresso
============

[ase-espresso](https://github.com/vossjo/ase-espresso) provides a Python interface compatible with
[Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/) for [Quantum Espresso](http://www.quantum-espresso.org/).

Installation
============

The recommended installation method is with [pip]. The current
version can be installed directly from [github]:

```bash
pip install https://github.com/lmmentel/ase-espresso/archive/master.zip
```
or cloned first
```bash
git clone https://github.com/lmmentel/ase-espresso.git
```
and installed via
```bash
pip install ./ase-espresso
```

[github]: https://github.com/lmmentel/ase-espresso
[pip]: https://pip.pypa.io/en/stable/
[vossjo]: https://github.com/vossjo/ase-espresso
[wiki]: https://github.com/vossjo/ase-espresso/wiki

Documentation
=============

Documentation of the package can be generated with [sphinx](http://www.sphinx-doc.org/en/stable/)
by going to the ``docs`` directory and typing:

```bash
make html
```

The built documentation can be viewed in a any browser
```bash
firefox build/html/index.html
```
