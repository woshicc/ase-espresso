# ase-espresso

[ase-espresso](https://github.com/lmmentel/ase-espresso) provides a Python interface compatible with
[Atomic Simulation Environment (ASE)][ase] for manging calculations with the [Quantum Espresso] code.

This is a fork from [vossjo][vossjo-ae] that offers a lot of improvements over the original version,
the most important ones include:

- the files were restructured into a python package
- a `setup.py` file was introduced to allow installation through [pip] or [setuptools]
- configuration for the documentation is provided through [sphinx] and a lot of docstrings were updated
- the `site.cfg` is obsolete now, and no additional configuration is required, the functionality is replaced
  by a new `SiteConfig` class that dynamically gathers information about the execution environment 
- the old `espresso` class is now split into two: `Espresso` preserving the standard functionality and
  `iEspresso` responsible for dynamic/interactive jobs with a custom version of pw.x
- changes were made to establish python 3.x compatibility
- the `Espresso` class were restructured according to [ase] guidelines regarding calculator objects to
  support full compatibility with [ase]
- most of the system call are now handled by [pexpect] and [subprocess] instead of the old `os.system`,
  `os.popen()`, `os.popen2()`, `os.popen3()`
- tests were added
- code style and readability were improved


# Installation

## Dependencies

- [Atomic Simulation Environment (ASE)][ase] [version 3.11.0](https://wiki.fysik.dtu.dk/ase/releasenotes.html#releasenotes) 
- [numpy]
- [pexpect]
- [future]
- [path.py]
- [python-hostlist]

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

[ase]: https://wiki.fysik.dtu.dk/ase/index.html
[future]: http://python-future.org/
[github]: https://github.com/lmmentel/ase-espresso
[python-hostlist]: https://www.nsc.liu.se/~kent/python-hostlist/
[numpy]: http://www.numpy.org/
[path.py]: https://github.com/jaraco/path.py
[pip]: https://pip.pypa.io/en/stable/
[pexpect]: https://pexpect.readthedocs.io/en/stable
[setuptools]: https://pypi.python.org/pypi/setuptools
[sphinx]: http://www.sphinx-doc.org/en/stable/
[subprocess]: https://docs.python.org/2/library/subprocess.html
[vossjo-ae]: https://github.com/vossjo/ase-espresso
[Quantum Espresso]: http://www.quantum-espresso.org/
[wiki]: https://github.com/vossjo/ase-espresso/wiki

## Documentation

Documentation of the package can be generated using [sphinx]
by going to the ``docs`` directory and typing:

```bash
make html
```

The built documentation can be viewed in a any browser
```bash
firefox build/html/index.html
```
