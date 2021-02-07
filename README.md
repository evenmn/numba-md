# numba-md
Molecular dynamics solver written in object-oriented Python and accelerated by the numba JIT-compiler.

## Installation
First download the contents:
``` bash
$ git clone https://github.com/evenmn/numba-md
```
and then install the mdsolver:
``` bash
$ cd numba-md
$ pip install .
```

## Example usage
An example script could look like this:
``` python
from mdsolver import MDSolver
from mdsolver.potential import LennardJones
from mdsolver.integrator import EulerChromer
from mdsolver.initpositions import Manual

# Simulate two particles in one dimension separated by a distance 1.5 sigma
solver = MDSolver(positions=Manual([[0.0], [1.5]]), T=10, dt=0.001)
solver.dump(100, "2N_1D.xyz", "x", "vx")
solver.thermo(100, "2N_1D.log", "step", "time", "atoms", "poteng", "kineng")
solver.run(potential=LennardJones(solver), integrator=EulerChromer(solver))
```
