""" Example: Two particles moving in one dimension with open boundaries
Initial positions: ((0), (1.5))
Initial velocities: ((0), (0))
Total time: 10 ps
Time step: 0.01 ps
Potential: Lennard-Jones
Integrator: Euler-Chromer

The distance between the particles and the energy is plotted
"""

from mdsolver import MDSolver
from mdsolver.potential import LennardJones
from mdsolver.integrator import EulerChromer
from mdsolver.initpositions import Manual

# Simulate four particles in two dimensions initially separated by a distance 1.5 sigma
positions = Manual([[0.0, 0.0, 0.0], [0.0, 1.5, 0.0],
                    [1.5, 0.0, 0.0], [1.5, 1.5, 0.0],
                    [0.0, 0.0, 1.5], [0.0, 1.5, 1.5],
                    [1.5, 0.0, 1.5], [1.5, 1.5, 1.5]])
solver = MDSolver(positions=positions, T=10, dt=0.001)
solver.dump(10, "8N_3D.xyz", "x", "y", "z")
solver.thermo(10, "8N_3D.log", "step", "time", "atoms", "poteng", "kineng")
solver.run(potential=LennardJones(solver), integrator=EulerChromer(solver))
