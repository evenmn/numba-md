import numpy as np
from numba import jit


class Potential:
    """ Potential class. Find the force acting on the particles
    given a potential.
    """
    def __init__(self):
        pass

    def __call__(self, r):
        raise NotImplementedError ("Class {} has no instance '__call__'."
                                   .format(self.__class__.__name__))

    @staticmethod
    def potentialEnergy(u, cutoff):
        raise NotImplementedError ("Class {} has no instance 'potentialEnergy'."
                                   .format(self.__class__.__name__))


class LennardJones(Potential):
    """ The Lennard-Jones potential. Taking the form
        U(r) = 4ε((σ/r)^12 - (σ/r)^6)

    Parameters
    ----------
    solver : obj
        class object defined by moleculardynamics.py. Takes the MDSolver
        class as argument
    cutoff : float
        cutoff distance: maximum length of the interactions. 3 by default.
    """
    def __init__(self, solver, cutoff=3):
        self.cutoff = cutoff
        self.cutoffSqrd = cutoff * cutoff
        self.checkDistance = solver.boundaries.checkDistance

        # Generate indices of upper and lower triangles
        self.numparticles = solver.numparticles
        self.numdimensions = solver.numdimensions

    def __repr__(self):
        """ Representing the potential.
        """
        return "Lennard-Jones potential"

    def __call__(self, r, return_energy=False):
        """ Lennard-Jones inter-atomic force. This is used in the
        integration loop to calculate the acceleration of particles.

        Parameters
        ----------
        r : ndarray
            spatial coordinates at some timestep

        Returns
        -------
        ndarray
            the net force acting on every particle
        float
            total potential energy
        ndarray
            current distance matrix
        """
        return self.compute_force(r, self.numparticles, self.numdimensions, self.cutoffSqrd, return_energy) #, self.checkDistance)

    @staticmethod
    @jit(nopython=True)
    def compute_force(r, numparticles, numdimensions, cutoffSqrd, return_energy): #, checkDistance):
        poteng = 0
        force = np.zeros((numparticles, numdimensions))
        for i in range(numparticles):
            for j in range(i):
                R = r[i] - r[j]
                # R = checkDistance(R)
                R_norm_sqrd = np.sum(R**2)
                if R_norm_sqrd < cutoffSqrd:
                    R6 = R_norm_sqrd**(-3)
                    R12 = R6**2
                    force_ij = 24 * (2*R12 - R6) * R / R_norm_sqrd
                    force[i] += force_ij
                    force[j] -= force_ij
                    if return_energy:
                        cutoff6 = cutoffSqrd**(-3)
                        cutoff12 = cutoff6**2
                        poteng += 4 * (R12 - R6 - cutoff12 + cutoff6)
        return force, poteng
