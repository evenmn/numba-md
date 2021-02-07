import time
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MDSolver:
    """ Initialize the MDSolver class. This includes defining the
    time scales, initialize positions and velocities, and define
    matplotlib fixes.

    Parameters
    ----------
    positions : obj
        class object defined by initpositions.py. Face-centered cube
        with length 3 and 4 particles as default.
    velocity : obj
        class object defined by initvelocities.py. No velocity as default.
    boundaries : obj
        class object defined by boundaryconditions.py. Open boundaries
        as default.
    T : float
        total time
    dt : float
        time step
    """

    from .initpositions import FCC
    from .initvelocities import Zero
    from .boundaryconditions import Open

    def __init__(self, positions=FCC(cells=1, lenbulk=3),
                       velocities=Zero(),
                       boundaries=Open(),
                       T=5,
                       dt=0.01):

        self.boundaries = boundaries

        # Define time scale and number of steps
        self.T = T
        self.dt = dt
        self.N = int(T/dt)
        self.time = np.linspace(0, T, self.N)

        # Initialize positions
        r0 = positions()
        self.numparticles = len(r0)
        self.numdimensions = len(r0[0])
        #self.r = np.zeros((self.N+1, self.numparticles, self.numdimensions))
        self.r = np.asarray(r0)

        # Initialize velocities
        #self.v = np.zeros(self.r.shape)
        self.v = velocities(self.numparticles, self.numdimensions)

        # print to terminal
        self.print_to_terminal()

        self.compute_poteng = False

    def __repr__(self):
        return """MDSolver is the heart of the molecular dynamics code.
                  It sets up the solver and distribute tasks to other
                  classes. """

    def dump(self, freq, file, *quantities):
        """Dump atom-quantities to file
        """
        from .dump import Dump
        self.dumpobj = Dump(freq, file, quantities)

    def thermo(self, freq, file, *quantities):
        """Print thermo-quantities to file
        """
        from .thermo import Thermo
        if "poteng" in quantities:
            self.compute_poteng = True
        self.thermoobj = Thermo(freq, file, quantities)

    def print_to_terminal(self):
        """ Print information to terminal
        """
        print("\n\n" + 14 * "=", " SYSTEM INFORMATION ", 14 * "=")
        print("Number of particles:  ", self.numparticles)
        print("Number of dimensions: ", self.numdimensions)
        print("Boundary conditions:  ", self.boundaries)
        print("Total time:           ", self.T, "\tps")
        print("Timestep:             ", self.dt, "\tps")
        print(50 * "=" + "\n\n")

    @staticmethod
    def print_simulation(potential, integrator):
        """ Print information to terminal when starting a simulation

        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        """
        print("\n\n" + 12 * "=", " SIMULATION INFORMATION ", 12 * "=")
        print("Potential:            ", potential)
        print("Integrator:           ", integrator)
        print(50 * "=" + "\n\n")

    def run(self, potential, integrator, dumpfile=None):
        """ Integration loop. Computes the time-development of position and
        velocity using a given integrator and inter-atomic potential.

        Parameters
        ----------
        potential : obj
            object defining the inter-atomic potential
        integrator : obj
            object defining the integrator
        poteng : bool or int
            boolean saying whether or not the potential
            energy should be calculated and stored.
        distance : bool or int
            boolean saying whether or not the distance matrix should be stored.
        dumpfile : str
            filename that all the positions should be dumped to. If not
            specified, positions are not dumped.
        """
        self.potential = potential

        # Print information
        self.print_simulation(potential, integrator)

        # Compute initial acceleration, potential energy and distance matrix
        a, self.u = potential(self.r, return_energy=self.compute_poteng)

        # Integration loop
        start = time.time()
        for t in range(self.N):   # Integration loop
            self.t = t
            self.r, self.v, a, self.u = integrator(self.r, self.v, a)

            self.dumpobj(self)
            self.thermoobj(self)
        end = time.time()
        print("Elapsed time: ", end-start)
