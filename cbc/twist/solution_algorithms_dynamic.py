__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from nonlinear_solver import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient, Jacobian
from sys import exit
from numpy import array, loadtxt, linalg

def default_parameters():
    "Return default solver parameters."
    p = Parameters("solver_parameters")
    p.add("plot_solution", True)
    p.add("save_solution", False)
    p.add("store_solution_data", False)
    p.add("element_degree",2)
    p.add("problem_formulation",'displacement')
    rel = Parameters("newton_solver")
    rel.add("value", 1.0)
    rel.add("adaptive", True)
    rel.add("loading_number_of_steps", 1)
    p.add(rel)

    return p

class CG1MomentumBalanceSolver(CBCSolver):
    """Solves the dynamic balance of linear momentum using a CG1
    time-stepping scheme"""

    def __init__(self, problem, parameters):

        """Initialise the momentum balance solver"""

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time    = problem.end_time()
        info("Using time step dt = %g" % dt)

        # Define function spaces
        element_degree = parameters["element_degree"]
        #scalar = FiniteElement("CG", mesh.ufl_cell(), element_degree)
        vector = VectorElement("CG", mesh.ufl_cell(), element_degree)
        
        

        vector_space = FunctionSpace(mesh, vector)
        mixed_space = FunctionSpace(mesh, vector*vector)
        V = TestFunction(mixed_space)
        dU = TrialFunction(mixed_space)
        U = Function(mixed_space)
        U0 = Function(mixed_space)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector_space.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector_space.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector_space, file_name)
        if isinstance(v0, str):
            info("Loading initial velocity from file.")
            file_name = v0
            v0 = Function(vector_space, file_name)

        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        bcu = create_dirichlet_conditions(dirichlet_values,
                                          problem.dirichlet_boundaries(),
                                          mixed_space.sub(0))

        # Functions
        xi, eta = split(V)
        u, v = split(U)
        u_plot = Function(vector_space)

        # Project u0 and v0 into U0
        a_proj = inner(dU, V)*dx
        L_proj = inner(u0, xi)*dx + inner(v0, eta)*dx
        solve(a_proj == L_proj, U0)
        u0, v0 = split(U0)

        # Driving forces
        B  = problem.body_force()
        if B == []: B = problem.body_force_u(0.5*(u0 + u))

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector_space.mesh().geometry().dim())

        # Evaluate displacements and velocities at mid points
        u_mid = 0.5*(u0 + u)
        v_mid = 0.5*(v0 + v)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            info("Converting given density to a DOLFIN Constant.")
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u_mid)

        # Convert time step to a DOLFIN constant
        k = Constant(dt)

        # The variational form corresponding to hyperelasticity
        L = rho0*inner(v - v0, xi)*dx + k*inner(P, grad(xi))*dx \
            - k*inner(B, xi)*dx + inner(u - u0, eta)*dx \
            - k*inner(v_mid, eta)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            info("Applying Neumann boundary condition.")
            info(str(neumann_boundary))
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - k*inner(neumann_conditions[i], xi)*dsb(i)

        a = derivative(L, U, dU)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.U0 = U0
        self.U = U
        self.B = B
        self.dirichlet_values = dirichlet_values
        self.neumann_conditions = neumann_conditions

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        # Kristoffer's fix in order to sync the F and S solvers dt...
        self.t = dt

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_velocity_series = None
        #self.u_plot = u_plot
	self.uplot = plot(u,mode="displacement",title="Displacement")

        # Store parameters
        self.parameters = parameters

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            info("Solving the problem at time t = " + str(self.t))
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

        return self.U.split(True)[0]

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # Update time step
        self.dt = dt
        self.k.assign(dt)

        #problem = NonlinearVariationalProblem(self.L, self.U, self.bcu, self.a)
        #solver = NonlinearVariationalSolver(problem)
        solver = AugmentedNewtonSolver(self.L, self.U, self.a, \
                         self.bcu)
        #solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        #solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        #solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.solve()
        return self.U.split(True)

    def update(self):
        """Update problem at time t"""

        u, v = self.U.split()

        # Propagate the displacements and velocities
        self.U0.assign(self.U)

        # Plot solution
        if self.parameters["plot_solution"]:
            # Copy to a fixed function to trick Viper into not opening
            # up multiple windows
            "THIS ASSIGN DOES NOT WORK FOR SOME REASON!" #self.u_plot.assign(u)
            #plot(u, title="Displacement", mode="displacement", rescale=True)
            "This is a new ploting"
	    self.uplot.plot(u)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None: self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
            self.displacement_file << u
            self.velocity_file << v

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_velocity_series is None: self.displacement_velocity_series = TimeSeries("displacement_velocity")
            self.displacement_velocity_series.store(self.U.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_values:
            if isinstance(bc, Expression):
                bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

    def solution(self):
        "Return current solution values"
        return self.U.split(True)
