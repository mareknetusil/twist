__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from cbc.twist.nonlinear_solver import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient, Jacobian
from sys import exit
from numpy import array, loadtxt, linalg

def default_parameters():
    "Return default solver parameters."
    p = Parameters("solver_parameters")
    p.add("plot_solution", True)
    p.add("save_solution", True)
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
        CBCSolver.__init__(self)

        # Get problem parameters
        mesh = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time = problem.end_time()
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
        if not u0:
            u0 = Constant((0,)*vector_space.mesh().geometry().dim())
        if not v0:
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
        if not B: B = problem.body_force_u(0.5 * (u0 + u))

        # If no body forces are specified, assume it is 0
        if not B:
            B = Constant((0,)*vector_space.mesh().geometry().dim())

        # Evaluate displacements and velocities at mid points
        u_mid = 0.5*(u0 + u)
        v_mid = 0.5*(v0 + v)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if not rho0:
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

        #if self.parameters["plot_solution"]:
        #    interactive()

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
        #if self.parameters["plot_solution"]:
        #    # Copy to a fixed function to trick Viper into not opening
        #    # up multiple windows
        #    "THIS ASSIGN DOES NOT WORK FOR SOME REASON!" #self.u_plot.assign(u)
        #    #plot(u, title="Displacement", mode="displacement", rescale=True)
        #    "This is a new ploting"
        #    self.uplot.plot(u)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None:
                self.displacement_file = XDMFFile("displacement.xdmf")
            if self.velocity_file is None:
                self.velocity_file = XDMFFile("velocity.xdmf")
            u.rename('u', "displacement")
            v.rename('v', "velocity")
            self.displacement_file.write(u, self.t)
            self.velocity_file.write(v, self.t)
            #self.displacement_file << u
            #self.velocity_file << v

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_velocity_series is None:
                self.displacement_velocity_series = TimeSeries("displacement_velocity")
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


class LinearPoroElasticitySolver(CBCSolver):
    """Solves the dynamic balance of linear momentum using a CG1
    time-stepping scheme"""

    def __init__(self, problem, parameters):
        """Initialise the momentum balance solver"""
        CBCSolver.__init__(self)

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time    = problem.end_time()
        info("Using time step dt = %g" % dt)

        # Define function spaces
        P_0 = FiniteElement("DG", mesh.ufl_cell(), 0)
        P_1 = VectorElement("CG", mesh.ufl_cell(), 1)
        mixed_elem = MixedElement([P_1, P_1, P_0])
        vector_space = FunctionSpace(mesh, P_1)
        scalar_space = FunctionSpace(mesh, P_0)
        mixed_space = FunctionSpace(mesh, mixed_elem)

        u, z, p = TrialFunctions(mixed_space)
        v_0, v_1, q = TestFunctions(mixed_space)
        w = Function(mixed_space)

        # Get initial conditions
        u0, z0, p0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0.0,)*vector_space.mesh().geometry().dim())
        if z0 == []:
            z0 = Constant((0.0,)*vector_space.mesh().geometry().dim())
        if p0 == []:
            p0 = Constant(0.0)

        u_n = project(u0, vector_space)
        p_n = project(p0, scalar_space)

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector_space, file_name)
        if isinstance(z0, str):
            info("Loading initial velocity from file.")
            file_name = z0
            v0 = Function(vector_space, file_name)
        if isinstance(p0, str):
            info("Loading initial pressure from file.")
            file_name = p0
            p0 = Function(vector_space, file_name)

        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        bcu = create_dirichlet_conditions(dirichlet_values_u,
                                          problem.dirichlet_boundaries_u(),
                                          mixed_space.sub(0))
        bcz = create_dirichlet_conditions(dirichlet_values_z,
                                          problem.dirichlet_boundaries_z(),
                                          mixed_space.sub(1))
        bcp = create_dirichlet_conditions(dirichlet_values_p,
                                          problem.dirichlet_boundaries_p(),
                                          mixed_space.sub(2))
        bcs = [bcu, bcz, bcp]

        # Driving forces
        f_rhs  = problem.body_force()
        # If no body forces are specified, assume it is 0
        if f_rhs is None:
            f_rhs = Constant((0.0,)*vector_space.mesh().geometry().dim())

        b_rhs = problem.velocity_source()
        if b_rhs is None:
            b_rhs = Constant((0.0,)*vector_space.mesh().geometry().dim())

        g_rhs = problem.fluid_source()
        if g_rhs is None:
            g_rhs = Constant(0.0)

        # Convert time step to a DOLFIN constant
        dt = Constant(dt)

        # The variational form corresponding to hyperelasticity
        def e(u):
            return 0.5 * (grad(u) + grad(u).T)

        def a(u, v):
            mu = problem.mu
            lmbda = problem.lmbda
            return 2*inner(e(u), e(v))*dx + lmbda*div(u)*div(v)*dx

        def b(q, v):
            return q*div(v)*dx

        def J(p, q):
            delta = 0.1
            h = MaxFacetEdgeLength(mesh)
            return delta*h*jump(p)*jump(q)*dS

        LHS_u = a(u, v_0) - b(p, v_0)
        RHS_u = inner(f_rhs, v_0)*dx
        LHS_z = dt*inner(problem.k_inv()*z, v_1)*dx - dt*b(p, v_1)
        RHS_z = dt*inner(b_rhs, v_1)*dx
        LHS_p = -b(q, u) - dt*q*div(z)*dx - J(p, q)
        RHS_p = dt*g_rhs*q*dx - b(q, u_n) - J(p_n, q)

        LHS = LHS_u + LHS_z + LHS_p
        RHS = RHS_u + RHS_z + RHS_p
        L = LHS - RHS
        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        #dsb = ds[boundary]
        #for (i, neumann_boundary) in enumerate(neumann_boundaries):
        #    info("Applying Neumann boundary condition.")
        #    info(str(neumann_boundary))
        #    compiled_boundary = CompiledSubDomain(neumann_boundary)
        #    compiled_boundary.mark(boundary, i)
        #    L = L - k*inner(neumann_conditions[i], xi)*dsb(i)

        #a = derivative(L, U, dU)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.w
        self.u_n
        self.p_n
        self.L = L
        self.f_rhs = f_rhs
        self.b_rhs = b_rhs
        self.g_rhs = g_rhs
        self.bcs = bcs
        self.problem = problem
        #self.dirichlet_values = dirichlet_values
        #self.neumann_conditions = neumann_conditions

        self.mesh = mesh

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

        #if self.parameters["plot_solution"]:
        #    interactive()

        return self.w

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        #problem = NonlinearVariationalProblem(self.L, self.U, self.bcu, self.a)
        #solver = NonlinearVariationalSolver(problem)
        #solver = AugmentedNewtonSolver(self.L, self.U, self.a, \
        #                 self.bcu)
        #solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        #solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        #solver.parameters["newton_solver"]["maximum_iterations"] = 100
        #solver.solve()
        solve(self.L == 0, self.w, self.bcs)
        return self.w

    def update(self):
        """Update problem at time t"""
        u_n, _, p_n = self.w.split()
        self.u_n.assign(u_n)
        self.p_n.assign(p_n)

        # Plot solution
        #if self.parameters["plot_solution"]:
        #    # Copy to a fixed function to trick Viper into not opening
        #    # up multiple windows
        #    "THIS ASSIGN DOES NOT WORK FOR SOME REASON!" #self.u_plot.assign(u)
        #    #plot(u, title="Displacement", mode="displacement", rescale=True)
        #    "This is a new ploting"
        #    self.uplot.plot(self.u_n)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None:
                self.displacement_file = XDMFFile("displacement.xdmf")
            if self.velocity_file is None:
                self.velocity_file = XDMFFile("velocity.xdmf")
            self.displacement_file.write(u_n)
            #self.velocity_file.write(v)
            #self.displacement_file << u
            #self.velocity_file << v

        # Store solution data
        #if self.parameters["store_solution_data"]:
        #    if self.displacement_velocity_series is None: self.displacement_velocity_series = TimeSeries("displacement_velocity")
        #    self.displacement_velocity_series.store(self.U.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc_u in self.problem.dirichlet_values_u():
            if isinstance(bc_u, Expression):
                bc_u.t = self.t
        for bc_z in self.problem.dirichlet_values_z():
            if isinstance(bc_z, Expression):
                bc_z.t = self.t
        for bc_p in self.problem.dirichlet_values_p():
            if isinstance(bc_p, Expression):
                bc_p.t = self.t
        #for bc in self.neumann_conditions:
        #    bc.t = self.t
        self.f_rhs.t = self.t
        self.b_rhs.t = self.t
        self.g_rhs.t = self.t

    def solution(self):
        "Return current solution values"
        return self.w

#class NonLinearPoroElasticitySolver(CBCSolver):
#    """Solves the dynamic balance of linear momentum using a CG1
#    time-stepping scheme"""
#
#    def __init__(self, problem, parameters):
#
#        """Initialise the momentum balance solver"""
#
#        # Get problem parameters
#        mesh        = problem.mesh()
#        dt, t_range = timestep_range_cfl(problem, mesh)
#        end_time    = problem.end_time()
#        info("Using time step dt = %g" % dt)
#
#        # Define function spaces
#        element_degree = parameters["element_degree"]
#        #scalar = FiniteElement("CG", mesh.ufl_cell(), element_degree)
#        vector = VectorElement("CG", mesh.ufl_cell(), element_degree)
#
#
#
#        vector_space = FunctionSpace(mesh, vector)
#        mixed_space = FunctionSpace(mesh, vector*vector)
#        V = TestFunction(mixed_space)
#        dU = TrialFunction(mixed_space)
#        U = Function(mixed_space)
#        U0 = Function(mixed_space)
#
#        # Get initial conditions
#        u0, v0 = problem.initial_conditions()
#
#        # If no initial conditions are specified, assume they are 0
#        if u0 == []:
#            u0 = Constant((0,)*vector_space.mesh().geometry().dim())
#        if v0 == []:
#            v0 = Constant((0,)*vector_space.mesh().geometry().dim())
#
#        # If either are text strings, assume those are file names and
#        # load conditions from those files
#        if isinstance(u0, str):
#            info("Loading initial displacement from file.")
#            file_name = u0
#            u0 = Function(vector_space, file_name)
#        if isinstance(v0, str):
#            info("Loading initial velocity from file.")
#            file_name = v0
#            v0 = Function(vector_space, file_name)
#
#        # Create boundary conditions
#        dirichlet_values = problem.dirichlet_values()
#        bcu = create_dirichlet_conditions(dirichlet_values,
#                                          problem.dirichlet_boundaries(),
#                                          mixed_space.sub(0))
#
#        # Functions
#        xi, eta = split(V)
#        u, v = split(U)
#        u_plot = Function(vector_space)
#
#        # Project u0 and v0 into U0
#        a_proj = inner(dU, V)*dx
#        L_proj = inner(u0, xi)*dx + inner(v0, eta)*dx
#        solve(a_proj == L_proj, U0)
#        u0, v0 = split(U0)
#
#        # Driving forces
#        B  = problem.body_force()
#        if B == []: B = problem.body_force_u(0.5*(u0 + u))
#
#        # If no body forces are specified, assume it is 0
#        if B == []:
#            B = Constant((0,)*vector_space.mesh().geometry().dim())
#
#        # Evaluate displacements and velocities at mid points
#        u_mid = 0.5*(u0 + u)
#        v_mid = 0.5*(v0 + v)
#
#        # Get reference density
#        rho0 = problem.reference_density()
#
#        # If no reference density is specified, assume it is 1.0
#        if rho0 == []:
#            rho0 = Constant(1.0)
#
#        density_type = str(rho0.__class__)
#        if not ("dolfin" in density_type):
#            info("Converting given density to a DOLFIN Constant.")
#            rho0 = Constant(rho0)
#
#        # Piola-Kirchhoff stress tensor based on the material model
#        P = problem.first_pk_stress(u_mid)
#
#        # Convert time step to a DOLFIN constant
#        k = Constant(dt)
#
#        # The variational form corresponding to hyperelasticity
#        L = rho0*inner(v - v0, xi)*dx + k*inner(P, grad(xi))*dx \
#            - k*inner(B, xi)*dx + inner(u - u0, eta)*dx \
#            - k*inner(v_mid, eta)*dx
#
#        # Add contributions to the form from the Neumann boundary
#        # conditions
#
#        # Get Neumann boundary conditions on the stress
#        neumann_conditions = problem.neumann_conditions()
#        neumann_boundaries = problem.neumann_boundaries()
#
#        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
#        boundary.set_all(len(neumann_boundaries) + 1)
#
#        dsb = ds[boundary]
#        for (i, neumann_boundary) in enumerate(neumann_boundaries):
#            info("Applying Neumann boundary condition.")
#            info(str(neumann_boundary))
#            compiled_boundary = CompiledSubDomain(neumann_boundary)
#            compiled_boundary.mark(boundary, i)
#            L = L - k*inner(neumann_conditions[i], xi)*dsb(i)
#
#        a = derivative(L, U, dU)
#
#        # Store variables needed for time-stepping
#        self.dt = dt
#        self.k = k
#        self.t_range = t_range
#        self.end_time = end_time
#        self.a = a
#        self.L = L
#        self.bcu = bcu
#        self.U0 = U0
#        self.U = U
#        self.B = B
#        self.dirichlet_values = dirichlet_values
#        self.neumann_conditions = neumann_conditions
#
#        # FIXME: Figure out why I am needed
#        self.mesh = mesh
#        # Kristoffer's fix in order to sync the F and S solvers dt...
#        self.t = dt
#
#        # Empty file handlers / time series
#        self.displacement_file = None
#        self.velocity_file = None
#        self.displacement_velocity_series = None
#        #self.u_plot = u_plot
#	self.uplot = plot(u,mode="displacement",title="Displacement")
#
#        # Store parameters
#        self.parameters = parameters
#
#    def solve(self):
#        """Solve the mechanics problem and return the computed
#        displacement field"""
#
#        # Time loop
#        for t in self.t_range:
#            info("Solving the problem at time t = " + str(self.t))
#            self.step(self.dt)
#            self.update()
#
#        if self.parameters["plot_solution"]:
#            interactive()
#
#        return self.U.split(True)[0]
#
#    def step(self, dt):
#        """Setup and solve the problem at the current time step"""
#
#        # Update time step
#        self.dt = dt
#        self.k.assign(dt)
#
#        #problem = NonlinearVariationalProblem(self.L, self.U, self.bcu, self.a)
#        #solver = NonlinearVariationalSolver(problem)
#        solver = AugmentedNewtonSolver(self.L, self.U, self.a, \
#                         self.bcu)
#        #solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
#        #solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
#        #solver.parameters["newton_solver"]["maximum_iterations"] = 100
#        solver.solve()
#        return self.U.split(True)
#
#    def update(self):
#        """Update problem at time t"""
#
#        u, v = self.U.split()
#
#        # Propagate the displacements and velocities
#        self.U0.assign(self.U)
#
#        # Plot solution
#        if self.parameters["plot_solution"]:
#            # Copy to a fixed function to trick Viper into not opening
#            # up multiple windows
#            "THIS ASSIGN DOES NOT WORK FOR SOME REASON!" #self.u_plot.assign(u)
#            #plot(u, title="Displacement", mode="displacement", rescale=True)
#            "This is a new ploting"
#	    self.uplot.plot(u)
#
#        # Store solution (for plotting)
#        if self.parameters["save_solution"]:
#            if self.displacement_file is None: self.displacement_file = File("displacement.pvd")
#            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
#            self.displacement_file << u
#            self.velocity_file << v
#
#        # Store solution data
#        if self.parameters["store_solution_data"]:
#            if self.displacement_velocity_series is None: self.displacement_velocity_series = TimeSeries("displacement_velocity")
#            self.displacement_velocity_series.store(self.U.vector(), self.t)
#
#        # Move to next time step
#        self.t = self.t + self.dt
#
#        # Inform time-dependent functions of new time
#        for bc in self.dirichlet_values:
#            if isinstance(bc, Expression):
#                bc.t = self.t
#        for bc in self.neumann_conditions:
#            bc.t = self.t
#        self.B.t = self.t
#
#    def solution(self):
#        "Return current solution values"
#        return self.U.split(True)
