__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from nonlinear_solver import *
from solution_algorithms_blocks import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient, Jacobian, Grad_Cyl
from cbc.twist.coordinate_system import *
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
    new = Parameters("newton_solver")
    new.add("value", 1.0)
    new.add("adaptive", True)
    new.add("loading_number_of_steps", 1)
    p.add(new)

    return p

class StaticMomentumBalanceSolver_U(CBCSolver):
    "Solves the static balance of linear momentum"

    def __init__(self, problem, parameters, coordinate_system = CartesianSystem()):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        element_degree = parameters['element_degree']
        pbc = problem.periodic_boundaries()

        vector = FunctionSpace_U(mesh, 'CG', element_degree, pbc)
        vector.create_dirichlet_conditions(problem)

        # Print DOFs
        print "Number of DOFs = %d" % vector.space.dim()

        # Define fields
        # Test and trial functions
        v = vector.test_function()
        u = vector.unknown()
        du = vector.trial_function()

        coordinate_system.set_mesh(mesh)
        coordinate_system.set_displacement(u)

        # Driving forces
        B = problem.body_force()
        P  = problem.first_pk_stress(u, coordinate_system)

        
        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*mesh.geometry().dim())
        
        self.theta = Constant(1.0)


        L1 = ElasticityDisplacementTerm(P, self.theta*B, v, coordinate_system)

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = [Constant((0,)*mesh.geometry().dim())]

        neumann_boundaries = problem.neumann_boundaries()
        neumann_conditions = [self.theta*g for g in neumann_conditions]

        L2 = NeumannBoundaryTerm(neumann_conditions, neumann_boundaries, v, coordinate_system)

        L = L1 + L2

        a = derivative(L, u, du)

        solver = AugmentedNewtonSolver(L, u, a, vector.bcu,\
                                         load_increment = self.theta)

        newton_parameters = parameters['newton_solver']
        solver.parameters['newton_solver'].update(newton_parameters)

        self.functionspace = vector

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()
        u = self.functionspace.u


        # Plot solution
        if self.parameters["plot_solution"]:
            plot(u, title="Displacement", mode="displacement", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.xdmf")
            displacement_file << u

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            displacement_series.store(u.vector(), 0.0)

        return u

class StaticMomentumBalanceSolver_UP(CBCSolver):
    "Solves the static balance of linear momentum"

    parameters['form_compiler']['representation'] = 'uflacs'
    parameters['form_compiler']['optimize'] = True
    parameters['form_compiler']['quadrature_degree'] = 4

    def __init__(self, problem, parameters, coordinate_system = CartesianSystem()):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        element_degree = parameters["element_degree"]

        
        pbc = problem.periodic_boundaries()
        mixed_space = FunctionSpace_UP(mesh, 'CG', element_degree, pbc)
        mixed_space.create_dirichlet_conditions(problem)

        # Print DOFs
        print "Number of DOFs = %d" % mixed_space.space.dim()

        # Create boundary conditions

        # Define fields
        # Test and trial functions
        (v,q) = mixed_space.test_functions()
        w = mixed_space.unknown()
        (u,p) = split(w)
        dw = mixed_space.trial_function()

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*mesh.geometry().dim())

        # First Piola-Kirchhoff stress tensor based on the material
        # model

        coordinate_system.set_mesh(mesh)
        coordinate_system.set_displacement(u)

        self.theta = Constant(1.0)
        P  = problem.first_pk_stress(u, coordinate_system)

        L = ElasticityPressureTerm(u, p, v, coordinate_system)
        L += ElasticityDisplacementTerm(P, self.theta*B, v, coordinate_system)
        L += VolumeChangeTerm(u, p, q, problem, coordinate_system)


        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = [Constant((0,)*mesh.geometry().dim())]

        neumann_boundaries = problem.neumann_boundaries()
        neumann_conditions = [self.theta*g for g in neumann_conditions]

        L += NeumannBoundaryTerm(neumann_conditions, neumann_boundaries, v, coordinate_system)

        a = derivative(L, w, dw)

        solver = AugmentedNewtonSolver(L, w, a, mixed_space.bcu,\
                                         load_increment = self.theta)
        newton_parameters = parameters['newton_solver']
        solver.parameters['newton_solver'].update(newton_parameters)


        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver
        (u, p) = w.split()
        self.u = u
        self.p = p

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()


        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            plot(self.p, title="Pressure", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.xdmf")
            pressure_file = File("pressure.xdmf")
            displacement_file << self.u
            pressure_file << self.p

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            pressure_series = TimeSeries("pressure")
            displacement_series.store(self.u.vector(), 0.0)
            pressure_series.store(self.p.vector(),0.0)

        return self.u, self.p


