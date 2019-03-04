__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__ = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

import fenics
import cbc.twist.nonlinear_solver as solvers
import cbc.twist.equation_terms as terms
import cbc.twist.function_spaces as spaces
from cbc.common import *
from cbc.common.utils import *
from sys import exit
from numpy import array, loadtxt, linalg

solvers.parameters['form_compiler']['representation'] = 'uflacs'
solvers.parameters['form_compiler']['optimize'] = True
solvers.parameters['form_compiler']['quadrature_degree'] = 4


# def default_parameters():
#     """Return default solver parameters."""
#     p = fenics.Parameters("solver_parameters")
#     p.add("plot_solution", True)
#     p.add("save_solution", True)
#     p.add("store_solution_data", False)
#     p.add("element_degree", 2)
#     p.add("problem_formulation", 'displacement')
#     new = fenics.Parameters("newton_solver")
#     new.add("value", 1.0)
#     new.add("adaptive", True)
#     new.add("loading_number_of_steps", 1)
#     p.add(new)
#
#     return p


class StaticMomentumBalanceSolver_U(CBCSolver):
    """Solves the static balance of linear momentum"""

    def __init__(self, problem, parameters):
        """Initialise the static momentum balance solver"""
        CBCSolver.__init__(self)

        # Define function spaces
        element_degree = parameters['element_degree']
        pbc = problem.periodic_boundaries()

        vector = spaces.FunctionSpace_U(problem.mesh(), 'CG', element_degree,
                                        pbc)
        vector.create_dirichlet_conditions(problem)

        # Print DOFs
        print("Number of DOFs = %d" % vector.space.dim())

        # Driving forces
        # B = problem.body_force()
        # B_div = problem.body_force_div()
        P = problem.first_pk_stress(vector.unknown_displacement)

        # If no body forces are specified, assume it is 0
        # if not B:
        #     B = Constant((0,) * vector.test_displacement.geometric_dimension())

        self.theta = Constant(1.0)

        L1 = terms.elasticity_displacement(P, vector.test_displacement)
        # L2 = terms.body_force(self.theta * B, vector.test_displacement)
        B = problem.body_force()
        if B:
            L1 -= self.theta * terms.body_force(B, vector.test_displacement)

        B_div = problem.body_force_div()
        if B_div:
            L1 -= self.theta * terms.body_force_div(B_div, vector.test_displacement)

        # Add contributions to the form from the Neumann boundary
        # conditions
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if not neumann_conditions:
            neumann_conditions = [
                Constant((0,) * vector.test_displacement.geometric_dimension())]

        neumann_conditions = [self.theta * g for g in neumann_conditions]

        L3 = terms.neumann_condition(neumann_conditions,
                                     problem.neumann_boundaries(),
                                     vector.test_displacement, problem.mesh())

        # L = L1 - L2 - L3
        L = L1 - L3

        a = fenics.derivative(L, vector.unknown_displacement,
                              vector.trial_displacement)

        solver = solvers.AugmentedNewtonSolver(L, vector.unknown_displacement,
                                               a, vector.bcu,
                                               load_increment=self.theta)

        newton_parameters = parameters['newton_solver']
        solver.parameters['newton_solver'].update(newton_parameters)

        self.functionSpace = vector

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = problem.mesh()
        self.equation = solver
        self.a = a

    def solve(self):
        """Solve the mechanics problem and return the computed
      displacement field"""

        # Solve problem
        self.equation.solve()
        u = self.functionSpace.unknown_displacement

        # Plot solution
        if self.parameters["plot_solution"]:
            fenics.plot(u, title="Displacement", mode="displacement", axes=True,
                 rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            dir = self.parameters["output_dir"]
            displacement_file \
                = fenics.XDMFFile("{}/displacement.xdmf".format(dir))
            u.rename('u', "displacement")
            displacement_file.write(u)
            # displacement_file << u

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series \
                = fenics.TimeSeries("{}/displacement".format(dir))
            displacement_series.store(u.vector(), 0.0)

        return u
