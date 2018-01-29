from dolfin import *
from cbc.common import CBCProblem
from cbc.common import CBCSolver
from cbc.twist.solution_algorithms_blocks import FunctionSpace_U, \
    LinearElasticityTerm
import itertools


def default_parameters():
    """Return default parameters for linear homogenization"""
    p = Parameters("solver_parameters")
    p.add("plot_solution", True)
    p.add("save_solution", False)
    p.add("store_solution_data", False)
    p.add("element_degree", 2)

    return p


class LinearHomogenization(CBCProblem):

    def __init__(self):
        CBCProblem.__init__(self)

        self.parameters = default_parameters()
        self._correctors_chi = {}
        self.solver = None
        self.indxs = (0, 0)
        self.dim = None
        self.Pi_functions = None

    def solve(self):
        self.dim = self.mesh().geometry().dim()
        self.solver = LinearHomogenizationSolver(self, self.parameters)
        for (i, j) in itertools.product(range(self.dim), range(self.dim)):
            self.indxs = (i, j)
            self._correctors_chi[(i, j)] = self.solver.solve()
        return self._correctors_chi

    def elasticity_tensor(self):
        """Return the elasticity (tangent) tensor.
           IMPLEMENTED BY A USER"""

    def periodic_boundaries(self):
        """Return the periodic boundary conditions.
           IMPLEMENTED BY A USER"""

    def generate_Pi_functions(self):
        self.Pi_functions = []
        for (i, j) in itertools.product(range(self.dim), range(self.dim)):
            val = ["0.0", ] * self.dim
            val[i] = "x[j]"
            Pi_ij = Expression(val, j=j, degree=1)
            self.Pi_functions.append(Pi_ij)

    def correctors_chi(self, indxs=None):
        """Return \chi_ij corrector.
           For None return a list of all \chi"""
        if self._correctors_chi is None:
            self.solve()

        if indxs is None:
            return self._correctors_chi
        return self._correctors_chi[(indxs[0], indxs[1])]

    def correctors_omega(self, indxs=None):
        """Return \omega_ij corrector.
           For None return a list of all \omega"""

    def displacement_correction(self):
        """Return u_1"""

    def __str__(self):
        """Return a short description of the problem"""
        return "Linear homogenization problem"


class LinearHomogenizationSolver(CBCSolver):
    """Solves the linear homogenization equation"""

    def __init__(self, problem, parameters):
        """Initialise the solver"""

        # Define function spaces
        element_degree = parameters['element_degree']
        pbc = problem.periodic_boundaries()
        vector = FunctionSpace_U(problem.mesh(), 'CG', element_degree, pbc)
        print "Number of DOFs = %d" % vector.space.dim()

        problem.generate_Pi_functions()

        self.f_space = vector
        self.parameters = parameters
        self.mesh = problem.mesh()
        self.equation = None
        self.problem = problem

    def solve(self):

        # Equation
        A = self.problem.elasticity_tensor()
        a = LinearElasticityTerm(A, self.f_space.unknown_displacement,
                                 self.f_space.test_displacement)
        (i, j) = self.problem.indxs
        Pi = self.problem.Pi_functions[2*i + j]
        Pi_function = project(Pi, self.f_space.space, mesh=self.problem.mesh())
        L = LinearElasticityTerm(A, Pi_function,
                                 self.f_space.test_displacement)

        problem = LinearVariationalProblem(a, L, self.f_space.unknown_displacement)
        solver = LinearVariationalSolver(problem)
        self.equation = solver

        """Solve the homogenization problem"""
        # TODO:Implement
        self.equation.solve()
        chi = self.functionspace.unknown_displacement
        return chi
