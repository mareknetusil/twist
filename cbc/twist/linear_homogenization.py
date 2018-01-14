from dolfin import *
from cbc.common import CBCProblem
from cbc.twist.solution_algorithms_blocks import FunctionSpace_U,
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
        self.indxs = (0,0)
        self.dim = self.mesh().geometry().dim()

    def solve(self):
        self.solver = LinearHomogenizationSolver(self, self.parameters)
        for (i,j) in itertools.product(range(dim), range(dim)):
            self.indxs = (i,j)
            self._correctors_chi[(i,j)] = self.solver.solve()
        return self._correctors_chi

    def elasticity_tensor(self):
        """Return the elasticity (tangent) tensor.
           IMPLEMENTED BY A USER"""

    def periodic_boundaries(self):
        """Return the periodic boundary conditions.
           IMPLEMENTED BY A USER"""

    def Pi_functions(self, dim):
        val = ("0.0",)*dim
        val[self.indxs[0]] = "x[j]"
        return Expression(val,j=self.indxs[1],degree=1)

    def correctors_chi(self, indxs = None):
        """Return \chi_ij corrector.
           For None return a list of all \chi"""
        if self._correctors_chi is None:
            self.solve()

        if indxs is None:
            return self._correctors_chi
        return self._correctors_chi[(indxs[0],indxs[1])]

    def correctors_omega(self, indxs = None):
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

        #Define function spaces
        element_degree = parameters['element_degree']
        pbc = problem.periodic_boundaries()

        vector = FunctionSpace_U(problem.mesh(), 'CG', element_degree, pbc)
        print "Number of DOFs = %d" % vector.space.dim()

        #Equation
        A = problem.elasticity_tensor()
        L1 = LinearElasticityTerm(A, vector.unknown_displacement,
               vector.test_displacement)
        L2 = LinearElasticityTerm(A, problem.Pi_functions(),
                vector.test_displacement)

        #TODO:RHS and Pi operator

        #TODO:Linear variational solver

        self.function_space = vector
        self.parameters = parameters
        self.mesh = problem.mesh()
        self.equation = solver

    def solve(self, indxs):
        """Solve the homogenization problem"""
        #TODO:Implement
        self.equation.solve()
        chi = self.functionspace.unknown_displacement
        return chi
