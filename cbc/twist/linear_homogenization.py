from dolfin import *
from cbc.common import CBCProblem
from cbc.common import CBCSolver
from cbc.twist.solution_algorithms_blocks import FunctionSpace_U, \
    LinearElasticityTerm, homogenization_rhs, HyperelasticityTerm
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
        self._volume = None

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

    def volume(self):
        if self._volume is None:
            V = FunctionSpace(self._mesh, 'DG', 0)
            One = project(Constant(1.0), V)
            self._volume = assemble(One*dx)
        return self._volume


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
        return self._correctors_chi[indxs]

    def averaged_elasticity_tensor(self):
        A = self.elasticity_tensor()
        A_av = {}
        V = FunctionSpace(self._mesh, 'CG', 1)
        for (i,j,k,l) in itertools.product(range(2), range(2), range(2), range(2)):
            A_ijkl = project(A[i,j,k,l], V)
            A_av[(i,j,k,l)] = assemble(A_ijkl*dx)/self.volume()

        B = [[[[A_av[(i,j,k,l)] for l in range(2)] for k in range(2)]
              for j in range(2)] for i in range(2)]
        return as_tensor(B)

    def corrector_elasticity_tensor(self):
        A = self.elasticity_tensor()
        A_corr = {}
        V = FunctionSpace(self._mesh, 'CG', 1)
        for (i,j,k,l) in itertools.product(range(2), range(2), range(2), range(2)):
            m, n = indices(2)
            chi_kl = self.correctors_chi((k,l))
            P_ijkl = as_tensor(A[i,j,m,n]*grad(chi_kl)[m,n])
            A_corr[(i,j,k,l)] = assemble(P_ijkl*dx)/self.volume()

        B = [[[[A_corr[(i,j,k,l)] for l in range(2)] for k in range(2)]
              for j in range(2)] for i in range(2)]
        return as_tensor(B)

    def homogenized_elasticity_tensor(self):
        A_av = self.averaged_elasticity_tensor()
        A_corr = self.corrector_elasticity_tensor()
        i,j,k,l = indices(4)
        A0_ijkl = A_av[i,j,k,l] - A_corr[i,j,k,l]
        A0 = as_tensor(A0_ijkl, (i,j,k,l))
        return A0


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
        a = LinearElasticityTerm(A, self.f_space.trial_displacement,
                                  self.f_space.test_displacement)

        (i, j) = self.problem.indxs
        A_mn = homogenization_rhs(A, i, j)
        L = HyperelasticityTerm(A_mn, self.f_space.test_displacement)

        u = self.f_space.unknown_displacement
        problem = LinearVariationalProblem(a, L, u)
        solver = LinearVariationalSolver(problem)
        self.equation = solver
        solver.solve()

        """Solve the homogenization problem"""

        chi = self.f_space.unknown_displacement.copy(deepcopy=True)
        return chi
