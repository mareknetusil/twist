from fenics import *
# from cbc.common import CBCProblem
# from cbc.common import CBCSolver
from cbc.twist.problem_definitions import StaticHyperelasticity
# from cbc.twist.solution_algorithms_blocks import FunctionSpace_U, \
#     LinearElasticityTerm, homogenization_rhs, HyperelasticityTerm
from cbc.twist.material_models import LinearGeneral
from cbc.twist.function_spaces import FunctionSpace_U
from cbc.twist.equation_terms import elasticity_displacement, homogenization_rhs, \
    linear_pk_stress
from cbc.twist.material_models import LinearGeneral
import itertools


class PeriodicBoundary(SubDomain):
    def __init__(self, mesh):
        SubDomain.__init__(self)
        coors = mesh.coordinates()
        x_list = coors[:, 0]
        y_list = coors[:, 1]
        self.x_min = min(x_list)
        self.x_max = max(x_list)
        self.y_min = min(y_list)
        self.y_max = max(y_list)
        # self.x_min = 0.0
        # self.x_max = 1.0
        # self.y_min = 0.0
        # self.y_max = 1.0

    def inside(self, x, on_boundary):
        return bool((near(x[0], self.x_min) or near(x[1], self.y_min)) and
                    (not ((near(x[0], self.x_max) and near(x[1], self.y_min)) or
                          (near(x[0], self.x_max) and near(x[1], self.y_max))))
                    and on_boundary)

    def map(self, x, y):
        x_len = self.x_max - self.x_min
        y_len = self.y_max - self.y_min
        if near(x[0], self.x_max) and near(x[1], self.y_max):
            y[0] = x[0] - x_len
            y[1] = x[1] - y_len
        elif near(x[0], self.x_max):
            y[0] = x[0] - x_len
            y[1] = x[1]
        elif near(x[1], self.y_max):
            y[0] = x[0]
            y[1] = x[1] - y_len
        else:
            y[0] = 1e40
            x[0] = 1e40


class LinearHomogenization(StaticHyperelasticity):

    def __init__(self, elasticity_tensor = None):
        StaticHyperelasticity.__init__(self)

        self._material = None
        self.A = elasticity_tensor
        # self.parameters = default_parameters()
        self._correctors_chi = {}
        # self.solver = None
        self.indxs = (0, 0)
        self.dim = None
        self.Pi_functions = None
        self._volume = None

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, elasticity_tensor):
        self._A = elasticity_tensor

    def periodic_boundaries(self):
        return PeriodicBoundary(self.mesh()) if self.mesh() is not None \
            else None

    def material_model(self):
        self._material = LinearGeneral({'A': self.A})
        return self._material

    def solve(self):
        """
        Computes all corrector functions and stores them in _correctors_chi
        :return: _2-list of correctors
        """
        self.dim = self.mesh().geometry().dim()
        # self.solver = LinearHomogenizationSolver(self, self.parameters)
        for (i, j) in itertools.product(range(self.dim), range(self.dim)):
            self.indxs = (i, j)
            self._correctors_chi[(i, j)] = StaticHyperelasticity.solve(self)
        return self._correctors_chi

    def body_force_div(self):
        (m, n) = self.indxs
        A_mn = homogenization_rhs(self.A, m, n)
        return A_mn

    def volume(self):
        """
        :return: Volume of the mesh
        """
        if self._volume is None:
            V = FunctionSpace(self._mesh, 'DG', 0)
            One = project(Constant(1.0), V)
            self._volume = assemble(One*dx)
        return self._volume

    def generate_Pi_functions(self):
        """
        :return: None
        """
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
        """
        :return: Integral avg of the elasticity tensor
        """
        A_av = {}
        V = FunctionSpace(self._mesh, 'CG', 1)
        for (i,j,k,l) in itertools.product(range(2), range(2), range(2), range(2)):
            A_ijkl = project(self.A[i,j,k,l], V)
            A_av[(i,j,k,l)] = assemble(A_ijkl*dx)/self.volume()

        B = [[[[A_av[(i,j,k,l)] for l in range(2)] for k in range(2)]
              for j in range(2)] for i in range(2)]
        return as_tensor(B)

    def corrector_elasticity_tensor(self):
        """
        :return: Corrector term for the homogenized elasticity tensor
        """
        # A = self.elasticity_tensor()
        A_corr = {}
        V = FunctionSpace(self._mesh, 'CG', 1)
        for (i,j,k,l) in itertools.product(range(2), range(2), range(2), range(2)):
            m, n = indices(2)
            chi_kl = self.correctors_chi((k,l))
            P_ijkl = as_tensor(self.A[i,j,m,n]*grad(chi_kl)[m,n])
            A_corr[(i,j,k,l)] = assemble(P_ijkl*dx)/self.volume()

        B = [[[[A_corr[(i,j,k,l)] for l in range(2)] for k in range(2)]
              for j in range(2)] for i in range(2)]
        return as_tensor(B)

    def homogenized_elasticity_tensor(self):
        """
        :return: A^0
        """
        A_av = self.averaged_elasticity_tensor()
        A_corr = self.corrector_elasticity_tensor()
        i,j,k,l = indices(4)
        A0 = [[[[A_av[i,j,k,l] - A_corr[i,j,k,l] for l in range(2)] for k in range(2)]
                for j in range(2)] for i in range(2)]
        return as_tensor(A0)

    @staticmethod
    def print_elasticity_tensor(A):
        indx = [(0,0), (1,1), (0,1)]
        retval = as_matrix([[A[i[0],i[1],j[0],j[1]] for j in indx] for i in indx])
        return retval

    def correctors_omega(self, indxs=None):
        """Return \omega_ij corrector.
           For None return a list of all \omega"""

    def displacement_correction(self, u_0):
        """Return u_1"""
        Chi_mn = [[self.correctors_chi((m,n)) for n in range(2)] for m in range(2)]
        Chi = as_tensor(Chi_mn)
        m, n = indices(2)
        u_1 = as_vector([- Chi[m,n,i]*grad(u_0)[m, n] for i in range(2)])

        V = VectorFunctionSpace(self.mesh(), 'CG', 1)
        u_1_fce = project(u_1, V)
        return u_1_fce

    def __str__(self):
        """Return a short description of the problem"""
        return "Linear homogenization problem"


# class LinearHomogenizationSolver(CBCSolver):
#     """Solves the linear homogenization equation"""
#
#     def __init__(self, problem, parameters):
#         """Initialise the solver"""
#
#         # Define function spaces
#         element_degree = parameters['element_degree']
#         pbc = problem.periodic_boundaries()
#         vector = FunctionSpace_U(problem.mesh(), 'CG', element_degree, pbc)
#         print "Number of DOFs = %d" % vector.space.dim()
#
#         problem.generate_Pi_functions()
#
#         self.f_space = vector
#         self.parameters = parameters
#         self.mesh = problem.mesh()
#         self.equation = None
#         self.problem = problem
#
#     def solve(self):
#         """Solve the homogenization problem"""
#
#         # Equation
#         # a = LinearElasticityTerm(A, self.f_space.trial_displacement,
#         #                           self.f_space.test_displacement)
#         # mat = self.problem.material()
#         # P = mat.first_pk_tensor(self.f_space.trial_displacement)
#         P = linear_pk_stress(self.problem.A, self.f_space.trial_displacement)
#         a = elasticity_displacement(P, self.f_space.test_displacement)
#
#         (i, j) = self.problem.indxs
#         A_mn = homogenization_rhs(A, i, j)
#         L = elasticity_displacement(A_mn, self.f_space.test_displacement)
#
#         u = self.f_space.unknown_displacement
#         problem = LinearVariationalProblem(a, L, u)
#         solver = LinearVariationalSolver(problem)
#         self.equation = solver
#         solver.solve()
#
#         chi = self.f_space.unknown_displacement.copy(deepcopy=True)
#         return chi


def function_from_cell_function(values, subdomains):
    import numpy
    helper = numpy.asarray(subdomains.array(), dtype=numpy.int32)
    mesh = subdomains.mesh()
    V = FunctionSpace(mesh, 'DG', 0)
    dm = V.dofmap()
    for cell in cells(mesh):
        helper[dm.cell_dofs(cell.index())] = subdomains[cell]
    u = Function(V)
    u.vector()[:] = numpy.choose(helper, values)
    return u
