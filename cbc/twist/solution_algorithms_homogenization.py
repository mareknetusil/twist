from dolfin import *
#from nonlinear solver import *
from solution_algorithms_blocks import *
from cbc.common import *
from cbc.twist.solution_algorithms_static import default_parameters
#from cbc.utils import *

class LinearHomogenizationCorrectors_U(CBCSolver):
    
    def __init__(self, problem, parameters, coordinate_system = None):
    
        vector = FunctionSpace_U(problem.mesh(), 'CG', parameters['element_degree'],\
                         problem.periodic_boundaries())
        vector.create_dirichlet_conditions(problem)

        print "Number of DOFs = %d" % vector.space.dim()

        P = problem.first_pk_stress(vector.trial_displacement)

        i, j = indices(2)
        A_ij = problem.elasticity_tensor()[i, j, problem.kl()[0], problem.kl()[1]]
        A = as_tensor(A_ij, (i, j))

        L = ElasticityDisplacementTerm(P, vector.test_displacement, coordinate_system)
        L += DivergenceBodyForceTerm(A, vector.test_displacement, coordinate_system)

        self.equation = LinearVariationalSolver(LinearVariationalProblem(lhs(L), rhs(L),
                vector.unknown_displacement))
        self.mesh = problem.mesh()
        self.functionspace = vector


    def solve(self):
        self.equation.solve()
        u = self.functionspace.unknown_displacement

        plot(u, title='Corrector', mode = 'displacement')
        interactive()

        return u

