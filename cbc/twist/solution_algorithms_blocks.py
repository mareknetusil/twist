from dolfin import *
from nonlinear_solver import *
from cbc.twist.coordinate_system import *
from cbc.twist.kinematics import *
from cbc.common import *

def ElasticityDisplacementTerm(P, B, v, coordinate_system = None):

    if coordinate_system:
        G_raise = coordinate_system.metric_tensor('raise')
        jacobian = coordinate_system.volume_jacobian()
    else:
        G_raise = SecondOrderIdentity(v)
        jacobian = Constant(1.0)

    L = -inner(B, v)*jacobian*dx

    # The variational form corresponding to hyperelasticity
    if isinstance(P, tuple):
        P_list, subdomains_list = P

        for (index, P) in enumerate(P_list):
            new_dx = Measure('dx')[subdomains_list[index][0]]
            L += inner(P*G_raise, Grad_Cyl(v, coordinate_system))*jacobian*new_dx(subdomains_list[index][1])
    else:
        L += inner(P*G_raise, Grad_Cyl(v, coordinate_system))*jacobian*dx

    return L


def ElasticityPressureTerm(u, p, v, coordinate_system = None):
    if coordinate_system:
        g = coordinate_system.metric_tensor('raise', deformed = True)
        jacobian = coordinate_system.volume_jacobian()
    else:
        g = SecondOrderIdentity(v)
        jacobian = Constant(1.0)
    J = Jacobian(u, coordinate_system)
    F = DeformationGradient(u)

    L = -p*J*inner(g*inv(F.T), Grad_Cyl(v, coordinate_system))*jacobian*dx
    return L



def VolumeChangeTerm(u, p, q, problem, coordinate_system = None):
    material_model = problem.material_model(problem.mesh())
    jacobian = coordinate_system.volume_jacobian() if coordinate_system else Constant(1.0)

    J = Jacobian(u, coordinate_system)
    
    L = Constant(0.0)*q*jacobian*dx
    if isinstance(material_model, tuple):
        material_list, cell_function = material_model
        new_dx = Measure('dx')[cell_function]
        for (index, material) in enumerate(material_list):
            material_parameters = material_list[index].parameters
            lb = material_parameters['bulk']
            if lb <= 0.0:
                L =+ (J - 1.0)*q*jacobian*new_dx(index)
            else:
                L += (1.0/lb*p + J - 1.0)*q*jacobian*new_dx(index)
    else:
        lb = problem.material_model(mesh).parameters['bulk']
        if lb <= 0.0:
            L+= (J - 1.0)*q*jacobian*dx
        else:
            L += (1.0/lb*p + J - 1.0)*q*jacobian*dx

    return L




#TODO: Implement the curvilinear coordinates
def NeumannBoundaryTerm(neumann_conditions, neumann_boundaries, v):


    boundary = FacetFunction("size_t", v.function_space().mesh())
    boundary.set_all(len(neumann_boundaries) + 1)

    L = - inner(Constant((0,)*v.geometric_dimension()), v)*ds
    dsb = Measure('ds')[boundary]
    for (i, neumann_boundary) in enumerate(neumann_boundaries):
        compiled_boundary = CompiledSubDomain(neumann_boundary) 
        compiled_boundary.mark(boundary, i)
        L += - inner(neumann_conditions[i], v)*dsb(i)

    return L



class FunctionSpace_U():

    def __init__(self, mesh, element_type, element_degree, pbc = []):
        if pbc == []:
            self.space = VectorFunctionSpace(mesh, element_type, element_degree)
        else:
            self.space = VectorFunctionSpace(mesh, element_type, element_degree, constrained_domain = pbc)
        self._unknown = Function(self.space)
        self._test_function = TestFunction(self.space)
        self._trial_function = TrialFunction(self.space)

    def unknown(self):
        return self._unknown

    def test_function(self):
        return self._test_function

    def trial_function(self):
        return self._trial_function

    def create_dirichlet_conditions(self, problem):
        self.bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          self.space)
        return self.bcu


class FunctionSpace_UP():
    def __init__(self, mesh, element_type, element_degree, pbc = []):
        if pbc == []:
            vector = VectorFunctionSpace(mesh, element_type, element_degree)
            scalar = FunctionSpace(mesh, element_type, element_degree - 1)
        else:
            vector = VectorFunctionSpace(mesh, element_type, element_degree, constrained_domain = pbc)
            scalar = FunctionSpace(mesh, element_type, element_degree - 1, constrained_domain = pbc)
        self.space = MixedFunctionSpace([vector,scalar])

    def unknown(self):
        self.w = Function(self.space)
        return self.w

    def test_functions(self):
        (self.v, self.q) = TestFunctions(self.space)
        return (self.v, self.q)

    def trial_function(self):
        self.dw = TrialFunction(self.space)
        return self.dw

    def create_dirichlet_conditions(self, problem):
        self.bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                            problem.dirichlet_boundaries(),
                                            self.space.sub(0))
        return self.bcu
