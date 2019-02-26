from dolfin import *
from cbc.twist.nonlinear_solver import *
from cbc.twist.kinematics import *
from cbc.common import *

def ElasticityDisplacementTerm(P, B, v, coordinate_system = None):
    """
    Basis for the dislacement formulation: - Div(P) = B
    Returns: P:Grad(v)*dx - inner(B,v)*dx
    """
    L = -inner(B, v)*dx

    # The variational form corresponding to hyperelasticity
    if isinstance(P, tuple):
        P_list, subdomains_list = P

        for (index, P) in enumerate(P_list):
            new_dx = Measure('dx')(subdomain_data = subdomains_list[index][0])
            L += inner(P, Grad(v))*new_dx(subdomains_list[index][1])
    else:
        L += inner(P, Grad(v))*dx

    return L


def ElasticityPressureTerm(u, p, v, coordinate_system = None):
    """
    The nonlinear pressure term: pJF^(-T)
    Returns: -p*j*inner(F^(-T),Grad(v))*dx
    """
    J = Jacobian(u)
    F = DeformationGradient(u)

    L = -p*J*inner(g*inv(F.T), Grad(v))*dx
    return L


def VolumeChangeTerm(u, p, q, problem, coordinate_system = None):
    """
    The equation of volume change. Compressible material for bulk modulus
    positive and incompressible otherwise.
    Returns: (1/lb*p + J - 1)*q*dx for compressible, (J - 1)*q*dx for incompressible
    """
    material_model = problem.material_model()
    J = Jacobian(u)

    L = Constant(0.0)*q*dx
    if isinstance(material_model, tuple):
        material_list, cell_function = material_model
        new_dx = Measure('dx')[cell_function]
        for (index, material) in enumerate(material_list):
            material_parameters = material_list[index].parameters
            lb = material_parameters['bulk']
            if lb <= 0.0:
                L =+ (J - 1.0)*q*new_dx(index)
            else:
                L += (1.0/lb*p + J - 1.0)*q*new_dx(index)
    else:
        lb = problem.material_model().parameters['bulk']
        if lb <= 0.0:
            L+= (J - 1.0)*q*dx
        else:
            L += (1.0/lb*p + J - 1.0)*q*dx
    return L


#TODO: Get rid of the mesh argument
def NeumannBoundaryTerm(neumann_conditions, neumann_boundaries, v, mesh):
    """
    Neumann boundary condition: dU/dN = g
    Returns: - inner(g,v)*dS
    """

    boundary = FacetFunction("size_t", mesh)
    boundary.set_all(len(neumann_boundaries) + 1)

    L = - inner(Constant((0,)*v.geometric_dimension()), v)*ds
    dsb = Measure('ds')(subdomain_data = boundary)
    for (i, neumann_boundary) in enumerate(neumann_boundaries):
        compiled_boundary = CompiledSubDomain(neumann_boundary)
        compiled_boundary.mark(boundary, i)
        L += - inner(neumann_conditions[i], v)*dsb(i)

    return L


class FunctionSpace_U():
    """
    Discrete function space for the displacement U
    """
    def __init__(self, mesh, element_type, element_degree, pbc = []):
        if pbc == []:
            self.space = VectorFunctionSpace(mesh, element_type, element_degree)
        else:
            self.space = VectorFunctionSpace(mesh, element_type, element_degree, constrained_domain = pbc)
        self._unknown_displacement = Function(self.space)
        self._test_displacement = TestFunction(self.space)
        self._trial_displacement = TrialFunction(self.space)

    @property
    def unknown_displacement(self):
        return self._unknown_displacement
    @property
    def test_displacement(self):
        return self._test_displacement
    @property
    def trial_displacement(self):
        return self._trial_displacement

    def create_dirichlet_conditions(self, problem):
        self.bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          self.space)
        return self.bcu


class FunctionSpace_UP():
    """
    Discrete space for the (U,P)-mixed formulation
    """
    def __init__(self, mesh, element_type, element_degree, pbc = []):
        if pbc == []:
            vector = VectorFunctionSpace(mesh, element_type, element_degree)
            scalar = FunctionSpace(mesh, element_type, element_degree - 1)
        else:
            vector = VectorFunctionSpace(mesh, element_type, element_degree, constrained_domain = pbc)
            scalar = FunctionSpace(mesh, element_type, element_degree - 1, constrained_domain = pbc)
        self.space = MixedFunctionSpace([vector,scalar])
        self._unknown_vector = Function(self.space)
        (self._test_displacement, self._test_pressure) = TestFunctions(self.space)
        self._trial_vector = TrialFunction(self.space)

    @property
    def unknown_vector(self):
        return self._unknown_vector
    @property
    def test_vector(self):
        return (self._test_displacement, self._test_pressure)
    @property
    def trial_vector(self):
        return self._trial_vector

    def create_dirichlet_conditions(self, problem):
        self.bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                            problem.dirichlet_boundaries(),
                                            self.space.sub(0))
        return self.bcu
