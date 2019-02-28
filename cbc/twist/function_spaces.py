import fenics
from cbc.common import create_dirichlet_conditions


class FunctionSpace_U():
    """
    Discrete function space for the displacement U
    """

    def __init__(self, mesh, element_type, element_degree, pbc=None):
        if not pbc:
            self.space = fenics.VectorFunctionSpace(mesh, element_type,
                                                    element_degree)
        else:
            self.space = fenics.VectorFunctionSpace(mesh, element_type,
                                                    element_degree,
                                                    constrained_domain=pbc)
        self._unknown_displacement = fenics.Function(self.space)
        self._test_displacement = fenics.TestFunction(self.space)
        self._trial_displacement = fenics.TrialFunction(self.space)

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

    def __init__(self, mesh, element_type, element_degree, pbc=None):
        if not pbc:
            vector = fenics.VectorFunctionSpace(mesh, element_type,
                                                element_degree)
            scalar = fenics.FunctionSpace(mesh, element_type,
                                          element_degree - 1)
        else:
            vector = fenics.VectorFunctionSpace(mesh, element_type,
                                                element_degree,
                                                constrained_domain=pbc)
            scalar = fenics.FunctionSpace(mesh, element_type,
                                          element_degree - 1,
                                          constrained_domain=pbc)
        self.space = fenics.MixedFunctionSpace([vector, scalar])
        self._unknown_vector = fenics.Function(self.space)
        (self._test_displacement, self._test_pressure) \
            = fenics.TestFunctions(self.space)
        self._trial_vector = fenics.TrialFunction(self.space)
        self.bcu = None

    @property
    def unknown_vector(self):
        return self._unknown_vector

    @property
    def test_vector(self):
        return self._test_displacement, self._test_pressure

    @property
    def trial_vector(self):
        return self._trial_vector

    def create_dirichlet_conditions(self, problem):
        self.bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                               problem.dirichlet_boundaries(),
                                               self.space.sub(0))
        return self.bcu
