__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__ = "GNU GPL Version 3 or any later version"

import fenics
from cbc.common import CBCProblem
from cbc.twist.solution_algorithms_static import StaticMomentumBalanceSolver_U
from cbc.twist.solution_algorithms_dynamic import CG1MomentumBalanceSolver
# from cbc.twist.solution_algorithms_static import \
#     default_parameters as solver_parameters_static
# from cbc.twist.solution_algorithms_dynamic import \
#     default_parameters as solver_parameters_dynamic
from cbc.common.CBCSolver import default_parameters as solver_parameters_static
solver_parameters_dynamic = solver_parameters_static
from cbc.twist.kinematics import GreenLagrangeStrain


class StaticHyperelasticity(CBCProblem):
    """Base class for all static hyperelasticity problems"""

    def __init__(self):
        """Create the static hyperelasticity problem"""

        # Set up parameters
        self.parameters = solver_parameters_static()

    def solve(self):
        """Solve for and return the computed displacement field, u"""

        # Create solver
        formulation = self.parameters['problem_formulation']
        if formulation == 'displacement':
            self.solver = StaticMomentumBalanceSolver_U(self, self.parameters)
        else:
            fenics.error("%s formulation not supported." % formulation)

        # Call solver
        return self.solver.solve()

    def body_force(self):
        """Return body force, B"""
        return []

    def body_force_u(self, u):
        # FIXME: This is currently only implemented for the cG(1) solver
        """Return body force, B, depending on displacement u"""
        return []

    def dirichlet_values(self):
        """Return Dirichlet boundary conditions for the displacement field"""
        return []

    def dirichlet_boundaries(self):
        """Return boundaries over which Dirichlet conditions act"""
        return []

    def neumann_conditions(self):
        """Return Neumann boundary conditions for the stress field"""
        return []

    def neumann_boundaries(self):
        """Return boundaries over which Neumann conditions act"""
        return []

    def periodic_boundaries(self):
        """ Return periodic boundaries """
        return []

    def material_model(self):
        pass

    def first_pk_stress(self, u):
        """Return the first Piola-Kirchhoff stress tensor, P, given a
      displacement field, u"""

        material_model = self.material_model()
        if isinstance(material_model, tuple):
            fpk_list = []
            material_list, subdomains_list = material_model
            for material in material_list:
                fpk_list.append(material.first_pk_stress(u))
            return fpk_list, subdomains_list
        else:
            return material_model.first_pk_stress(u)

    def second_pk_stress(self, u):
        """Return the second Piola-Kirchhoff stress tensor, S, given a
      displacement field, u"""
        material_model = self.material_model()
        if isinstance(material_model, tuple):
            spk_list = []
            material_list, subdomains_list = material_model
            for material in material_list:
                spk_list.append(material.second_pk_stress(u))
            return spk_list, subdomains_list
        else:
            return self.material_model().second_pk_stress(u)

    def strain_energy(self, u):
        """Return the strain (potential) energy density given a displacement
      field, u"""

        S = self.second_pk_stress(u)
        E = GreenLagrangeStrain(u)
        if isinstance(S, tuple):
            S_list, subdomains_list = S
            V = fenics.FunctionSpace(u.function_space().mesh(), 'DG', 0)
            temp = fenics.Function(V)
            psi = fenics.Function(V)
            subdomains = subdomains_list[0]
            for cell_no in range(len(subdomains[0].array())):
                subdomain_no = subdomains[0].array()[cell_no]
                temp = fenics.project(0.5 *
                                      fenics.inner(S_list[int(subdomain_no - 1)],
                                                   E),
                                      V)
                psi.vector()[cell_no] = temp.vector()[cell_no][0]
        else:
            psi = 0.5 * fenics.inner(S, E)
        return psi

    def functional(self, u):
        """Return value of goal functional"""
        return None

    def reference(self):
        """Return reference value for the goal functional"""
        return None

    def __str__(self):
        """Return a short description of the problem"""
        return "Static hyperelasticity problem"


class Hyperelasticity(StaticHyperelasticity):
    """Base class for all quasistatic/dynamic hyperelasticity
   problems"""

    def __init__(self):
        """Create the hyperelasticity problem"""
        # Set up parameters
        self.parameters = solver_parameters_dynamic()

        # Create solver later
        self.solver = None

    def solve(self):
        """Solve for and return the computed displacement field, u"""

        # Create solver
        self._create_solver()

        # Update solver parameters
        self.solver.parameters.update(self.parameters)

        # Call solver
        return self.solver.solve()

    def step(self, dt):
        "Take a time step of size dt"

        # Create solver
        self._create_solver()

        # Update solver parameters
        self.solver.parameters.update(self.parameters)

        # Call solver
        return self.solver.step(dt)

    def update(self):
        "Propagate values to next time step"
        return self.solver.update()

    def solution(self):
        "Return current solution values"
        if self.solver is None:
            self._create_solver()
        return self.solver.solution()

    def end_time(self):
        """Return the end time of the computation"""
        pass

    def time_step(self):
        """Return the time step size"""
        pass

    def is_dynamic(self):
        """Return True if the inertia term is to be considered, or
      False if it is to be neglected (quasi-static)"""
        return False

    def time_stepping(self):
        """Set the default time-stepping scheme to
      Hilber-Hughes-Taylor"""
        return "HHT"

    def reference_density(self):
        """Return the reference density of the material"""
        return []

    def initial_conditions(self):
        """Return initial conditions for displacement field, u0, and
      velocity field, v0"""
        return [], []

    def dirichlet_values(self):
        """Return Dirichlet boundary conditions for the displacment
      field"""
        return []

    def dirichlet_boundaries(self):
        """Return boundaries over which Dirichlet conditions act"""
        return []

    def neumann_conditions(self):
        """Return Neumann boundary conditions for the stress field"""
        return []

    def neumann_boundaries(self):
        """Return boundaries over which Neumann conditions act"""
        return []

    def kinetic_energy(self, v):
        """Return the kinetic energy given a velocity field, v"""

        rho0 = self.reference_density()
        ke = fenics.assemble(0.5 * rho0 * fenics.inner(v, v) * fenics.dx,
                             mesh=v.function_space().mesh())
        return ke

    def _create_solver(self):
        "Create solver if not already created"

        # Don't create solver if already created
        if self.solver is not None: return

        # Select solver
        scheme = self.time_stepping()
        if scheme is "CG1":
            fenics.info("Using CG1 time-stepping.")
            self.solver = CG1MomentumBalanceSolver(self, self.parameters)
        else:
            fenics.error("%s time-stepping scheme not supported." % str(
                self.time_stepping()))
