__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__ = "GNU GPL Version 3 or any later version"

# Last changed: 2012-02-21

from time import time
from fenics import Parameters
from dolfin import info, error, Progress
from dolfin import CompiledSubDomain, interpolate
from dolfin import SubDomain, DirichletBC, Constant, Expression
from dolfin.cpp import GenericFunction


def default_parameters():
    """Return default solver parameters."""
    p = Parameters("solver_parameters")
    p.add("output_dir", "output")
    p.add("plot_solution", True)
    p.add("save_solution", True)
    p.add("store_solution_data", False)
    p.add("element_degree", 2)
    p.add("problem_formulation", 'displacement')
    new = Parameters("newton_solver")
    new.add("value", 1.0)
    new.add("adaptive", True)
    new.add("loading_number_of_steps", 1)
    p.add(new)

    return p


class CBCSolver:
    "Base class for all solvers"

    def __init__(self):
        "Constructor"
        self._time_step = 1
        self._progress = None
        self._cpu_time = time()

    # --- Functions that must be overloaded by subclasses ---

    def solve():
        error("solve() function not implemented by solver.")

    def __str__():
        error("__str__ not implemented by solver.")

    # --- Useful functions for solvers ---

    def _end_time_step(self, t, T):
        "Call at end of time step"

        # Record CPU time
        cpu_time = time()
        elapsed_time = cpu_time - self._cpu_time
        self._cpu_time = cpu_time

        # Write some useful information
        s = "Time step %d (t = %g) finished in %g seconds." % (
        self._time_step, t, elapsed_time)
        info("\n" + s + "\n" + len(s) * "-" + "\n")

        # Update progress bar
        if self._progress is None:
            self._progress = Progress("Time-stepping")
        self._progress.update(t / T)

        # Increase time step counter
        self._time_step += 1

        ## Store solution
        # if self.parameters["save_solution"]:
        #    self.velocity_series.store(self.u1.vector(), t)
        #    self.pressure_series.store(self.p1.vector(), t)


# --- Useful functions for solvers (non-member functions) ---

def create_dirichlet_conditions(values, boundaries, function_space):
    """Create Dirichlet boundary conditions for given boundary values,
    boundaries and function space."""

    # Check that the size matches
    if len(values) != len(boundaries):
        error(
            "The number of Dirichlet values does not match the number of Dirichlet boundaries.")

    info("Creating %d Dirichlet boundary condition(s)." % len(values))

    # Create Dirichlet conditions
    bcs = []
    for (i, value) in enumerate(values):

        # Get current boundary
        boundary = boundaries[i]

        if isinstance(value, tuple):
            subdim = value[1]
            value = value[0]
            temp_function_space = function_space.sub(subdim)
        else:
            temp_function_space = function_space

        # Case 0: boundary is a string
        if isinstance(boundary, str):
            boundary = CompiledSubDomain(boundary)
            bc = DirichletBC(temp_function_space, value, boundary)

        # Case 1: boundary is a SubDomain
        elif isinstance(boundary, SubDomain):
            bc = DirichletBC(temp_function_space, value, boundary)

        # Case 2: boundary is defined by a MeshFunction
        elif isinstance(boundary, tuple):
            mesh_function, index = boundary
            bc = DirichletBC(temp_function_space, value, mesh_function, index)

        # Unhandled case
        else:
            error("Unhandled boundary specification for boundary condition. "
                  "Expecting a string, a SubDomain or a (MeshFunction, int) tuple.")

        bcs.append(bc)

    return bcs


def create_initial_condition(value, function_space):
    """Create initial condition from given user data (with intelligent
    handling of different kinds of input)."""

    # Check if we get a GenericFunction subclass (Function, Expression, Constant)
    if isinstance(value, GenericFunction):
        return interpolate(value, function_space)

    # Check if we get an expression
    if isinstance(value, str):
        return create_initial_condition(Expression(value), function_space)

    # Try wrapping input as a Constant
    print(value)
    return create_initial_condition(Constant(value), function_space)
