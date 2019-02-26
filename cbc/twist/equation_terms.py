from fenics import dx, ds, inner, Measure
from cbc.twist.nonlinear_solver import *
from cbc.twist.kinematics import *
from cbc.common import *


def elasticity_displacement(P, B, v):
    """
    Basis for the dislacement formulation: - Div(P) = B
    Returns: P:Grad(v)*dx - inner(B,v)*dx
    """
    L = -inner(B, v) * dx

    # The variational form corresponding to hyperelasticity
    if isinstance(P, tuple):
        P_list, subdomains_list = P

        for (index, P) in enumerate(P_list):
            new_dx = Measure('dx')(subdomain_data=subdomains_list[index][0])
            L += inner(P, Grad(v)) * new_dx(subdomains_list[index][1])
    else:
        L += inner(P, Grad(v)) * dx

    return L


def elasticity_pressure(u, p, v):
    """
    The nonlinear pressure term: pJF^(-T)
    Returns: -p*j*inner(F^(-T),Grad(v))*dx
    """
    J = Jacobian(u)
    F = DeformationGradient(u)

    L = -p * J * inner(g * inv(F.T), Grad(v)) * dx
    return L


def volume_change(u, p, q, problem):
    """
    The equation of volume change. Compressible material for bulk modulus
    positive and incompressible otherwise.
    Returns: (1/lb*p + J - 1)*q*dx for compressible, (J - 1)*q*dx for incompressible
    """
    material_model = problem.material_model()
    J = Jacobian(u)

    L = Constant(0.0) * q * dx
    if isinstance(material_model, tuple):
        material_list, cell_function = material_model
        new_dx = Measure('dx')[cell_function]
        for (index, material) in enumerate(material_list):
            material_parameters = material_list[index].parameters
            lb = material_parameters['bulk']
            if lb <= 0.0:
                L = + (J - 1.0) * q * new_dx(index)
            else:
                L += (1.0 / lb * p + J - 1.0) * q * new_dx(index)
    else:
        lb = problem.material_model().parameters['bulk']
        if lb <= 0.0:
            L += (J - 1.0) * q * dx
        else:
            L += (1.0 / lb * p + J - 1.0) * q * dx
    return L


# TODO: Get rid of the mesh argument
def neumann_condition(neumann_conditions, neumann_boundaries, v, mesh):
    """
    Neumann boundary condition: dU/dN = g
    Returns: - inner(g,v)*dS
    """
    boundary = FacetFunction("size_t", mesh)
    boundary.set_all(len(neumann_boundaries) + 1)

    L = - inner(Constant((0,) * v.geometric_dimension()), v) * ds
    dsb = Measure('ds')(subdomain_data=boundary)
    for (i, neumann_boundary) in enumerate(neumann_boundaries):
        compiled_boundary = CompiledSubDomain(neumann_boundary)
        compiled_boundary.mark(boundary, i)
        L += - inner(neumann_conditions[i], v) * dsb(i)

    return L
