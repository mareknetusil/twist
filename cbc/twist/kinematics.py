__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
#from cbc.twist.coordinate_system import CartesianSystem

# Renaming grad to Grad because it looks nicer in the reference
# configuration

from ufl import grad as ufl_grad



# Deformation gradient
def DeformationGradient(u):
    I = SecondOrderIdentity(u)
    return variable(I + Grad(u))

def Grad(v):
    return ufl_grad(v)

# Infinitesimal strain tensor
def InfinitesimalStrain(u):
    return variable(0.5*(Grad(u) + Grad(u).T))

# Second order identity tensor
def SecondOrderIdentity(u):
    return variable(Identity(u.geometric_dimension()))

# Determinant of the deformation gradient
def Jacobian(u):
    F = DeformationGradient(u)
    return variable(det(F))

# Right Cauchy-Green tensor
def RightCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F.T*F)

# Green-Lagrange strain tensor
def GreenLagrangeStrain(u):
    I = SecondOrderIdentity(u)
    C = RightCauchyGreen(u)
    return variable(0.5*(C - I))

# Left Cauchy-Green tensor
def LeftCauchyGreen(u):
    F = DeformationGradient(u)
    return variable(F*F.T)

# Euler-Almansi strain tensor
def EulerAlmansiStrain(u):
    I = SecondOrderIdentity(u)
    b = LeftCauchyGreen(u)
    return variable(0.5*(I - inv(b)))

# Invariants of an arbitrary tensor, A
def Invariants(A):
    I1 = tr(A)
    I2 = 0.5*(tr(A)**2 - tr(A*A))
    I3 = det(A)
    return [I1, I2, I3]

# Invariants of the (right/left) Cauchy-Green tensor
#TODO: NEEDS TESTING
def CauchyGreenInvariants(u):
    C = RightCauchyGreen(u)
    [I1, I2, I3] = Invariants(C)
    return [variable(I1), variable(I2), variable(I3)]

# Isochoric part of the deformation gradient
#TODO: NEEDS TESTING
def IsochoricDeformationGradient(u):
    F = DeformationGradient(u)
    J = Jacobian(u)
    return variable(J**(-1.0/3.0)*F)

# Isochoric part of the right Cauchy-Green tensor
#TODO: NEEDS TESTING
def IsochoricRightCauchyGreen(u):
    C = RightCauchyGreen(u)
    J = Jacobian(u)
    return variable(J**(-2.0/3.0)*C)

# Invariants of the ischoric part of the (right/left) Cauchy-Green
# tensor. Note that I3bar = 1 by definition.
#TODO: NEEDS TESTING
def IsochoricCauchyGreenInvariants(u):
    Cbar = IsochoricRightCauchyGreen(u)
    [I1bar, I2bar, I3bar] = Invariants(Cbar)
    return [variable(I1bar), variable(I2bar)]

# Principal stretches
#TODO: NEEDS TESTING
def PrincipalStretches(u):
    C = RightCauchyGreen(u)
    S = FunctionSpace(u.function_space().mesh(), "CG", 1)
    if (u.cell().geometric_dimension() == 2):
        D = sqrt(tr(C)*tr(C) - 4.0*det(C))
    eig1 = sqrt(0.5*(tr(C) + D))
    eig2 = sqrt(0.5*(tr(C) - D))
    return [variable(eig1), variable(eig2)]

    if (u.cell().geometric_dimension() == 3):
        c = (1.0/3.0)*tr(C)
        D = C - c*SecondOrderIdentity(u)
        q = (1.0/2.0)*det(D)
        p = (1.0/6.0)*inner(D, D)
        ph = project(p, S)
        if (norm(ph) < DOLFIN_EPS):
            eig1 = sqrt(c)
            eig2 = sqrt(c)
            eig3 = sqrt(c)
        else:
            phi = (1.0/3.0)*atan(sqrt(p**3.0 - q**2.0)/q)
            if (phi < 0.0):
                    phi = phi + DOLFIN_PI/3.0
            end
            eig1 = sqrt(c + 2*sqrt(p)*cos(phi))
            eig2 = sqrt(c - sqrt(p)*(cos(phi) + sqrt(3)*sin(phi)))
            eig3 = sqrt(c - sqrt(p)*(cos(phi) - sqrt(3)*sin(phi)))
            return [variable(eig1), variable(eig2), variable(eig3)]

# Pull-back of a two-tensor from the current to the reference
# configuration
#TODO: NEEDS TESTING
def PiolaTransform(A, u):
    J = Jacobian(u)
    F = DeformationGradient(u)
    B = J*A*inv(F).T
    return B

# Push-forward of a two-tensor from the reference to the current
# configuration
#TODO: NEEDS TESTING
def InversePiolaTransform(A, u):
    J = Jacobian(u)
    F = DeformationGradient(u)
    B = (1/J)*A*F.T
    return B


# Computes M*C^nM
# for n = 1 equals to the stretch in the direction M
#TODO: NEEDS TESTING
def DirectionalStretch(u, M, degree = 1):
    C = RightCauchyGreen(u)
    Cpow = SecondOrderIdentity(u)
    if degree >= 1:
        for i in range(degree):
            Cpow = C*Cpow

    directionalstretch = inner(M,Cpow*M)
    return variable(directionalstretch)
