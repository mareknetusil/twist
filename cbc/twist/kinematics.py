__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from cbc.twist.coordinate_system import CartesianSystem

# Renaming grad to Grad because it looks nicer in the reference
# configuration 

from ufl import grad as ufl_grad
def metric_tensor(u,i):
    chi = SpatialCoordinate(u.domain())
    if i == 'up':
        h = as_vector([1.0, 1.0/(chi[0] + u[0]), 1.0])
    elif i == 'down':
        h = as_vector([1.0, chi[0] + u[0], 1.0])

    return as_matrix([[h[0]*h[0],0.0,0.0], [0.0,h[1]*h[1],0.0], [0.0,0.0,h[2]*h[2]]])

def Metric_Tensor(u,i):
    chi = SpatialCoordinate(u.domain())
    if i == 'up':
        H = as_vector([1.0,1.0/chi[0],1.0])
    elif i == 'down':
        H = as_vector([1.0,chi[0],1.0])

    return as_matrix([[H[0]*H[0],0.0,0.0], [0.0,H[1]*H[1],0.0], [0.0,0.0,H[2]*H[2]]])




# Deformation gradient
def DeformationGradient(u):
    I = SecondOrderIdentity(u)
    return variable(I + Grad(u))
    

def Grad_Cyl(v, coordinate_system = CartesianSystem()):

    u = coordinate_system.displacement
    F = DeformationGradient(u)
    
    i, j, k, I = indices(4)
    gamma = coordinate_system.christoffel_symbols()
    v_iI = Dx(v[i],I) - v[k]*gamma[k,i,j]*F[j,I]

    """
    chi = SpatialCoordinate(u.domain())

    a00 = Dx(v[0],0) - v[1]/(chi[0]+u[0])*Dx(chi[1]+u[1],0)
    a01 = Dx(v[0],1) - v[1]/(chi[0]+u[0])*Dx(chi[1]+u[1],1)
    a02 = Dx(v[0],2) - v[1]/(chi[0]+u[0])*Dx(chi[1]+u[1],2)

    a10 = Dx(v[1],0) + (chi[0]+u[0])*v[0]*Dx(chi[1]+u[1],0) - v[1]/(chi[0]+u[0])*Dx(chi[0]+u[0],0)
    a11 = Dx(v[1],1) + (chi[0]+u[0])*v[0]*Dx(chi[1]+u[1],1) - v[1]/(chi[0]+u[0])*Dx(chi[0]+u[0],1)
    a12 = Dx(v[1],2) + (chi[0]+u[0])*v[0]*Dx(chi[1]+u[1],2) - v[1]/(chi[0]+u[0])*Dx(chi[0]+u[0],2)

    a20 = Dx(v[2],0)
    a21 = Dx(v[2],1)
    a22 = Dx(v[2],2)


    return as_tensor([[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]])
    """

    return as_tensor(v_iI, (i, I))


def Grad_U(u, coordinate_system = CartesianSystem()):

    I, J, K = indices(3)
    Gamma = coordinate_system.christoffel_symbols()

    u_IJ = Dx(u[I],J) + Gamma[I,J,K]*u[K]

    return as_tensor(u_IJ, (I, J))


    """
    chi = SpatialCoordinate(u.domain())

    a00 = Dx(u[0],0)
    a01 = Dx(u[0],1) - chi[0]*u[1]
    a02 = Dx(u[0],2)
    
    a10 = Dx(u[1],0) + 1/chi[0]*u[1]
    a11 = Dx(u[1],1) + 1/chi[0]*u[0]
    a12 = Dx(u[1],2)

    a20 = Dx(u[2],0)
    a21 = Dx(u[2],1)
    a22 = Dx(u[2],2)

    return as_tensor([[a00,a01,a02],[a10,a11,a12],[a20,a21,a22]])
    """


def Grad(v):
    return ufl_grad(v) 

# Infinitesimal strain tensor
def InfinitesimalStrain(u):
    return variable(0.5*(Grad(u) + Grad(u).T))

# Second order identity tensor
def SecondOrderIdentity(u):
    return variable(Identity(u.cell().geometric_dimension()))


# Determinant of the deformation gradient
def Jacobian(u, coordinate_system = CartesianSystem()):
    G = coordinate_system.metric_tensor('raise')
    g = coordinate_system.metric_tensor('lower', deformed = True)
    u = coordinate_system.displacement
    F = DeformationGradient(u)

    #TODO: Should it be sqrt(det(g)*det(G))?
    return variable(det(g)*det(G)*det(F))


    """
    G = det(Metric_Tensor(u,'up'))
    g = det(metric_tensor(u,'down'))
    F = DeformationGradient(u)
    return variable(g*G*det(F))
    """

# Right Cauchy-Green tensor
def RightCauchyGreen(u, coordinate_system = CartesianSystem()):

    G_raise = coordinate_system.metric_tensor('raise')
    G_lower = coordinate_system.metric_tensor('lower')
    I = SecondOrderIdentity(u)
    gradu = Grad_U(u, coordinate_system)

    return variable(I + gradu + G_raise*gradu.T*G_lower + (G_raise*gradu.T*G_lower)*gradu)
    

    """
    Gu = Metric_Tensor(u,'up')
    Gd = Metric_Tensor(u,'down')
    I = SecondOrderIdentity(u)
    
    return variable(I + Grad_U(u) + Gu*(Grad_U(u).T)*Gd + Gu*(Grad_U(u).T)*Gd*Grad_U(u))
    """

# Green-Lagrange strain tensor
def GreenLagrangeStrain(u, coordinate_system):
    I = SecondOrderIdentity(u)
    C = RightCauchyGreen(u, coordinate_system)
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
def CauchyGreenInvariants(u, coordinate_system):
    C = RightCauchyGreen(u, coordinate_system)
    [I1, I2, I3] = Invariants(C)
    return [variable(I1), variable(I2), variable(I3)]

# Isochoric part of the deformation gradient
def IsochoricDeformationGradient(u, coordinate_system):
    F = DeformationGradient(u)
    J = Jacobian(u, coordinate_system)
    return variable(J**(-1.0/3.0)*F)

# Isochoric part of the right Cauchy-Green tensor
def IsochoricRightCauchyGreen(u, coordinate_system):
    C = RightCauchyGreen(u, coordinate_system)
    J = Jacobian(u, coordinate_system)
    return variable(J**(-2.0/3.0)*C)

# Invariants of the ischoric part of the (right/left) Cauchy-Green
# tensor. Note that I3bar = 1 by definition.
def IsochoricCauchyGreenInvariants(u, coordinate_system):
    Cbar = IsochoricRightCauchyGreen(u, coordinate_system)
    [I1bar, I2bar, I3bar] = Invariants(Cbar)
    return [variable(I1bar), variable(I2bar)]

# Principal stretches
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
def PiolaTransform(A, u):
    J = Jacobian(u)
    F = DeformationGradient(u)
    B = J*A*inv(F).T
    return B

# Push-forward of a two-tensor from the reference to the current
# configuration
def InversePiolaTransform(A, u):
    J = Jacobian(u)
    F = DeformationGradient(u)
    B = (1/J)*A*F.T
    return B


# Computes M*C^nM
# for n = 1 equals to the stretch in the direction M
def DirectionalStretch(u, M, degree = 1):
    C = RightCauchyGreen(u)
    Cpow = SecondOrderIdentity(u)
    if degree >= 1:
        for i in range(degree):
            Cpow = C*Cpow
        
    directionalstretch = inner(M,Cpow*M)
    return variable(directionalstretch)
