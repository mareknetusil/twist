from dolfin import *

class CoordinateSystem():
    def __init__(self, mesh = None, displacement = None):
        self.mesh = mesh
        self.displacement = displacement
        if mesh != None:
            self.dim = mesh.geometry().dim()
            self.x = SpatialCoordinate(mesh)
        else:
            self.dim = None
            self.x = None

    def set_displacement(self, displacement):
        self.displacement = displacement

    def set_mesh(self, mesh):
        self.mesh = mesh
        self.dim = mesh.geometry().dim()
        if self.x == None:
            self.x = SpatialCoordinate(mesh)

    def lame_coefficients(self):
        pass

    def christoffel_symbols(self, deformed = False):
        pass

    def metric_tensor(self, direction, deformed = False):
        pass

    def volume_jacobian(self):
        h = self.lame_coefficients()
        return h[0]*h[1]*h[2]



class CartesianSystem(CoordinateSystem):
    
    def lame_coefficients(self, deformed = False):
        return [1.0, 1.0, 1.0]

    def christoffel_symbols(self, deformed = False):
        return as_tensor([[[0.0 for i in range(self.dim)] for j in range(self.dim)] for k in range(self.dim)])

    def metric_tensor(self, direction, deformed = False):
        return Identity(self.dim)



class CylindricalSystem(CoordinateSystem):

    def lame_coefficients(self, deformed = False):
        if deformed:
            h = [1.0, self.x[0] + self.displacement[0], 1.0]
        else:
            h = [1.0, self.x[0], 1.0]
        return as_vector(h)

    def christoffel_symbols(self, deformed = False):
        def gamma(i,j,k):
            if (i == 0 and (j == 1 and k == 1)):
                return -self.x[0] if not deformed else -(self.x[0] + self.displacement[0])
            elif (i == 1 and ((j == 0 and k == 1) or (j == 1 and k == 0))):
                return 1.0/self.x[0] if not deformed else 1.0/(self.x[0] + self.displacement[0])
            else:
                return 0.0

        return as_tensor([[[gamma(i,j,k) for i in range(self.dim)] for j in range(self.dim)] for k in range(self.dim)])

    def metric_tensor(self, direction, deformed = False):
        def g(i,j):
            if i == j:
                h = self.lame_coefficients(deformed)[i]
                return h**2 if direction == 'lower' else 1.0/(h**2)
            else:
                return 0.0

        return as_tensor([[g(i,j) for i in range(self.dim)] for j in range(self.dim)])
