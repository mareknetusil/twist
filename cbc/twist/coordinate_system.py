from dolfin import *

class CoordinateSystem():
    def __init__(self, mesh = None, displacement = None):
        self.mesh = mesh
        self.displacement = displacement
        if mesh != None:
            self.dim = mesh.geometry().dim()

    def lame_coefficients(self):
        pass

    def christoffel_symbols(self, i, j, k, deformed = False):
        pass

    def metric_tensor(self, direction, deformed = False):
        pass


class CartesianSystem(CoordinateSystem):
    
    def lame_coefficients(self):
        return [1.0, 1.0, 1.0]

    def christoffel_symbols(self, deformed = False):
        return as_tensor([[[0.0 for i in range(self.dim)] for j in range(self.dim)] for k in range(self.dim)])

    def metric_tensor(self, direction, deformed = False):
        return Identity(self.dim)

    def volume_jacobian(self):
        h = self.lame_coefficients()
        return h[0]*h[1]*h[2]



#class CylindricalSystem(CoordinateSystem):
#
#    def lame_coefficients(self, )

