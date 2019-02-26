from __future__ import print_function
__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class Release(Hyperelasticity):

    def mesh(self):
        n = 8
        return UnitCubeMesh(n, n, n)

    def end_time(self):
        return 10.0

    def time_step(self):
        return 2.e-3

    def is_dynamic(self):
        return True

    def time_stepping(self):
        return "CG1"

    def reference_density(self):
        return 1.0

    def initial_conditions(self):
        """Return initial conditions for displacement field, u0, and
        velocity field, v0"""
        u0 = "saved_u.xml"
        v0 = Expression(("0.0", "0.0", "0.0"), degree=0)
        return u0, v0

    def dirichlet_values(self):
        return [(0, 0, 0)]

    def dirichlet_boundaries(self):
        return ["x[0] == 0.0"]

    def material_model(self):
        mu    = 1e3
        lmbda = 1e3
        material = neoHookean({'half_nkT': mu, 'bulk':lmbda})
        return material

    def __str__(self):
        return "A prestrained hyperelastic cube being let go"

# Setup and solve problem
problem = Release()
problem.parameters['element_degree'] = 1
problem.parameters['save_solution'] = False
print(problem)
problem.solve()
