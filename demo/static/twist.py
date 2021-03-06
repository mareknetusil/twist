from __future__ import print_function

__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

import fenics
from cbc.twist.problem_definitions import StaticHyperelasticity
from cbc.twist.material_models import *
from sys import argv


class TwistTest(StaticHyperelasticity):
    """ DEMO - Twisting of a hyperelastic cube """
    def __init__(self, name):
        StaticHyperelasticity.__init__(self)
        n = 8
        self._mesh = fenics.UnitCubeMesh(n, n, n)
        self.name_method(name)

    def mesh(self):
        return self._mesh

    # Setting up dirichlet conditions and boundaries
    def dirichlet_values(self):
        clamp = fenics.Constant((0.0, 0.0, 0.0))
        twist = fenics.Expression(("0.0",
                            "y0 + (x[1] - y0) * cos(theta) - (x[2] - z0) * sin(theta) - x[1]",
                            "z0 + (x[1] - y0) * sin(theta) + (x[2] - z0) * cos(theta) - x[2]"),
                           y0=0.5, z0=0.5, theta=fenics.pi/6, degree = 2)
        return [clamp, twist]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        right = "x[0] == 1.0"
        return [left, right]

    # List of material models
    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu = 3.8461
        lmbda = fenics.Expression("x[0]*5.8 + (1 - x[0])*5.7", degree = 2)
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4

        l = fenics.sqrt(2.0) / 2.0
        M = fenics.Constant((l, 0.0, l))
        k1 = 1e2;
        k2 = 1e1

        materials = [
            MooneyRivlin({'C1': mu / 2, 'C2': mu / 2, 'bulk': lmbda}),
            StVenantKirchhoff({'mu': mu, 'bulk': lmbda}),
            neoHookean({'half_nkT': mu, 'bulk': lmbda}),
            Isihara({'C10': C10, 'C01': C01, 'C20': C20, 'bulk': lmbda}),
            Biderman({'C10': C10, 'C01': C01, 'C20': C20, 'C30': C30,
                      'bulk': lmbda}),
            AnisoTest({'mu1': mu, 'mu2': 2 * mu, 'M': M, 'bulk': lmbda}),
            neoHookean({'half_nkT': mu, 'bulk': lmbda})
            + GasserHolzapfelOgden({'k1': k1, 'k2': k2, 'M': M}),
            Ogden({'alpha1': 1.3, 'alpha2': 5.0, 'alpha3': -2.0, 'mu1': 6.3e5,
                   'mu2': 0.012e5, 'mu3': -0.1e5}),
            LinearIsotropic({'mu': mu, 'bulk': lmbda})
        ]

        try:
            index = int(argv[1])
        except:
            index = 2
        print(materials[index])
        return materials[index]

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic cube twisted by 30 degrees solved by " + self.method



# Setup the problem
twistTest = TwistTest("DISPLACEMENT BASED FORMULATION")
twistTest.parameters["output_dir"] \
    = "output/twist/{}".format(twistTest.material_model())

# Solve the problem
print(twistTest)
twistTest.solve()
