from __future__ import print_function

__author__ = "Marek Netusil"

import fenics
from cbc.twist.problem_definitions import StaticHyperelasticity
from cbc.twist.material_models import *
from sys import argv


class Pull(StaticHyperelasticity):
    """ DEMO - Hyperelastic cube is stretched/compressed by a traction acting
    on one side """

    def __init__(self, name):
        StaticHyperelasticity.__init__(self)
        n = 10
        self._mesh = fenics.UnitCubeMesh(n, n, n)
        self.name_method(name)

    def mesh(self):
        return self._mesh

    # Setting up dirichlet conditions and boundaries
    def dirichlet_values(self):
        clamp = fenics.Constant((0.0, 0.0, 0.0))
        return [clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        return [left]

    # Setting up neumann conditions and boundaries
    def neumann_conditions(self):
        try:
            traction = fenics.Constant((float(argv[1]), 0.0, 0.0))
        except:
            traction = fenics.Constant((200.0, 0.0, 0.0))
        return [traction]

    def neumann_boundaries(self):
        right = "x[0] == 1.0"
        return [right]

    # List of material models
    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu = 1e2
        lmbda = 1e2
        C10 = 0.171;
        C01 = 4.89e-3;
        C20 = -2.4e-4;
        C30 = 5.e-4

        l = fenics.sqrt(2.0) / 2.0
        M = fenics.Constant((l, 0.0, l))
        k1 = 1e3;
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

        subdomains = fenics.MeshFunction('size_t', self._mesh,
                                         self._mesh.topology().dim())
        subdomains.set_all(0)
        right = fenics.AutoSubDomain(lambda x: x[0] >= .8)
        middle = fenics.AutoSubDomain(lambda x: pow(x[0] - .5, 2) <= .01)
        left = fenics.AutoSubDomain(lambda x: x[0] <= .2)
        right.mark(subdomains, 1)
        middle.mark(subdomains, 1)
        left.mark(subdomains, 1)

        material_list = [materials[2], materials[6]]
        subdomains_list = [(subdomains, 0), (subdomains, 1)]

        return (material_list, subdomains_list)

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic heterogeneous cube stretching/compression solved by " + self.method



# Setup the problem
pull = Pull()
pull.name_method("DISPLACEMENT BASED FORMULATION")

# Solve the problem
print(pull)
pull.solve()
