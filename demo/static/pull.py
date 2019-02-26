from __future__ import print_function

__author__ = "Marek Netusil"

import fenics
from cbc.twist.problem_definitions import StaticHyperelasticity
from cbc.twist.material_models import *
from sys import argv

""" DEMO - Hyperelastic cube is stretched/compressed by a traction acting on one side """

class Pull(StaticHyperelasticity):
    """ Definition of the hyperelastic problem """
    def __init__(self, *args, **kwargs):
        StaticHyperelasticity.__init__(self, *args, **kwargs)
        n = 8
        self._mesh = fenics.UnitCubeMesh(n, n, n)

    def mesh(self):
        return self._mesh

    # Setting up dirichlet conditions and boundaries
    def dirichlet_values(self):
        clamp = Constant((0.0, 0.0, 0.0))
        return [clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        return [left]

    # Setting up neumann conditions and boundaries
    def neumann_conditions(self):
        try:
            traction = Constant((float(argv[1]),0.0,0.0))
        except:
            traction = Constant((200.0,0.0,0.0))
        return [traction]

    def neumann_boundaries(self):
        right = "x[0] == 1.0"
        return [right]


    # List of material models
    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 1e2
        lmbda    = 1e2
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4
        l = sqrt(2.0)/2.0
        M = Constant((l,0.0,l))
        k1 = 1e2; k2 = 1e1


        materials = []
        materials.append(MooneyRivlin({'C1':mu/2, 'C2':mu/2, 'bulk':lmbda}))
        materials.append(StVenantKirchhoff({'mu':mu, 'bulk':lmbda}))
        materials.append(neoHookean({'half_nkT':mu, 'bulk':lmbda}))
        materials.append(Isihara({'C10':C10,'C01':C01,'C20':C20,'bulk':lmbda}))
        materials.append(Biderman({'C10':C10,'C01':C01,'C20':C20,'C30':C30,'bulk':lmbda}))
        materials.append(AnisoTest({'mu1':mu,'mu2':2*mu,'M':M,'bulk':lmbda}))
        materials.append(GasserHolzapfelOgden({'mu':mu,'k1':k1,'k2':k2,'M':M,'bulk':lmbda}))
        materials.append(Ogden({'alpha1':1.3,'alpha2':5.0,'alpha3':-2.0,\
                                'mu1':6.3e5,'mu2':0.012e5,'mu3':-0.1e5}))

        try:
            index = int(argv[2])
        except:
            index = 2
        print(str(materials[index]))
        return materials[index]

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic cube stretching/compression solved by " + self.method



# Setup the problem
pull = Pull()
pull.name_method("DISPLACEMENT BASED FORMULATION")
#pull.parameters['problem_formulation'] = 'mixed_up'

# Solve the problem
print(pull)
pull.solve()

