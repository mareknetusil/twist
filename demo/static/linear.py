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
        n = 8
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

        i, j, k, l = fenics.indices(4)
        I = fenics.Identity(3)
        A_ijkl = lmbda * I[i, j] * I[k, l] \
                 + mu * (I[i, k] * I[j, l] + I[i, l] * I[j, k]
                         - 2/3.0 * I[i, j] * I[k, l])
        A = fenics.as_tensor(A_ijkl, (i, j, k, l))

        mat = LinearGeneral({'A': A})
        return mat


    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic cube stretching/compression solved by " + self.method


# Setup the problem
pull = Pull("DISPLACEMENT BASED FORMULATION")
pull.parameters["output_dir"] \
    = "output/pull/{}".format(pull.material_model())
# pull.parameters['problem_formulation'] = 'mixed_up'

# Solve the problem
print(pull)
pull.solve()
