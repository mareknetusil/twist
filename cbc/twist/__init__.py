from dolfin import *
from kinematics import *
from material_model_base import MaterialModel
from material_models import LinearElastic, StVenantKirchhoff, MooneyRivlin, neoHookean, Isihara, Biderman, GentThomas, Ogden, AnisoTest, GasserHolzapfelOgden
from problem_definitions import StaticHyperelasticity, Hyperelasticity
from linear_homogenization import *

# Optimise compilation of forms
parameters["form_compiler"]["cpp_optimize"] = True
# parameters["form_compiler"]["optimize"] = True
