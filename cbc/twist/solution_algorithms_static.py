__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from nonlinear_solver import *
from solution_algorithms_blocks import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient, Jacobian, Grad_Cyl
from sys import exit
from numpy import array, loadtxt, linalg

def default_parameters():
   "Return default solver parameters."
   p = Parameters("solver_parameters")
   p.add("plot_solution", True)
   p.add("save_solution", False)
   p.add("store_solution_data", False)
   p.add("element_degree",2)
   p.add("problem_formulation",'displacement')
   new = Parameters("newton_solver")
   new.add("value", 1.0)
   new.add("adaptive", True)
   new.add("loading_number_of_steps", 1)
   p.add(new)

   return p

class StaticMomentumBalanceSolver_U(CBCSolver):
   "Solves the static balance of linear momentum"

   def __init__(self, problem, parameters):
      """Initialise the static momentum balance solver"""

      # Define function spaces
      element_degree = parameters['element_degree']
      pbc = problem.periodic_boundaries()

      vector = FunctionSpace_U(problem.mesh(), 'CG', element_degree, pbc)
      vector.create_dirichlet_conditions(problem)

      # Print DOFs
      print "Number of DOFs = %d" % vector.space.dim()

      # Driving forces
      B = problem.body_force()
      P  = problem.first_pk_stress(vector.unknown_displacement)

      
      # If no body forces are specified, assume it is 0
      if B == []:
         B = Constant((0,)*vector.test_displacement.geometric_dimension())
      
      self.theta = Constant(1.0)

      L1 = ElasticityDisplacementTerm(P, self.theta*B, vector.test_displacement)

      # Add contributions to the form from the Neumann boundary
      # conditions
      neumann_conditions = problem.neumann_conditions()
      

      # If no Neumann conditions are specified, assume it is 0
      if neumann_conditions == []:
         neumann_conditions = [Constant((0,)*vector.test_displacement.geometric_dimension())]

      neumann_conditions = [self.theta*g for g in neumann_conditions]

      L2 = NeumannBoundaryTerm(neumann_conditions, problem.neumann_boundaries(),\
                                        vector.test_displacement, problem.mesh())

      L = L1 + L2

      a = derivative(L, vector.unknown_displacement, vector.trial_displacement)

      solver = AugmentedNewtonSolver(L, vector.unknown_displacement, a, vector.bcu,\
                                       load_increment = self.theta)

      newton_parameters = parameters['newton_solver']
      solver.parameters['newton_solver'].update(newton_parameters)

      self.functionspace = vector

      # Store parameters
      self.parameters = parameters

      # Store variables needed for time-stepping
      # FIXME: Figure out why I am needed
      self.mesh = problem.mesh()
      self.equation = solver
      self.a = a

   def solve(self):
      """Solve the mechanics problem and return the computed
      displacement field"""

      # Solve problem
      self.equation.solve()
      u = self.functionspace.unknown_displacement

      # Plot solution
      if self.parameters["plot_solution"]:
         plot(u, title="Displacement", mode="displacement", axes=True, rescale=True)
         interactive()

      # Store solution (for plotting)
      #FIXME: Update to XDMFFile
      if self.parameters["save_solution"]:
         displacement_file = File("displacement.xdmf")
         displacement_file << u

      # Store solution data
      if self.parameters["store_solution_data"]:
         displacement_series = TimeSeries("displacement")
         displacement_series.store(u.vector(), 0.0)

      return u

class StaticMomentumBalanceSolver_UP(CBCSolver):
   "Solves the static balance of linear momentum"

   parameters['form_compiler']['representation'] = 'uflacs'
   parameters['form_compiler']['optimize'] = True
   parameters['form_compiler']['quadrature_degree'] = 4

   def __init__(self, problem, parameters):
      """Initialise the static momentum balance solver"""

      # Get problem parameters
      element_degree = parameters["element_degree"]

      # Define function spaces
      pbc = problem.periodic_boundaries()

      mixed_space = FunctionSpace_UP(problem.mesh(), 'CG', element_degree, pbc)
      mixed_space.create_dirichlet_conditions(problem)

      # Print DOFs
      print "Number of DOFs = %d" % mixed_space.space.dim()

      # Create boundary conditions

      # Define fields
      # Test and trial functions
      w = mixed_space.unknown_vector
      (u,p) = split(w)

      # Driving forces
      B = problem.body_force()

      # If no body forces are specified, assume it is 0
      if B == []:
         B = Constant((0,)*problem.mesh().geometry().dim())

      # First Piola-Kirchhoff stress tensor based on the material
      # model
      self.theta = Constant(1.0)
      P  = problem.first_pk_stress(u)

      L = ElasticityPressureTerm(u, p, mixed_space.test_vector[0])
      L += ElasticityDisplacementTerm(P, self.theta*B, mixed_space.test_vector[0])
      L += VolumeChangeTerm(u, p, mixed_space.test_vector[1], problem)

      # Add contributions to the form from the Neumann boundary
      # conditions

      # Get Neumann boundary conditions on the stress
      neumann_conditions = problem.neumann_conditions()

      # If no Neumann conditions are specified, assume it is 0
      if neumann_conditions == []:
         neumann_conditions = [Constant((0,)*problem.mesh().geometry().dim())]

      neumann_conditions = [self.theta*g for g in neumann_conditions]

      L += NeumannBoundaryTerm(neumann_conditions, problem.neumann_boundaries(), \
                                     mixed_space.test_vector[0], problem.mesh())

      a = derivative(L, w, mixed_space.trial_vector)

      solver = AugmentedNewtonSolver(L, w, a, mixed_space.bcu,\
                                       load_increment = self.theta)
      newton_parameters = parameters['newton_solver']
      solver.parameters['newton_solver'].update(newton_parameters)


      # Store parameters
      self.parameters = parameters

      # Store variables needed for time-stepping
      # FIXME: Figure out why I am needed
      self.mesh = problem.mesh()
      self.equation = solver
      self.a = a
      (u, p) = w.split()
      self.u = u
      self.p = p

   def solve(self):
      """Solve the mechanics problem and return the computed
      displacement field"""

      # Solve problem
      self.equation.solve()


      # Plot solution
      if self.parameters["plot_solution"]:
         plot(self.u, title="Displacement", mode="displacement", axes=True, rescale=True)
         plot(self.p, title="Pressure", axes=True, rescale=True)
         interactive()

      # Store solution (for plotting)
      if self.parameters["save_solution"]:
         displacement_file = File("displacement.xdmf")
         pressure_file = File("pressure.xdmf")
         displacement_file << self.u
         pressure_file << self.p

      # Store solution data
      if self.parameters["store_solution_data"]:
         displacement_series = TimeSeries("displacement")
         pressure_series = TimeSeries("pressure")
         displacement_series.store(self.u.vector(), 0.0)
         pressure_series.store(self.p.vector(),0.0)

      return self.u, self.p
