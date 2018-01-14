from dolfin import *
from cbc.problem import CBCProblem

def default_parameters():
  """Return default parameters for linear homogenization"""
  p = Parameters("solver_parameters")
  p.add("plot_solution", True)
  p.add("save_solution", False)
  p.add("store_solution_data", False)
  p.add("element_degree", 2)
  
  return p

class LinearHomogenization(CBCProblem):
  
  def __init__(self):
  
    self.parameters = default_parameters()
    self._correctors_chi = None
    
  def solve(self):
  
  self.solver = LinearHomogenizationSolver(self, self.parameters) 
  #TODO:Implement+assign solutions to _correctors_chi
  #TODO:Solver should take indxs as an input
  return self.solver.solve()

  def elasticity_tensor(self):
    """Return the elasticity (tangent) tensor.
       IMPLEMENTED BY A USER"""
    print "Elasticity tensor is not implemented"
    return None 

  def periodic_boundaries(self):
    """Return the periodic boundary conditions.
       IMPLEMENTED BY A USER"""
    print "Periodic boundaries must be implemented"
    return None

  def Pi_functions(i,j,dim):
    val = ("0.0",)*dim
    val[i] = "x[j]"
    return Expression(val,j=j,degree=1)

  def correctors_chi(self, indxs = None):
    """Return \chi_ij corrector.
       For None return a list of all \chi"""
    if self._correctors_chi is None:
      self.solve()

    if indxs in None:
      return self._correctors_chi
    return self._correctors_chi[indxs[0],indxs[1]]

  def correctors_omega(self, indxs = None):
    """Return \omega_ij corrector.
       For None return a list of all \omega"""
    #TODO:Implement
  return None

  def displacement_correction(self):
    """Return u_1"""
    #TODO:Implement
    return None

  def __str__(self):
    """Return a short description of the problem"""
    return "Linear homogenization problem" 

class LinearHomogenizationSolver(CBCSolver):
  """Solves the linear homogenization equation"""
  def __init__(self, problem, parameters):
    """Initialise the solver"""

    #Define function spaces
    element_degree = parameters['element_degree']
    pbc = problem.periodic_boundaries()

    vector = FunctionSpace_U(problem.mesh(), 'CG', element_degree, pbc)
    print "Number of DOFs = %d" % vector.space.dim()

    #Equation
    A = problem.elasticity_tensor()
    L1 = LinearElasticityTerm(A, vector.unknown_displacement,
           vector.test_displacement)
    #L2 = LinearElasticityTerm(A, 

    #TODO:RHS and Pi operator

    #TODO:Linear variational solver

    self.function_space = vector
    self.parameters = parameters
    self.mesh = problem.mesh()
    self.equation = solver
    
  def solve(self, indxs):
    """Solve the homogenization problem"""
    #TODO:Implement
    self.equation.solve()
    chi = self.functionspace.unknown_displacement

    return chi
