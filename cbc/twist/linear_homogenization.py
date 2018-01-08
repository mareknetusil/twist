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
  
    self.parameters = Parameters("problem_parameters")
    self.parameters.add(default_parameters())
    
  def solve(self):
  
    
