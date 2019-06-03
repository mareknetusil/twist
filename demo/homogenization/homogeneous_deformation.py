from fenics import *
from cbc.twist.linear_homogenization import LinearHomogenization, \
    function_from_cell_function
from cbc.twist.material_models import *
import itertools as it
import numpy as np
from linear_microproblem import LinearMicroProblem

parameters['std_out_all_processes'] = False

N = 4       # num of time-steps
dF = 1.0   # load increment

dim = 3
n = 20

class SolutionExpression():
    def __init__(self, dim=2):
        self.dim = dim
        self.expression = \
            Expression(("c_xx*x[0] + c_xy*x[1]",
                        "c_yx*x[0] + c_yy*x[1]"),
                       c_xx=0.0, c_xy=0.0,
                       c_yx=0.0, c_yy=0.0,
                       degree=1) \
            if self.dim == 2 else \
            Expression(("c_xx*x[0] + c_xy*x[1] + c_xz*x[2]",
                        "c_yx*x[0] + c_yy*x[1] + c_yz*x[2]",
                        "c_xz*x[0] + c_zy*x[1] + c_zz*x[2]"),
                       c_xx=0.0, c_xy=0.0, c_xz=0.0,
                       c_yx=0.0, c_yy=0.0, c_yz=0.0,
                       c_zx=0.0, c_zy=0.0, c_zz=0.0,
                       degree=1)

    def __call__(self, *args, **kwargs):
        return self.expression

    def set_gradient(self, grad):
        # if not G.shape == (self.dim**2,):
        #     return None
        if self.dim == 2:
            self.expression.c_xx = grad[0]
            self.expression.c_xy = grad[1]
            self.expression.c_yx = grad[2]
            self.expression.c_yy = grad[3]
        elif self.dim == 3:
            self.expression.c_xx = grad[0]
            self.expression.c_xy = grad[1]
            self.expression.c_xz = grad[2]
            self.expression.c_yx = grad[3]
            self.expression.c_yy = grad[4]
            self.expression.c_yz = grad[5]
            self.expression.c_zx = grad[6]
            self.expression.c_zy = grad[7]
            self.expression.c_zz = grad[8]


def elasticity_tensor_to_matrix(A, dim=2):
    return np.array([[A[i,j,k,l].value() for (k,l) in it.product(range(dim), range(dim))]
                     for (i,j) in it.product(range(dim), range(dim))])

def stress_avg_vector(P, dim=2):
    return np.array([assemble(P[i,j]*dx) for (i,j) in it.product(range(dim), range(dim))])


linHom = LinearMicroProblem(dim=dim, n=n)
linHom.parameters["plot_solution"] = False
linHom.parameters["save_solution"] = False

output_dir = "output/homogenization/"

V = VectorFunctionSpace(linHom.mesh(), 'CG', 1)
T = TensorFunctionSpace(linHom.mesh(), 'CG', 1)
u0_exp = SolutionExpression(dim)

u0 = project(u0_exp(), V)
P = linHom.nonlin_mat.first_pk_stress(u0)
# u0 = Function(V)
uFile = File(output_dir + "u.pvd")
PFile = File(output_dir + "P.pvd")
for (i, j) in it.product(range(dim), range(dim)):
    chiFile = File(output_dir + "chi_({},{}).pvd".format(i, j))

# Computation
info('===== START =====')
for n in range(N):
    # Incremental loading - n-th time step
    # Assumption: computed u^{n-1} for loading F^{n-1}
    #
    # F_rhs = np.array([n*dF, 0.0, 0.0, 0.0])
    F_rhs = np.zeros((dim**2, ))
    F_rhs[0] = n*dF
    info('F_rhs = {}'.format(F_rhs))

    # Solving the equation - Newton algorithm
    for k in range(100):
        # Newton iteration
        info('n: {}\t k: {}'.format(n, k))
        linHom.create_elasticity_tensor(u0)
        linHom.solve()

        A = linHom.homogenized_elasticity_tensor()

        #for (i, j) in it.product(range(dim), range(dim)):
        #    chiFile.write(linHom.correctors_chi((i,j)), n)

        A_mat = elasticity_tensor_to_matrix(A, dim)
        P_rhs = stress_avg_vector(P, dim)

        G = np.linalg.solve(A_mat, F_rhs - P_rhs)

        u0_exp.set_gradient(G)

        u_inc = project(u0_exp(), V)
        u_corr = linHom.displacement_correction(u_inc)
        u0 = u0 + u_inc + u_corr
        u0 = project(u0, V)

        P = linHom.nonlin_mat.first_pk_stress(u0)
        P_fce = project(P, T)

        #inc_norm = assemble(inner(u_inc, u_inc)*dx)
        inc_norm = np.linalg.norm(G, ord=np.inf)
        info('|G| = {}'.format(inc_norm))
        if inc_norm < 1e-6:
            break

    # Save the final solution before proceeding to the next load increment
    u0.rename('u', 'displacement')
    P_fce.rename('P', 'first P-K stress')
    uFile << (u0, float(n))
    PFile << (P_fce , float(n))

# if __name__ == "__main__":
#     main()
