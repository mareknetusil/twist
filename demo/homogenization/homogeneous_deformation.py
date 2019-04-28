from fenics import *
from cbc.twist.linear_homogenization import LinearHomogenization, \
    function_from_cell_function
from cbc.twist.material_models import *
import itertools
import numpy as np
from linear_microproblem import LinearMicroProblem


# def main():
linHom = LinearMicroProblem()
linHom.parameters["plot_solution"] = False
linHom.parameters["save_solution"] = False

output_dir = "output/homogenization/"

V = VectorFunctionSpace(linHom.mesh(), 'CG', 2)
T = TensorFunctionSpace(linHom.mesh(), 'CG', 1)
# u0_exp = Expression(("x[0]*(lmbda-1.0)","x[1]*(1.0/lmbda - 1.0)"),
#                     lmbda = 1.0, degree=1)
u0_exp = Expression(("c_xx*x[0]+c_xy*x[1]", "c_yx*x[0]+c_yy*x[1]"),
                    c_xx = 0.0, c_xy = 0.0, c_yx = 0.0, c_yy = 0.0, degree = 1)

u0 = project(u0_exp, V)
P = linHom.nonlin_mat.first_pk_stress(u0)
# u0 = Function(V)

F = 0.0
dF = 50.0

for n in range(10):
    F_rhs = np.array([F, 0.0, 0.0, 0.0])

    uFile = XDMFFile(output_dir + "u.xdmf")
    PFile = XDMFFile(output_dir + "P.xdmf")

    for n in range(100):
        linHom.create_elasticity_tensor(u0)
        linHom.solve()

        A = linHom.homogenized_elasticity_tensor()

        for (i, j) in itertools.product(range(2), range(2)):
            chiFile = XDMFFile(output_dir + "chi_({},{}).xdmf".format(i, j))
            chiFile.write(linHom.correctors_chi((i,j)))

        A_mat = np.array([[A[0,0,0,0].value(), A[0,0,1,1].value(), A[0,0,0,1].value(), A[0,0,1,0].value()],
                          [A[1,1,0,0].value(), A[1,1,1,1].value(), A[1,1,0,1].value(), A[1,1,1,0].value()],
                          [A[0,1,0,0].value(), A[0,1,1,1].value(), A[0,1,0,1].value(), A[0,1,1,0].value()],
                          [A[1,0,0,0].value(), A[1,0,1,1].value(), A[1,0,0,1].value(), A[1,0,1,0].value()]
                         ])

        P_rhs = np.array([assemble(P[0,0]*dx), assemble(P[1,1]*dx),
                          assemble(P[0,1]*dx), assemble(P[1,0]*dx)])
        print(P_rhs)

        G = np.linalg.solve(A_mat, F_rhs - P_rhs)
        print(G)

        u0_exp.c_xx = G[0]
        u0_exp.c_xy = G[2]
        u0_exp.c_yx = G[3]
        u0_exp.c_yy = G[1]

        u_inc = project(u0_exp, V)
        u_corr = linHom.displacement_correction(u_inc)
        u0 = u0 + u_inc + u_corr
        u0 = project(u0, V)

        P = linHom.nonlin_mat.first_pk_stress(u0)
        P_fce = project(P, T)

        inc_norm = assemble(inner(u_inc, u_inc)*dx)
        if inc_norm < 1e-12:
            uFile.write(u0, n)
            PFile.write(P_fce, n)
            break

    F += dF

# if __name__ == "__main__":
#     main()
