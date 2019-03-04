from fenics import *
from cbc.twist.linear_homogenization import LinearHomogenization, \
    function_from_cell_function, PeriodicBoundary
from cbc.twist.material_models import *


class LinearMicroProblem(LinearHomogenization):
    def __init__(self):
        LinearHomogenization.__init__(self)
        n = 80
        self._mesh = UnitSquareMesh(n, n)
        self._pbc = PeriodicBoundary(self._mesh)

    def mesh(self):
        return self._mesh

    def periodic_boundaries(self):
        return self._pbc

    def create_elasticity_tensor(self, u):
        I = Identity(2)
        lmbda = 1e4
        mu = [1e3, 1e4]
        # mu = [1e5, 1e3]
        mu_const = 1e3

        subdomains = MeshFunction('size_t', self._mesh,
                                  self._mesh.topology().dim())
        subdomains.set_all(0)
        # right = AutoSubDomain(lambda x: x[0] > .5)
        right = AutoSubDomain(lambda x: pow(x[0] - 0.5, 2) +
                                        pow(x[1] - 0.5, 2) < 0.04)
        right.mark(subdomains, 1)

        # W = VectorFunctionSpace(self._mesh, 'DG', 0)
        # u0 = project(Constant((0.0,0.0)), W)

        # refactorize this into separate function
        mu_f = function_from_cell_function(mu, subdomains)
        mat = neoHookean({'half_nkT': mu_f, 'bulk': lmbda})
        # self.material = neoHookean({'half_nkT': mu_const, 'bulk': lmbda})

        # V = fenics.VectorFunctionSpace(self.mesh(), 'CG', 1)
        # u = fenics.Function(V)
        A = mat.elasticity_tensor(u)
        # i, j, k, l = indices(4)
        # I = Identity(2)
        # A_ijkl = lmbda*I[i,j]*I[k,l] - mu_f*(I[i,k]*I[j,l] + I[i,l]*I[j,k])
        # A = as_tensor(A_ijkl, (i,j,k,l))

        self.subdomains = subdomains
        self.A = A
        return A



linHom = LinearMicroProblem()
linHom.parameters["plot_solution"] = False
linHom.parameters["output_dir"] = "output/homogenization"

V = VectorFunctionSpace(linHom.mesh(), 'CG', 2)
u0_exp = Expression(("x[0]*(lmbda-1.0)","x[1]*(1.0/lmbda - 1.0)"),
                        lmbda = 1.5, degree=1)
u0 = project(u0_exp, V)
# u0 = Function(V)
linHom.create_elasticity_tensor(u0)

for t in range(1):
    linHom.solve()

    plot(linHom.correctors_chi((0,0)), mode='displacement',
         title='chi_00')
    plot(linHom.correctors_chi((1,0)), mode='displacement',
         title='chi_01')
    plot(linHom.correctors_chi((0,1)), mode='displacement',
         title='chi_10')
    plot(linHom.correctors_chi((1,1)), mode='displacement',
         title='chi_11')

    # plot(linHom.subdomains, interactive=True, title='subdomains')
    #
    A_av = linHom.averaged_elasticity_tensor()
    print("A_av: ", linHom.print_elasticity_tensor(A_av))
    A_corr = linHom.corrector_elasticity_tensor()
    print("A_corr: ", linHom.print_elasticity_tensor(A_corr))
    A = linHom.homogenized_elasticity_tensor()
    print("A: ", linHom.print_elasticity_tensor(A))

    # W = VectorFunctionSpace(linHom.mesh(), 'DG', 0)
    # u0_exp.lmbda = 1.05 + 0.05*t
    # u0 = project(u0_exp, V)
    # u1 = linHom.displacement_correction(u0)
    # plot(u1, mode='displacement', title='u_1')
    # plot(u0 + u1, mode='displacement', title='u_0 + u_1')
    # linHom.create_elasticity_tensor(u0 + u1)

# linHom.material._construct_local_kinematics(u0)
# V = TensorFunctionSpace(linHom.mesh(), 'CG', 1, shape=(2,2))
# F = project(linHom.material.F, V)
# plot(F[0,0], interactive=True)
# plot(F[1,0], interactive=True)
# plot(F[0,1], interactive=True)
# plot(F[1,1], interactive=True)
#
# S = project(linHom.material.SecondPiolaKirchhoffStress(u0), V)
# plot(S[0,0], interactive=True)
# plot(S[1,0], interactive=True)
# plot(S[0,1], interactive=True)
# plot(S[1,1], interactive=True)

