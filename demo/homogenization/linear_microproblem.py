from cbc.twist import *

class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 1) and near(x[1], 0)) or
                          (near(x[0], 1) and near(x[1],
                                                  1)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1
            y[1] = x[1] - 1
        elif near(x[0], 1):
            y[0] = x[0] - 1
            y[1] = x[1]
        elif near(x[1], 1):
            y[0] = x[0]
            y[1] = x[1] - 1
        else:
            y[0] = 1000
            x[0] = 1000

class LinearMicroProblem(LinearHomogenization):
    def __init__(self):
        LinearHomogenization.__init__(self)
        n = 80
        self._mesh = UnitSquareMesh(n, n)
        self._pbc = PeriodicBoundary()
        self._A = self.create_elasticity_tensor()

    def mesh(self):
        return self._mesh

    def elasticity_tensor(self):
        return self._A

    def periodic_boundaries(self):
        return self._pbc

    def create_elasticity_tensor(self):
        I = Identity(2)
        lmbda = 1e5
        mu = [1e3, 1e6]
        # mu = [1e5, 1e3]
        mu_const = 1e3

        subdomains = CellFunction('size_t', self._mesh)
        subdomains.set_all(0)
        # right = AutoSubDomain(lambda x: x[0] > .5)
        right = AutoSubDomain(lambda x: pow(x[0] - 0.5, 2) +
                                        pow(x[1] - 0.5, 2) < 0.04)
        right.mark(subdomains, 1)

        W = VectorFunctionSpace(self._mesh, 'DG', 0)
        u0 = project(Constant((0.0,0.0)), W)

        # refactorize this into separate function
        mu_f = function_from_cell_function(mu, subdomains)
        self.material = neoHookean({'half_nkT': mu_f, 'bulk': lmbda})
        # self.material = neoHookean({'half_nkT': mu_const, 'bulk': lmbda})
        A = self.material.FirstElasticityTensor(u0)
        # i, j, k, l = indices(4)
        # I = Identity(2)
        # A_ijkl = lmbda*I[i,j]*I[k,l] - mu_f*(I[i,k]*I[j,l] + I[i,l]*I[j,k])
        # A = as_tensor(A_ijkl, (i,j,k,l))

        self.subdomains = subdomains
        return A

linHom = LinearMicroProblem()
linHom.solve()

plot(linHom.correctors_chi((0,0)), interactive=True, mode='displacement',
     title='chi_00')
plot(linHom.correctors_chi((1,0)), interactive=True, mode='displacement',
     title='chi_01')
plot(linHom.correctors_chi((0,1)), interactive=True, mode='displacement',
     title='chi_10')
plot(linHom.correctors_chi((1,1)), interactive=True, mode='displacement',
     title='chi_11')

plot(linHom.subdomains, interactive=True, title='subdomains')

B_av = linHom.averaged_elasticity_tensor()
print "B_av: ", linHom.print_elasticity_tensor(B_av)
B_corr = linHom.corrector_elasticity_tensor()
print "B_corr: ", linHom.print_elasticity_tensor(B_corr)
B = linHom.homogenized_elasticity_tensor()
print "B: ", linHom.print_elasticity_tensor(B)

# W = VectorFunctionSpace(linHom.mesh(), 'DG', 0)
V = VectorFunctionSpace(linHom.mesh(), 'CG', 1)
# u0 = project(Expression(("x[0]*(lmbda-1.0)","x[1]*(1.0/lmbda - 1.0)"),
#                         lmbda = 1.5, degree=1), V)
u0 = project(Expression(("x[0]*(1.0/lmbda-1.0)","x[1]*(lmbda - 1.0)"),
                        lmbda = 1.2, degree=1), V)
u1 = linHom.displacement_correction(u0)
plot(u1, interactive=True, mode='displacement', title='u_1')
plot(u0 + u1, interactive=True, mode='displacement', title='u_0 + u_1')

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

