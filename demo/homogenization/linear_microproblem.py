from fenics import *
from cbc.twist.linear_homogenization import LinearHomogenization, \
    function_from_cell_function
from cbc.twist.material_models import *
import itertools


class LinearMicroProblem(LinearHomogenization):
    def __init__(self, dim=2):
        LinearHomogenization.__init__(self)
        n = 8
        self._mesh = UnitSquareMesh(n, n) if dim == 2 else UnitCubeMesh(n, n, n)

        lmbda = 1e4
        mu = [1e3, 1e5]

        subdomains = MeshFunction('size_t', self._mesh,
                                  self._mesh.topology().dim())
        subdomains.set_all(0)
        self.subdomains = subdomains
        # inclusion = AutoSubDomain(lambda x: pow(x[0] - 0.5, 2) +
        #                                 pow(x[1] - 0.5, 2) < 0.04)
        # inclusion.mark(subdomains, 1)
        right = AutoSubDomain(lambda x: x[1] < 0.5)
        right.mark(subdomains, 1)

        # refactorize this into separate function
        mu_f = function_from_cell_function(mu, subdomains)
        self.nonlin_mat = neoHookean({'half_nkT': mu_f, 'bulk': lmbda})

    def mesh(self):
        return self._mesh

    def create_elasticity_tensor(self, u):
        A = self.nonlin_mat.elasticity_tensor(u)

        self.A = A
        return A


def main():
    linHom = LinearMicroProblem()
    linHom.parameters["plot_solution"] = False
    linHom.parameters["save_solution"] = False

    output_dir = "output/homogenization/"

    V = VectorFunctionSpace(linHom.mesh(), 'CG', 2)
    u0_exp = Expression(("x[0]*(lmbda-1.0)","x[1]*(1.0/lmbda - 1.0)"),
                            lmbda = 1.5, degree=1)
    u0 = project(u0_exp, V)
    # u0 = Function(V)
    linHom.create_elasticity_tensor(u0)

    linHom.solve()
    A_av = linHom.averaged_elasticity_tensor()
    print("A_av: ", linHom.print_elasticity_tensor(A_av))
    A_corr = linHom.corrector_elasticity_tensor()
    print("A_corr: ", linHom.print_elasticity_tensor(A_corr))
    A = linHom.homogenized_elasticity_tensor()
    print("A: ", linHom.print_elasticity_tensor(A))

    for (i, j) in itertools.product(range(2), range(2)):
        chiFile = XDMFFile(output_dir + "chi_({},{}).xdmf".format(i, j))
        chiFile.write(linHom.correctors_chi((i,j)))


if __name__ == "__main__":
    main()
