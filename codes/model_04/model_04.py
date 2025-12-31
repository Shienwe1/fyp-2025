import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx.fem import functionspace, Constant, Function, form
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from ufl import FacetNormal, TestFunction, TrialFunction, Measure, as_vector, curl, inner, dx, ds


def cross_z(a, b):
    return a[0]*b[1]-a[1]*b[0]


def cross_xy(a, b):
    return as_vector((a[1]*b, -a[0]*b))


# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Read mesh and tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "maxwell/circular_scatterer_2d_1"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")

# Physical parameters
lmbda = 0.1  # wavelength
k0 = 2*np.pi/lmbda  # wave number
mu_val = 1.0
eps_val = 1.0
omega_val = np.sqrt(k0**2/(eps_val*mu_val))

mu = Constant(mesh, mu_val)
eps = Constant(mesh, eps_val)
omega = Constant(mesh, omega_val)

# Boundary facets
ds = Measure("ds", subdomain_data=mt_facet, domain=mesh)

# Facet normal
n = FacetNormal(mesh)

# Define finite element function space
pdegree = 3
V = functionspace(mesh, ("N1curl", pdegree))
E = TrialFunction(V)
v = TestFunction(V)

# Interpolate incident field
def incident(x):
    I_x = np.full(x.shape[1], 0.0)
    I_y = np.exp(-1j*k0*x[0])
    return (I_x, I_y)


E_inc = Function(V)
E_inc.interpolate(incident)

# Define forms
a = form(
    1/mu_val * inner(curl(E), curl(v)) * dx
    - omega_val**2 * eps_val * inner(E, v) * dx
    - 1/mu_val * inner(1j*k0 * cross_xy(n, cross_z(n, E)), v) * ds(1)
)
A = assemble_matrix(a)
A.assemble()

L = form(
    - inner(cross_xy(n, curl(E_inc)), v) * ds(1)
    - 1/mu_val * inner(1j*k0 * cross_xy(n, cross_z(n, E_inc)), v) * ds(1)
)
b = assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Solve
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setOperators(A)

Eh = Function(V)
solver.solve(b, Eh.x.petsc_vec)

Et = Function(V)
Et.x.array[:] = E_inc.x.array[:] + Eh.x.array[:]

D = functionspace(mesh, ("DG", pdegree, (2,)))
Et_dg = Function(D)
Et_dg.interpolate(Et)

with VTXWriter(mesh.comm, "Et.bp", Et_dg) as vtx:
    vtx.write(0.0)
