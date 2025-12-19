import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element
from dolfinx import default_real_type, default_scalar_type
from dolfinx.fem import functionspace, Function, Constant, form, locate_dofs_topological, dirichletbc
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx import cpp
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting
from ufl import FacetNormal, TestFunction, TrialFunction, Measure, as_vector, curl, cross, inner, dx

# MPI
mpi_rank = MPI.COMM_WORLD.rank
mpi_size = MPI.COMM_WORLD.size

# Physical parameters
wl = 0.1
k0 = 2 * np.pi / wl
mu = 1.0
eps = 1.0
omega = k0

# Read mesh and tags
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as fmesh:
    mesh_name = "planewave_2d_1"
    mesh = fmesh.read_mesh(name=f"{mesh_name}")
    tdim = mesh.topology.dim
    mt_cell = fmesh.read_meshtags(mesh, name=f"{mesh_name}_cells")
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mt_facet = fmesh.read_meshtags(mesh, name=f"{mesh_name}_facets")

# Boundary facets
ds = Measure("ds", subdomain_data=mt_facet, domain=mesh)

# Facet normal
n = FacetNormal(mesh)

# Define finite element function space
pdegree = 3
curl_el = element("N1curl", mesh.basix_cell(), pdegree, dtype=default_real_type)
V = functionspace(mesh, curl_el)
E = TrialFunction(V)
v = TestFunction(V)

# Boundary dofs
pecdofs = locate_dofs_topological(V, 1, mt_facet.find(3))

# Dirichlet BC
E_pec = Function(V)
E_pec.interpolate(lambda x: np.zeros((2, x.shape[1])))
pecbc = dirichletbc(E_pec, dofs=pecdofs)

# Interpolate incident field
E_inc = Function(V)

def incident(x):
    I_x = np.full(x.shape[1], 0.0)
    I_y = np.exp(-1j*k0*x[0])
    return (I_x, I_y)

E_inc.interpolate(incident)

def exact(x):
    values = np.zeros((2, x.shape[1]), dtype=np.complex128)
    kx = np.sqrt(k0**2 - np.pi**2)

    values[0, :] = (
        1/(1j*omega*eps) * np.pi * np.sin(np.pi*x[1]) * np.exp(-1j*kx*x[0])
    )
    values[1, :] = (
        kx/(omega*eps) * np.cos(np.pi*x[1]) * np.exp(-1j*kx*x[0])
    )

    return values

def cross_z(a, b):
    return a[0]*b[1]-a[1]*b[0]

def cross_xy(a, b):
    return as_vector((a[1]*b, -a[0]*b))

b_inc = cross_xy(n, curl(E_inc)) - 1.0j * k0 * cross_xy(n, cross_z(n, E_inc))

# Define forms
a = form(
    1/mu * inner(curl(E), curl(v)) * dx 
    - omega**2 * eps * inner(E, v) * dx
    - inner(1j*k0/mu * cross_xy(n, cross_z(n, E)), v) * ds(1) # determine mathematically what is this
    - inner(1j*k0/mu * cross_xy(n, cross_z(n, E)), v) * ds(2)
)
A = assemble_matrix(a, bcs=[pecbc])
A.assemble()

L = form(
    inner(1/mu * b_inc, v) * ds(1)
)
b = assemble_vector(L)
apply_lifting(b, [a], bcs=[[pecbc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Solve
solver = PETSc.KSP().create(MPI.COMM_WORLD)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setOperators(A)

Eh = Function(V)
solver.solve(b, Eh.x.petsc_vec)

D = functionspace(mesh, ("DG", pdegree, (2,)))
Eh_dg = Function(D)
Eh_dg.interpolate(Eh)

with VTXWriter(mesh.comm, "Eh.bp", Eh_dg) as vtx:
    vtx.write(0.0)

E_inc_dg = Function(D)
E_inc_dg.interpolate(E_inc)

with VTXWriter(mesh.comm, "E_inc.bp", E_inc_dg) as vtx:
    vtx.write(0.0)
