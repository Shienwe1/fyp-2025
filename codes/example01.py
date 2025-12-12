from mpi4py import MPI
import numpy as np

import ufl
from dolfinx.fem import Function, functionspace
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import create_rectangle, CellType
from dolfinx import cpp
from ufl import curl, grad, inner, dx

# Physical parameters
wl = 0.1
k0 = 2 * np.pi / wl

# Generate domain
domain = create_rectangle(
    MPI.COMM_WORLD,
    ((0.0, 0.0), (1.0, 1.0)),
    (16, 16),
    CellType.triangle
)

# Define finite element function space
V = functionspace(domain, ("N1curl", 3))

# Interpolate incident field
E_inc = Function(V)

def incident(x):
    return (x[0]*0, np.exp(-1j*k0*x[0]))

E_inc.interpolate(incident)

D = functionspace(domain, ("DG", 3, (2,)))
E_inc_dg = Function(D)
E_inc_dg.interpolate(E_inc)

with VTXWriter(domain.comm, "E_inc.bp", E_inc_dg) as vtx:
    vtx.write(0.0)