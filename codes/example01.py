from mpi4py import MPI
import numpy as np

import ufl
from dolfinx.fem import Function, functionspace
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.mesh import create_rectangle, CellType
from dolfinx import cpp
from ufl import curl, grad, inner, dx

# Generate domain
domain = create_rectangle(
    MPI.COMM_WORLD,
    ((0.0, 0.0), (1.0, 1.0)),
    (16, 16),
    CellType.triangle
)

# Define finite element function space
V = functionspace(domain, ("N1curl", 1))

