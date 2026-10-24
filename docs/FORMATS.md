# Data-format summary

## QCFD3D1 explicit-face mesh

Used by `sphere3d` for tetrahedral, wedge, hexahedral, and general polyhedral
control volumes.  It stores cells, oriented owner/neighbour faces, cell-face
adjacency, and gradient terms.  The binary declarations are in
`solvers/3d/mesh_format.h`.

## QCFDNC1 connectivity companion

Stores points, tetrahedron node IDs, node-to-cell CSR, boundary triangles, and
source zone IDs for node gradients, point location, and NNLCI interpolation.
See `solvers/3d/NNLCI_CONNECTIVITY_FORMAT.md`.

## QCFDNG1 node-gradient data

Stores adaptive node weights, tetrahedral shape gradients, and face-node data.
See `solvers/3d/NODE_GRADIENT_FORMAT.md`.

## QCFDMG1 agglomeration hierarchy

Stores deterministic coarse membership and volume weights.  See
`solvers/3d/AGGLOMERATION_FORMAT.md`.  The current solver uses it for residual
smoothing, not a full coarse-grid finite-volume solve.

## QCFDAM1 procedural Cartesian AMR

Used by `amr3d`.  `header.bin` defines counts and offsets; the directory holds
leaf blocks, 2:1 interfaces, boundary sides, agglomerated controls,
procedural-to-control mapping, cut-face exceptions, wall fragments,
source-triangle mapping, source points/triangles, and wall distance.  Struct
definitions are in `solvers/3d/amr_mesh_format.h`.

## State files

`state.bin` is little-endian float64 AoS with six values per control:

```text
rho, rho*u, rho*v, rho*w, rho*E, rho*nutilde
```

It contains no mesh ID or physics header.  Always retain the mesh manifest,
result JSON, method JSON, and command/settings with the state.
