# Node-gradient companion format

`node_gradient.bin` stores geometry coefficients for the opt-in
`--gradient node` solver path. It complements `mesh.bin` and
`connectivity.bin`; cell, face, and point ordering must match both files.

All values are little-endian. Version 2 is limited to tetrahedral volume
cells and triangular faces. The 80-byte header uses the native C++ layout in
`NodeGradientHeader`:

| Field | Type | Meaning |
|---|---:|---|
| magic | 8 bytes | `QCFDNG1\0` |
| version | uint32 | `2` |
| endian marker | uint32 | `0x01020304` |
| point count | uint64 | Must match `connectivity.bin` |
| cell count | uint64 | Must match both companion files |
| face count | uint64 | Must match `mesh.bin` |
| incidence count | uint64 | Four times the tetrahedron count |
| node-weight offset | uint64 | Absolute byte offset |
| cell-gradient offset | uint64 | Absolute byte offset |
| face-node offset | uint64 | Absolute byte offset |
| file bytes | uint64 | Exact expected length |

The contiguous tables are:

1. `node_weights[incidence_count]`: float64 weights aligned with the
   `node_cells` CSR entries in `connectivity.bin`.
2. `cell_gradients[cell_count][4][3]`: float64 gradients of the four linear
   tetrahedral shape functions.
3. `face_nodes[face_count][3]`: int32 node identifiers in `mesh.bin` face
   order. Interior triples are the intersection of the owner and neighbour
   tetrahedra; boundary triples are recovered from the face centroid.

`build_node_gradient` uses the natural node-to-cell stencil and an
adaptive-dimensional pseudo-Laplacian interpolation. Symmetry nodes are
restricted to at most two interpolation dimensions. The default hybrid mode
retains affine-exact weights unless a normalized weight magnitude exceeds 2;
only those extreme stencils fall back to bounded weights. The build report
records dimensional reductions, bounded nodes, weight extrema, tetrahedron
volume agreement, and an end-to-end affine-gradient audit.

The solver first forms nodal primitive values from the CSR weights. Cell
gradients use the four nodal values and shape gradients. Interior diffusion
gradients use the cell-center difference and two independent face-node
differences (Mitchell stencil). No-slip wall velocity and SA face gradients
use the wall value zero.

Build the CRM companion with:

```bash
cache_root=/path/to/CFD_cache
build_node_gradient \
  "${cache_root}/crm10m/raw/connectivity.bin" \
  "${cache_root}/crm10m/raw/mesh.bin" \
  "${cache_root}/crm10m/raw/node_gradient.bin" \
  16 "${cache_root}/crm10m/raw/node_gradient_audit.json" hybrid
```

The current 10M-cell file is approximately 1.53 GB. It remains in the ext4
cache and must not be copied into the worktree or OneDrive.
