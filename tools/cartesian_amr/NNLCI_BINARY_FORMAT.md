# QCFD NNLCI binary mesh tables

## Purpose

`QCFD_NNLCI_MESH_TABLES` is a versioned, zero-based binary format for reusing
mesh topology, point-location indices, and interpolation stencils without
reconstructing connectivity for every CFD or NNLCI run. Numeric arrays are
little-endian `.npy` files so Python can memory-map them without copying. The
JSON manifest records each table's dtype, shape, byte count, and meaning.

The current version describes two independently valid submeshes. It is marked
`solver_ready: false` because solid removal, Cartesian cut cells, and the
conservative prism/octree interface are not assembled yet.

## Explicit prism tables

- `prism_nodes`: double-precision coordinates in layer-major order.
- `prism_cell_nodes`: six node identifiers per triangular prism.
- `prism_cell_neighbors`: bottom, top, and three lateral neighbors per cell.
- `prism_node_cell_offsets` and `prism_node_cells`: node-to-cell CSR.
- `prism_node_neighbor_offsets` and `prism_node_neighbors`: node-to-node CSR.
- `prism_cell_centroids` and `prism_cell_volumes`: reusable geometry.
- `surface_*`: compact unique edge, triangle, patch, and triangle-neighbor data.
- `prism_column_bounds`, `prism_bvh_*`: an AABB tree for point location.

Neighbor `-1` is the solid wall. Neighbor `-2` is the unresolved outer prism
front; it will be replaced by a real cut-cell identifier during final assembly.
Cell and node identifiers are deterministic when surface ordering and layer
settings do not change.

The point locator first finds candidate wall-normal columns in the stored BVH,
then tests the three audited tetrahedra composing each prism. The returned four
node identifiers and barycentric weights can be used directly for nodal NNLCI
interpolation. Cell-centered methods can use the stored cell centroid and
neighbor table to construct a local least-squares stencil.

## Factorized octree tables

- `octree_block_*`: leaf origins, sizes, levels, face neighbors, surface-candidate
  flags, distance bounds, and boundary masks.
- `octree_finest_owner`: a 256-cubed integer locator for O(1) leaf lookup.

Each leaf contains a regular 8x8x8 Cartesian block. Its cell and node
connectivity are reconstructed exactly from the block identifier and local
`(i,j,k)` indices. Materializing those repeated indices would add hundreds of
megabytes without adding information. Cut cells will be stored explicitly as
exceptions in the final format.

## Final solver-ready extension

After overlap removal and interface clipping, the same directory will gain a
single global finite-volume topology:

- CSR cell-to-node and cell-to-face tables;
- CSR face-to-node tables;
- one `face_owner` and `face_neighbour` entry per physical face;
- boundary patch, face centroid, area vector, cell centroid, and volume tables;
- explicit cut-cell connectivity and a source-ID map back to the prism or
  Cartesian planning cell;
- node-to-cell and cell-neighbor CSR for NNLCI interpolation.

Where one outer prism triangle meets several Cartesian cells, it is partitioned
into matching subfaces. Every subface is evaluated once, with the prism and cut
cell as owner/neighbour. This is a conservative conformal link, not overset
interpolation or immersed-boundary forcing.

## Commands

Export and validate through the complete pipeline:

```bash
python3 local_development/backup/hybrid_mesh_research_spike/run_pipeline.py \
  --surface local_development/crm10m/geometry/CRM_full_closed.npz \
  --out local_development/backup/hybrid_mesh_research_spike/crm_first_pass \
  --target-cartesian-cells 10000000 \
  --build-interface \
  --export-nnlci-tables
```

Query a point without loading the complete tables into memory:

```bash
python3 local_development/backup/hybrid_mesh_research_spike/nnlci_table_reader.py \
  --mesh local_development/backup/hybrid_mesh_research_spike/crm_first_pass/nnlci_tables \
  --point 5.0 0.0 0.8
```
