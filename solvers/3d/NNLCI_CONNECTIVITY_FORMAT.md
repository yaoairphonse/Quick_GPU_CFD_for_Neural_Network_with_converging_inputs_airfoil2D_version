# NNLCI connectivity companion format

`connectivity.bin` stores the imported topology that is not required in the
finite-volume solver's hot loop. It complements `mesh.bin`; it does not replace
the owner/neighbour face table or alter solver cell ordering.

All integers and floating-point values are little-endian. Version 1 is limited
to tetrahedral volume cells and triangular boundary faces.

## Header

The 96-byte header uses the packed layout `<8sII10Q`:

| Field | Type | Meaning |
|---|---:|---|
| magic | 8 bytes | `QCFDNC1\0` |
| version | uint32 | `1` |
| endian marker | uint32 | `0x01020304` |
| point count | uint64 | Number of volume-mesh points |
| cell count | uint64 | Number of tetrahedra |
| boundary-face count | uint64 | Number of boundary triangles |
| incidence count | uint64 | Four times the cell count |
| five table offsets | uint64 each | Absolute byte offsets listed below |
| file bytes | uint64 | Exact expected file length |

## Tables

The five tables are contiguous and appear in this order:

1. `points[point_count][3]`: float64 normalized coordinates.
2. `cell_nodes[cell_count][4]`: int32 point identifiers. Cell order is exactly
   the solver/checkpoint order.
3. `node_offsets[point_count + 1]`: uint64 CSR offsets.
4. `node_cells[incidence_count]`: int32 cell identifiers. Together with
   `node_offsets`, this is the exact reverse of `cell_nodes`.
5. `boundary[boundary_face_count]`: three int32 point identifiers followed by
   int32 native boundary-condition and patch identifiers.

Native boundary-condition identifiers are `1` for a no-slip wall, `2` for a
farfield boundary, and `3` for a symmetry plane. Patch identifiers preserve the
source CGNS boundary-zone numbering.

The companion deliberately contains no solver state, interpolation weights, or
spatial acceleration structure. Those can be regenerated for a chosen NNLCI
algorithm without invalidating this topology contract.

Run the full consistency audit with:

```bash
python local_development/sphere3d/validate_connectivity.py \
  /path/to/connectivity.bin --report /path/to/connectivity_audit.json
```
