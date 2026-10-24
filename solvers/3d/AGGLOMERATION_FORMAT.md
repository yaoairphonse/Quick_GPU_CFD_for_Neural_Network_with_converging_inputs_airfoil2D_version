# QCFDMG1 aggregation hierarchy

`QCFDMG1` is a little-endian binary parent hierarchy for future geometric or
full-approximation multigrid operations. It is separate from the solver mesh
and does not change fine-cell numbering.

The 40-byte file header contains an eight-byte `QCFDMG1\0` magic value, four
32-bit values (`version`, endian marker, level count, merges per level), and two
64-bit values (fine-cell count and total file bytes). Each fine-to-coarse
transition then contains a 16-byte pair of 64-bit child and parent counts,
followed by one signed 32-bit parent identifier per child.

Parent identifiers are dense in `[0, parent_count)`. Every parent has at least
one child. The deterministic builder merges only aggregates connected by an
interior finite-volume face. Aggregates carrying different nonzero boundary
condition masks are not merged; an interior aggregate may join a boundary
aggregate. Two pair-matching passes per transition normally produce a
coarsening ratio near four.

The file currently supplies topology only. A production FAS implementation
must still construct coarse geometry and flux operators, define conservative
restriction and correction prolongation, preserve positive density and
internal energy, and treat wall distance and SA source terms consistently.

Build and validate with:

```bash
build_agglomeration --mesh mesh.bin --out agglomeration_3level.bin \
  --levels 3 --merges-per-level 2
python3 validate_agglomeration.py agglomeration_3level.bin
```
