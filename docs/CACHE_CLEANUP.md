# Cache cleanup and retention plan

The GitHub source package is not a restart archive.  Before deleting the large
CFD cache, choose one of the following retention levels.

## Level A: source-only reproducibility

Keep this release directory/tarball, the NASA/RAE source-data location and
checksums, and the closed CRM surface.  All large meshes and solutions must be
rebuilt/rerun.  This is the smallest option but loses the current numerical
trajectory and exact checkpoints.

Essential small private artifact:

```text
<CFD_CACHE>/workspaces/quick_cfd_for_nnlci_airfoil_opt_case/
  local_development/crm10m/geometry/CRM_full_closed.npz
```

Do not publish external geometry unless its redistribution terms are confirmed.

## Level B: result-preserving archive

In addition to Level A, keep one state and its complete metadata per useful
branch:

```text
2-D NACA final:
  verification/cutcell_order2_naca_final/

RAE2822 LF1:
  verification/2026-10-24/rae2822_hybrid_lf1/

CRM tetra checkpoint:
  runs/crm_order2_block_sa1_reversal_180s/best_state.bin
  runs/crm_order2_block_sa1_reversal_180s/{result.json,history.csv}

CRM AMR wall-refined source checkpoint:
  verification/2026-10-24/crm_amr_wallref_order2_blend05_continue2/

CRM AMR 21.5M checkpoint and diagnostics:
  verification/2026-10-24/crm_amr_upperwing_l10/
    prolongated_cfl1_reversal_120s/
```

A state is not usable without the exact mesh ordering.  Either retain the
matching mesh directories below or prove deterministic reconstruction with
hashes before deleting them:

```text
crm10m/raw/                       # tetra mesh/connectivity/node-gradient
crm10m/amr_wall_refined/          # source of the AMR prolongation
crm10m/amr_upperwing_l10/         # current 21.5M target mesh
```

## Level C: development-evidence archive

Also retain CPU/CUDA one-step comparisons, geometry/uniform audits, failed-
method reports needed for a paper/review, and the tetra volume inspector.  The
curated locations are listed by `CFD_WORK_INDEX_2026-10-24` outside this source
package.

## Safe deletion order

1. Verify the release tarball and checksum manifest in a second location.
2. Create and verify the selected private restart archive.
3. Confirm no solver process is using a directory.
4. Delete rejected probes and superseded continuation stages first.
5. Delete reproducible builds.
6. Delete large meshes only after deterministic reconstruction is tested.
7. Delete external source archives only if their URL and checksum are retained
   and the source remains available.

No deletion is performed by this package.
