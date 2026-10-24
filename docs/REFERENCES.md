# References and external projects

This file records the literature, standards, benchmark sources, and software
projects that informed the implementation or validation workflow.  A citation
means that the source motivated or defined a method; it does **not** mean this
code reproduces every detail of that publication.

## Implicit and explicit pseudo-time integration

- M. J. Wright, G. V. Candler, and M. Prampolini, “A Data-Parallel
  Lower-Upper Relaxation Method for the Navier–Stokes Equations,” *AIAA
  Journal*, 34(7), 1371–1377, 1996.
  [doi:10.2514/3.13242](https://doi.org/10.2514/3.13242).
- P. Zehner and A. Hashimoto, “Acceleration of the data-parallel lower-upper
  relaxation time-integration method on GPU for an unstructured CFD solver,”
  *Computers & Fluids*, 256, 105842, 2023.
  [doi:10.1016/j.compfluid.2023.105842](https://doi.org/10.1016/j.compfluid.2023.105842).
- S. Gottlieb and C.-W. Shu, “Total Variation Diminishing Runge–Kutta
  Schemes,” *Mathematics of Computation*, 67(221), 73–85, 1998.
  [doi:10.1090/S0025-5718-98-00913-2](https://doi.org/10.1090/S0025-5718-98-00913-2).

The explicit-face 3-D solver uses a project-specific matrix-free DP-LUR
approximation with optional five-variable Euler block action and a separate SA
scalar update.  The procedural-AMR solver uses a cheaper scalar-spectral
neighbour action.  Neither path should be described as a verbatim
implementation of Wright et al. or FaSTAR.

## Convective fluxes, low-Mach treatment, and reconstruction

- E. F. Toro, M. Spruce, and W. Speares, “Restoration of the Contact Surface
  in the HLL-Riemann Solver,” *Shock Waves*, 4, 25–34, 1994.
  [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629).
- M.-S. Liou, “A Sequel to AUSM, Part II: AUSM+-up for All Speeds,” *Journal
  of Computational Physics*, 214(1), 137–170, 2006.
  [doi:10.1016/j.jcp.2005.09.020](https://doi.org/10.1016/j.jcp.2005.09.020).
- E. Turkel, “Preconditioned Methods for Solving the Incompressible and Low
  Speed Compressible Equations,” *Journal of Computational Physics*, 72(2),
  277–298, 1987.
  [doi:10.1016/0021-9991(87)90084-2](https://doi.org/10.1016/0021-9991(87)90084-2).
- T. J. Barth and D. C. Jespersen, “The Design and Application of Upwind
  Schemes on Unstructured Meshes,” AIAA Paper 89-0366, 1989.
  [doi:10.2514/6.1989-366](https://doi.org/10.2514/6.1989-366).

The code contains HLLC and AUSM+-up-family flux paths, Turkel-style steady
low-Mach preconditioning, least-squares/Green–Gauss gradients, and a
Barth–Jespersen-style multidimensional limiter.  The exact algebra and safety
limits are the implementation in this repository.

## Turbulence and molecular transport

- P. R. Spalart and S. R. Allmaras, “A One-Equation Turbulence Model for
  Aerodynamic Flows,” AIAA Paper 92-0439, 1992.
  [doi:10.2514/6.1992-439](https://doi.org/10.2514/6.1992-439).
- S. R. Allmaras, F. T. Johnson, and P. R. Spalart, “Modifications and
  Clarifications for the Implementation of the Spalart–Allmaras Turbulence
  Model,” ICCFD7-1902, 2012.
  [paper](https://www.iccfd.org/iccfd7/assets/pdf/papers/ICCFD7-1902_paper.pdf).
- NASA Langley Turbulence Modeling Resource,
  [Spalart–Allmaras and SA-neg definitions](https://tmbwg.github.io/turbmodels/spalart.html).

The retained model is SA-neg with the documented Note-1c treatment and
mechanical viscous coupling.  Heat conduction, transition models, k-omega/SST,
DES, and LES are not implemented.

## Cartesian cut cells and small-cell stabilization

- P. Colella, D. T. Graves, B. J. Keen, and D. Modiano, “A Cartesian Grid
  Embedded Boundary Method for Hyperbolic Conservation Laws,” *Journal of
  Computational Physics*, 211(1), 347–366, 2006.
  [doi:10.1016/j.jcp.2005.05.026](https://doi.org/10.1016/j.jcp.2005.05.026).
- M. Berger and A. Giuliani, “A State Redistribution Algorithm for Finite
  Volume Schemes on Cut Cell Meshes,” *Journal of Computational Physics*, 428,
  109820, 2021.
  [doi:10.1016/j.jcp.2020.109820](https://doi.org/10.1016/j.jcp.2020.109820),
  [open preprint](https://arxiv.org/abs/2005.05734).

These works provide embedded-boundary and small-cell context.  The current 3-D
CRM path uses project-specific exact triangle clipping, boundary-integral
moments, largest-open-aperture agglomeration, procedural 8x8x8 blocks, and
conservative 2:1 subfaces.  The older 2-D path also contains an SRD option;
the production agglomerated 2-D and 3-D paths do not claim to reproduce the
papers above line for line.

## Benchmark geometries and comparison data

- NASA, [Common Research Model](https://www.nasa.gov/common_research_model/).
- AIAA Drag Prediction Workshop,
  [DPW-4 CRM geometry and reference dimensions](https://www.aiaa-dpw.org/Workshop4/DPW4-geom.html)
  and [current geometry portal](https://www.aiaa-dpw.org/geometry.html).
- J. C. Vassberg et al., “Summary of the Fourth AIAA CFD Drag Prediction
  Workshop,” AIAA Paper 2010-4547, 2010.
  [NASA NTRS copy](https://ntrs.nasa.gov/api/citations/20100025833/downloads/20100025833.pdf).
- P. H. Cook, M. A. McDonald, and M. C. P. Firmin, “Aerofoil RAE 2822 —
  Pressure Distributions, and Boundary Layer and Wake Measurements,” AGARD
  Report AR-138, 1979.  NASA's NPARC archive retains an
  [RAE2822 validation page](https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf04/raetaf04.html).
- M. Drela, “XFOIL: An Analysis and Design System for Low-Reynolds-Number
  Airfoils,” in *Low Reynolds Number Aerodynamics*, Lecture Notes in
  Engineering 54, 1989.  The NACA4412 comparison uses XFOIL as an independent
  numerical reference, not as experiment.

NASA CRM archives and the project owner's RAE2822 NPZ topology files are
external data and are not redistributed in this source package.  The case
runbooks record the required filenames, transformations, and checksums where
available.

## Data standards and software projects

- [CGNS standard and SIDS](https://cgns.org/standard.html): source format for
  the retained NASA tetrahedral CRM importer.
- [HDF5](https://www.hdfgroup.org/solutions/hdf5/): storage layer commonly
  used by CGNS files.
- [CUDA documentation](https://docs.nvidia.com/cuda/): GPU programming and
  runtime model used by the CUDA targets.
- [OpenMP specifications](https://www.openmp.org/specifications/): CPU
  parallelism used by selected preprocessing/build paths.
- [CMake](https://cmake.org/), [NumPy](https://numpy.org/),
  [Matplotlib](https://matplotlib.org/), and [Shapely](https://shapely.readthedocs.io/):
  build, array, visualization, and selected geometry tooling.

Project-specific `QCFD3D1`, `QCFDAM1`, `QCFDNC1`, and `QCFDMG1` formats are
documented in this repository and are not external standards.
