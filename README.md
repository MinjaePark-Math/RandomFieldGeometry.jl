# RandomFieldGeometry.jl

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19582531.svg)](https://doi.org/10.5281/zenodo.19582531)

`RandomFieldGeometry.jl` is a Julia package for generating random fields such as Gaussian free fields and log-correlated random fields, studying random geometry and related metric structures, and exporting or visualizing the resulting objects.

[![Live Demo](https://img.shields.io/badge/Live_Demo-aub.ie%2Flfpp3d-blue.svg)](https://aub.ie/lfpp3d)

<p align="center">
  <a href="https://aub.ie/lfpp3d">
    <img src="assets/demo.png" alt="3D Liouville First-Passage Percolation Geometry" width="600">
  </a>
</p>

At present, the main implemented workflow focuses on first-passage percolation (FPP) geodesics for exponentials of log-correlated random fields in 2D and 3D, with the 2D case closely related to Liouville quantum gravity (LQG) metrics. The public interface is centered around:

- `run_lfpp_simulation` for data generation,
- `export_vtk` for ParaView / VTK export,
- and a minimal GLMakie viewer through `interactive_viewer`.

The package also includes a square-domain imaginary-geometry workflow based on the Miller-Sheffield flow-line formalism. The current maintained path is the chordal square fan:

- `sample_chordal_square_ig_field` for efficient square Dirichlet IG fields,
- `free_square_gff` for efficient free-boundary square GFF samples (using a mean-zero convention),
- `trace_flowline`, `trace_angle_fan`, and `trace_sle_fan` for proxy path tracing on top of those sampled fields,
- and `plot_flowlines` for quick 2D Makie visualization when Makie is available.

For square chordal fields, `sample_chordal_square_ig_field(...; boundary_mode=:zero_force)` now supports both `:zero_force` and `:zero_boundary` half-plane boundary conventions via the coordinate-change rule from Imaginary Geometry I.

The repository is organized as a standard Julia package:

- `src/RandomFieldGeometry.jl` is the package entry point.
- `src/RandomFieldGenerators.jl` is the random-field layer.
  It now owns both the core Dirichlet samplers and the square-domain IG field samplers in `src/RandomFieldGenerators/`.
- `src/Flowlines.jl` is the curve-generation layer built on top of those sampled fields.
- `src/Pathfinders.jl` contains the current shortest-path solver, with optional GPU backends via CUDA and Metal.
- `examples/` contains runnable example scripts.
- `test/` contains a basic regression suite.

The codebase is designed with scale and performance in mind. As one reference point for the current LFPP workflow, on my local machine with an NVIDIA RTX 3090 Ti using CUDA, computing geodesics in a `1024^3` box from `256` sampled starting points takes under `2` minutes.

## Simulation Highlight: Confluence in 3D

[![Live Demo](https://img.shields.io/badge/Live_Demo-aub.ie%2Fconfluence3d-blue.svg)](https://aub.ie/confluence3d)

<p align="center">
  <a href="https://aub.ie/confluence3d">
    <img src="assets/confluence.png" alt="Confluence in 3D LFPP geometry" width="600">
  </a>
</p>

One striking feature suggested by these simulations is geodesic confluence in 3D, and possibly in higher dimensions, for LFPP-type metric geometries. To the best of my knowledge, this phenomenon has not yet been explicitly conjectured in the literature in this form, so the confluence demo is intended as visual evidence and motivation for further investigation.

## Citation

If you use `RandomFieldGeometry.jl` in research or course materials, please cite the Zenodo archive:

> Park, M. (2026). *RandomFieldGeometry.jl*. Zenodo. [https://doi.org/10.5281/zenodo.19582531](https://doi.org/10.5281/zenodo.19582531)

```bibtex
@misc{Park2026RandomFieldGeometry,
  author = {Park, Minjae},
  title = {RandomFieldGeometry.jl},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.19582531},
  url = {https://doi.org/10.5281/zenodo.19582531}
}
```

## Future Implementations

- Geometry of Gaussian free field in 2D and 3D, including percolation
- Square subdivision models
- Random field generators for other models or boundary conditions
- Other conformal random geometry in 2D and 3D

**Feature Requests & Collaboration:**
I am very open to implementing additional models and related geometry workflows. Please feel free to [reach out via email](mailto:minjaep@auburn.edu) for feature requests, questions, or research collaboration.

## Installation

To install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MinjaePark-Math/RandomFieldGeometry.jl")
```

## Optional GLMakie Viewer

The core package does not require Makie. The minimal interactive viewer is loaded through a Julia package extension when these optional packages are available in the current environment:

```julia
using Pkg
Pkg.add(["Colors", "GeometryBasics", "GLMakie"])
```

Load them in the same Julia session before calling `interactive_viewer`.

## Quick start

For a basic 3D LFPP simulation and VTK export:

```julia
using RandomFieldGeometry

sim = run_lfpp_simulation(128, 1.0f0; dim=3)
export_vtk(sim.distances, "vtk_results/example"; path_step=8)
```

For a 2D Makie viewer:

```julia
using Colors
using GeometryBasics
using GLMakie
using RandomFieldGeometry

sim = run_lfpp_simulation(128, 0.8f0; dim=2)
interactive_viewer(sim.distances; path_step=8)
```

For a free-boundary square GFF sample:

```julia
using RandomFieldGeometry

h = free_square_gff(257, 20260420; T=Float64)
```

The scripts in [examples](examples/) provide minimal entry points for Makie viewing, ParaView export, and the maintained square-domain IG fan demo:

- `examples/run_glmakie.jl`
- `examples/run_paraview.jl`
- `examples/run_sle_fan.jl`

## Imaginary Geometry Quick Start

The square-domain imaginary-geometry routines follow the conventions of:

- Miller and Sheffield, *Imaginary Geometry I: Interacting SLEs*, [arXiv:1201.1496](https://arxiv.org/abs/1201.1496)
- Miller and Sheffield, *Imaginary Geometry IV: interior rays, whole-plane reversibility, and space-filling trees*, [arXiv:1302.4738](https://arxiv.org/abs/1302.4738)

Current status:
The present square IG visualizers are still proxy constructions rather than exact continuum IG flow lines. Structurally, the package treats an IG field as a random field object: the sampler stores the random and deterministic pieces on the square grid, and the flowline layer traces curves with respect to that sampled field. For chordal square fields, the code first pulls the half-plane boundary data (`0` or `±λ`) onto the square boundary, adds the coordinate-change correction from `h_S = h_H ∘ ψ - χ arg ψ'`, harmonically extends that boundary data with Dirichlet sine transforms, then adds a zero-boundary GFF. The tracer follows small steps in the direction `exp(i h / χ)` of the interpolated field. The optional multiscale backend gives a vanishing-regularization ladder below the display mesh, but it should also be interpreted as a proxy rather than an exact SLE simulation.

Flowline angles use the square convention `θ = 0` north, `θ = π/2` west, `θ = -π/2` east, and `θ = π` south. On a single fixed grid, making `ds` much smaller than the mesh size only resolves the same bilinear proxy field more accurately; for a better IG approximation, prefer `multiscale=true` or a finer hidden field resolution over `ds << h`.

The maintained example is the square chordal fan in `examples/run_sle_fan.jl`. Programmatically, the same workflow looks like:

```julia
using RandomFieldGeometry

ig = sample_chordal_square_ig_field(385, 2.0; boundary_mode=:zero_force, seed=20260418, T=Float64)
ds = 0.02 * ig.domain.hx
seed = complex((ig.domain.xmin + ig.domain.xmax) / 2, ig.domain.ymin + ds)
angles = range(-pi / 2, pi / 2; length=11)

fan = trace_sle_fan(
    ig,
    seed,
    angles;
    ds=ds,
    integrator=:euler,
    boundary_margin=0.0,
)
```

If you have Makie loaded, you can visualize that result with `plot_flowlines(ig, fan)`.

For the maintained black-background fan script with CLI-tunable `kappa`, grid size, angle range, `ds/h`, and optional multiscale refinement, run:

```bash
julia --project examples/run_sle_fan.jl kappa=2 grid=513 nflow=11 theta_left=pi/2 theta_right=pi/2 boundary_mode=zero_force ds_over_h=0.02
```

Useful knobs for the current IG fan workflow:

- `boundary_mode=zero_force` or `boundary_mode=zero_boundary`
- `ds_over_h=...` to choose the tracing step as a mesh fraction
- `integrator=euler` for the default step-by-step proxy trace
- `multiscale=true levels=... min_cutoff=... spectral_oversample=...` for hidden-resolution refinement

The other IG-family experiments that existed during development were intentionally removed from the example surface until they are improved and re-validated.
