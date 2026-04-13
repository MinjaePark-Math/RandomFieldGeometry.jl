using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

try
    @eval using CUDA
catch
end
try
    @eval using Metal
catch
end

using Colors
using GeometryBasics
using GLMakie
using RandomFieldGeometry

backend = default_backend()
println("Using backend: " * describe_backend(backend))
if @isdefined(CUDA)
    println("CUDA.functional() = $(CUDA.functional())")
end

dim = 3
N = 128
xi = 0.8f0

sim = run_lfpp_simulation(N, xi; dim=dim, backend=backend)
fig = interactive_viewer(sim.distances; path_step=16)

display(fig)
