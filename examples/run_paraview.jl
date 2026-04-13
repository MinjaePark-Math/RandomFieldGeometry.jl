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

using RandomFieldGeometry

backend = default_backend()
println("Using backend: " * describe_backend(backend))
if @isdefined(CUDA)
    println("CUDA.functional() = $(CUDA.functional())")
end

dim = 3
N = 512
xi = 1.0f0

sim = run_lfpp_simulation(N, xi; dim=dim, backend=backend)

output_dir = joinpath(@__DIR__, "..", "vtk_results")
mkpath(output_dir)

export_vtk(
    sim.distances,
    joinpath(output_dir, "LFPP_dim$(dim)_N$(N)_xi$(xi)");
    path_step=5,
)
