module LiouvilleFPP

# Legacy include-based entry point kept for local scripts that still do
# `include("src/LiouvilleFPP.jl")`. The public package entry point is
# `using RandomFieldGeometry`.

include("RandomFieldGeometry.jl")
using .RandomFieldGeometry

const Exporters = RandomFieldGeometry.Exporters
const Geodesics = RandomFieldGeometry.Geodesics
const Pathfinders = RandomFieldGeometry.Pathfinders
const RandomFieldGenerators = RandomFieldGeometry.RandomFieldGenerators

for name in (
    :check_optimal_N,
    :default_backend,
    :dirichlet_fgf,
    :dirichlet_gff,
    :dirichlet_lgf,
    :export_vtk,
    :export_web_binary,
    :interactive_viewer,
    :run_lfpp_simulation,
    :run_liouville_simulation,
    :solve_fpp,
    :trace_all_geodesics,
    :trace_path,
)
    @eval const $(name) = RandomFieldGeometry.$(name)
end

export check_optimal_N, default_backend, dirichlet_fgf, dirichlet_gff, dirichlet_lgf
export export_vtk, export_web_binary, interactive_viewer, run_lfpp_simulation, run_liouville_simulation
export solve_fpp, trace_all_geodesics, trace_path

end
