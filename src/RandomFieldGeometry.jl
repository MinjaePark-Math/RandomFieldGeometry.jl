module RandomFieldGeometry

using KernelAbstractions

include("RandomFieldGenerators.jl")
include("Flowlines.jl")
include("Pathfinders.jl")
include("Geodesics.jl")
include("Exporters.jl")
include("AnalysisTools.jl")
include("ResearchExports.jl")

using .AnalysisTools
using .Exporters
using .Geodesics
using .Pathfinders
using .RandomFieldGenerators
using .ResearchExports
using .Flowlines

export check_optimal_N, cuda_backend, default_backend, describe_backend, metal_backend, run_lfpp_simulation
export dirichlet_fgf, dirichlet_gff, dirichlet_lgf
export free_fgf, free_gff, free_lgf, free_square_gff
export estimate_ball_growth_dimension, estimate_geodesic_dimension, estimate_shell_growth_exponent
export export_confluence_web, export_metric_ball_web, export_sphere_web, export_slice_web, export_vtk, export_web_binary
export geodesic_edge_weights, interactive_viewer, metric_ball_mask, metric_shell_mask, plot_flowlines, sample_distance_points
export confluence_viewer, metric_ball_viewer, slice_viewer, solve_fpp, sphere_viewer, trace_all_geodesics, trace_path
export IGConstants, IGField, SquareDomain
export FlowlineField, FlowlineTrace, MultiscaleFlowlineField
export add_interior_singularity!, boundary_seed, domain_coordinates
export flowline_field, ig_constants, ig_critical_angle, multiscale_flowline_field, sample_chordal_square_ig_field, sample_interior_ig_field
export square_chordal_boundary_data, square_domain, square_seed_grid
export trace_angle_fan, trace_flowline, trace_sle_fan

const VERSION = v"0.1.0"

function _cuda_backend_impl end
function _metal_backend_impl end

"""
    default_backend()

Return the default KernelAbstractions backend used by the package.
"""
cuda_backend() = applicable(_cuda_backend_impl) ? _cuda_backend_impl() : nothing
metal_backend() = applicable(_metal_backend_impl) ? _metal_backend_impl() : nothing

function default_backend()
    backend = cuda_backend()
    backend !== nothing && return backend

    backend = metal_backend()
    backend !== nothing && return backend

    return KernelAbstractions.CPU()
end

describe_backend(backend=default_backend()) = string(typeof(backend))

"""
    check_optimal_N(N::Integer)

Warn if `N` is not friendly to the internal sine-transform-based field generator.
"""
function check_optimal_N(N::Integer)
    N > 1 || throw(ArgumentError("`N` must be at least 2."))

    if isodd(N)
        @warn "N = $N is odd. The internal sine transform is typically faster when N is even."
    end

    remainder = Int(N)
    for p in (2, 3, 5, 7)
        while remainder % p == 0
            remainder = div(remainder, p)
        end
    end

    if remainder > 1
        @warn "N = $N has a large prime factor ($remainder). FFTW may be slower than for powers of 2 or products of small primes."
    end

    return nothing
end

"""
    run_lfpp_simulation(N, xi; dim=3, seed=42, backend=default_backend(),
                        max_iters=nothing, sweep_factor=8, print_every=100, return_info=false)

Generate a Dirichlet random field on an `N`-box, convert it to LFPP weights
`exp.(xi * h)`, and solve the resulting 2D or 3D first-passage problem.

For `dim = 2`, the field is sampled with `dirichlet_gff`. For `dim = 3`, the
field is sampled with `dirichlet_lgf`.
"""
function run_lfpp_simulation(
    N::Integer,
    xi::Real;
    dim::Integer=3,
    seed::Integer=42,
    backend=default_backend(),
    max_iters::Union{Nothing,Integer}=nothing,
    sweep_factor::Integer=8,
    print_every::Integer=100,
    return_info::Bool=false,
)
    dim in (2, 3) || throw(ArgumentError("`run_lfpp_simulation` currently supports only `dim = 2` or `dim = 3`."))

    check_optimal_N(N)

    println("Generating random field... (dim = $(dim), N = $(N), seed = $(seed))")
    field_start_time = time()
    field = dim == 2 ?
        RandomFieldGenerators.dirichlet_gff(2, Int(N), Int(seed)) :
        RandomFieldGenerators.dirichlet_lgf(3, Int(N), Int(seed))
    field_elapsed = round(time() - field_start_time, digits=2)
    println("  -> Field generation complete. Time: $(field_elapsed)s")

    weights = exp.(Float32(xi) .* field) .+ 1f-5
    solve_result = Pathfinders.solve_fpp(
        weights;
        backend=backend,
        max_iters=max_iters,
        sweep_factor=sweep_factor,
        print_every=print_every,
        return_info=return_info,
    )

    if return_info
        info = merge(solve_result.info, (field_generation_seconds=Float64(field_elapsed),))
        return (distances=solve_result.distances, weights=weights, info=info)
    end

    return (distances=solve_result, weights=weights)
end

function _require_makie_extension(name::AbstractString)
    error("`$name` lives in the optional GLMakie extension. Install `GLMakie`, `GeometryBasics`, and `Colors`, load them, and call `$name` again.")
end

function interactive_viewer(args...; kwargs...)
    _require_makie_extension("interactive_viewer")
end

function confluence_viewer(args...; kwargs...)
    _require_makie_extension("confluence_viewer")
end

function metric_ball_viewer(args...; kwargs...)
    _require_makie_extension("metric_ball_viewer")
end

function slice_viewer(args...; kwargs...)
    _require_makie_extension("slice_viewer")
end

function sphere_viewer(args...; kwargs...)
    _require_makie_extension("sphere_viewer")
end

function plot_flowlines(args...; kwargs...)
    _require_makie_extension("plot_flowlines")
end

end
