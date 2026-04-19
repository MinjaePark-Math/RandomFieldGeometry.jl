# Shared utilities for the maintained IG/SLE fan example.
# Run from the package root, for example:
#   julia examples/run_sle_fan.jl grid=385 kappa=2 ds_over_h=0.02

IG_SCRIPT_ROOT = normpath(joinpath(@__DIR__, ".."))

function _required_rfg_names()
    return (
        :sample_chordal_square_ig_field,
        :domain_coordinates,
        :trace_angle_fan,
    )
end

_current_rfg() = getfield(Main, :RandomFieldGeometry)

function _load_local_rfg!()
    src = joinpath(IG_SCRIPT_ROOT, "src", "RandomFieldGeometry.jl")

    if isdefined(Main, :RandomFieldGeometry)
        mod = Base.invokelatest(_current_rfg)
        missing = [name for name in _required_rfg_names() if !isdefined(mod, name)]
        if isempty(missing)
            return mod
        end
        @warn "Loading a fresh local RandomFieldGeometry module from source because the currently loaded REPL binding is missing required IG/SLE helpers." missing=missing
    end

    loader = Module(gensym(:RFGScriptLoader))
    Base.invokelatest(Base.include, loader, src)
    isdefined(loader, :RandomFieldGeometry) || error("Failed to load local RandomFieldGeometry module from source.")
    return Base.invokelatest(getfield, loader, :RandomFieldGeometry)
end

# Load the local package module when this script is included directly from the
# Julia REPL. Do not rely on `using .RandomFieldGeometry` importing exported
# names into Main; REPL sessions can keep an older package binding alive.
# The example below calls package functions through the explicit `RFG.` alias.
RFG = _load_local_rfg!()

try
    @eval using CairoMakie
    @eval using Colors
catch err
    error("These scripts require CairoMakie and Colors. Install them with: import Pkg; Pkg.add([\"CairoMakie\", \"Colors\"])")
end

CM = CairoMakie

function parse_cli_args(args)
    opts = Dict{String,String}()
    for arg in args
        occursin('=', arg) || throw(ArgumentError("arguments must be key=value, got `$arg`."))
        key, value = split(arg, '='; limit=2)
        opts[replace(lowercase(strip(key)), "-" => "_")] = strip(value)
    end
    return opts
end

parse_real_expr(x::Real) = Float64(x)

function parse_real_expr(text::AbstractString)
    return parse_real_expr(Meta.parse(text))
end

function parse_real_expr(ex::Symbol)
    ex === :pi && return Float64(pi)
    throw(ArgumentError("only numeric literals and `pi` are supported in scalar expressions, got `$ex`."))
end

function parse_real_expr(ex::Expr)
    ex.head === :call || throw(ArgumentError("unsupported expression `$ex`."))
    op = ex.args[1]
    values = [parse_real_expr(arg) for arg in ex.args[2:end]]

    if op === :+
        return length(values) == 1 ? values[1] : values[1] + values[2]
    elseif op === :-
        return length(values) == 1 ? -values[1] : values[1] - values[2]
    elseif op === :*
        return prod(values)
    elseif op === :/
        length(values) == 2 || throw(ArgumentError("division expects two operands."))
        return values[1] / values[2]
    elseif op === :^
        length(values) == 2 || throw(ArgumentError("power expects two operands."))
        return values[1]^values[2]
    end

    throw(ArgumentError("unsupported operator `$op` in expression `$ex`."))
end

function parse_bool(text::AbstractString, default::Bool=false)
    lowered = lowercase(strip(text))
    lowered in ("1", "true", "yes", "on") && return true
    lowered in ("0", "false", "no", "off") && return false
    return default
end

get_opt(opts::Dict{String,String}, key::AbstractString, default) = get(opts, key, default)

function parse_int_expr(text::AbstractString)
    value = parse_real_expr(text)
    rounded = round(value)
    if !isfinite(value) || abs(value - rounded) > 100 * eps(Float64) * max(1.0, abs(value))
        throw(ArgumentError("expected an integer expression, got `$text`."))
    end
    return Int(rounded)
end

maybe_int(opts, key) = haskey(opts, key) ? parse_int_expr(opts[key]) : nothing
maybe_real(opts, key) = haskey(opts, key) ? parse_real_expr(opts[key]) : nothing
parse_symbol_text(text) = Symbol(replace(lowercase(strip(String(text))), r"^:" => ""))
maybe_symbol(opts, key) = haskey(opts, key) ? parse_symbol_text(opts[key]) : nothing

function parse_optional_int(text::AbstractString)
    lowered = lowercase(strip(text))
    lowered in ("nothing", "none", "null") && return nothing
    return parse_int_expr(text)
end

maybe_optional_int(opts, key) = haskey(opts, key) ? parse_optional_int(opts[key]) : nothing

_cli_value(x::Symbol) = String(x)
_cli_value(x::AbstractString) = x
_cli_value(x::Bool) = x ? "true" : "false"
_cli_value(::Nothing) = "nothing"
_cli_value(x) = string(x)
keyword_cli_args(kwargs) = [string(k, "=", _cli_value(v)) for (k, v) in kwargs]

function _push_if_some!(pairs::Vector{Pair{Symbol,Any}}, key::Symbol, value)
    value === nothing || push!(pairs, key => value)
    return pairs
end

function _trace_ds_over_h(opts::Dict{String,String}, domain; default::Real=0.02, multiscale::Bool=false)
    h = min(domain.hx, domain.hy)
    if haskey(opts, "ds_over_h")
        ratio = parse_real_expr(opts["ds_over_h"])
    elseif haskey(opts, "ds_factor")
        ratio = parse_real_expr(opts["ds_factor"])
    elseif multiscale && haskey(opts, "step_factor")
        ratio = parse_real_expr(opts["step_factor"])
    elseif haskey(opts, "ds")
        ratio = parse_real_expr(opts["ds"]) / h
    else
        ratio = default
    end
    ratio > 0 || throw(ArgumentError("`ds_over_h` must be positive."))
    return Float64(ratio)
end

function build_trace_config(opts::Dict{String,String}, domain)
    h = min(domain.hx, domain.hy)
    multiscale = parse_bool(get_opt(opts, "multiscale", "false"), false)
    ds_over_h = _trace_ds_over_h(opts, domain; multiscale=multiscale)
    pairs = Pair{Symbol,Any}[]
    push!(pairs, :multiscale => multiscale)
    _push_if_some!(pairs, :levels, maybe_int(opts, "levels"))
    _push_if_some!(pairs, :min_cutoff, maybe_int(opts, "min_cutoff"))
    _push_if_some!(pairs, :spectral_oversample, maybe_int(opts, "spectral_oversample"))
    push!(pairs, :max_steps => maybe_optional_int(opts, "max_steps"))
    _push_if_some!(pairs, :step_factor, maybe_real(opts, "step_factor"))
    _push_if_some!(pairs, :refinement_slack, maybe_real(opts, "refinement_slack"))
    _push_if_some!(pairs, :join_factor, maybe_real(opts, "join_factor"))
    push!(pairs, :boundary_margin => get(opts, "boundary_margin", nothing) === nothing ? 0.0 * h : parse_real_expr(opts["boundary_margin"]))
    push!(pairs, :integrator => get(opts, "integrator", nothing) === nothing ? :euler : parse_symbol_text(opts["integrator"]))
    push!(pairs, :goal_capture_steps => get(opts, "goal_capture_steps", nothing) === nothing ? 0.1 : parse_real_expr(opts["goal_capture_steps"]))
    _push_if_some!(pairs, :adaptive_stuck_steps, maybe_int(opts, "adaptive_stuck_steps"))
    _push_if_some!(pairs, :adaptive_stuck_growth, maybe_real(opts, "adaptive_stuck_growth"))
    _push_if_some!(pairs, :adaptive_max_scale, maybe_real(opts, "adaptive_max_scale"))
    _push_if_some!(pairs, :adaptive_stuck_radius, maybe_real(opts, "adaptive_stuck_radius"))
    push!(pairs, :min_ds_factor => get(opts, "min_ds_factor", nothing) === nothing ? nothing : begin
        raw = lowercase(strip(opts["min_ds_factor"]))
        raw in ("nothing", "none", "null") ? nothing : parse_real_expr(opts["min_ds_factor"])
    end)
    push!(pairs, :phase_interpolation => get(opts, "phase_interpolation", nothing) === nothing ? :tip_phase : parse_symbol_text(opts["phase_interpolation"]))
    return (trace_kwargs=(; pairs...), ds_over_h=ds_over_h, ds=ds_over_h * h, multiscale=multiscale)
end

function sample_chordal_from_opts(opts::Dict{String,String})
    grid = parse(Int, get_opt(opts, "grid", "385"))
    kappa = parse_real_expr(get_opt(opts, "kappa", "2"))
    seed = parse(Int, get_opt(opts, "seed", "20260418"))
    boundary_mode = parse_symbol_text(get_opt(opts, "boundary_mode", "zero_force"))
    boundary_mode in (:zero_boundary, :zero_force) || throw(ArgumentError("boundary_mode must be zero_boundary or zero_force."))
    return RFG.sample_chordal_square_ig_field(grid, kappa; boundary_mode=boundary_mode, seed=seed, T=Float64)
end

function centered_angles(center::Real, left::Real, right::Real, n::Integer)
    n = Int(n)
    n >= 1 || throw(ArgumentError("number of angles must be positive."))
    return n == 1 ? [Float64(center)] : collect(range(Float64(center - left), Float64(center + right); length=n))
end

function trace_termination_counts(traces)
    counts = Dict{Symbol,Int}()
    for trace in traces
        counts[trace.termination] = get(counts, trace.termination, 0) + 1
    end
    return sort(collect(counts); by=first)
end

function compact_trace_summary(; output, terminations, ds, ds_over_h, attempts, retried)
    return (
        output=abspath(String(output)),
        terminations=terminations,
        ds=ds,
        ds_over_h=ds_over_h,
        attempts=attempts,
        retried=retried,
    )
end

trace_scale_kwargs(config) = config.multiscale ? (; step_factor=config.ds_over_h) : (; ds=config.ds)

function _trace_color(idx::Int, total::Int, angle::Real, amin::Real, amax::Real, alpha::Real)
    span = max(Float64(amax - amin), eps(Float64))
    # Square-angle convention: west is +pi/2 and east is -pi/2, so map
    # decreasing angle from west to east onto red -> purple.
    hue = clamp((Float64(amax) - Float64(angle)) / span, 0.0, 1.0)
    rgb = RGB(HSV(300 * hue, 0.86, 1.0))
    return CM.RGBAf(Float32(red(rgb)), Float32(green(rgb)), Float32(blue(rgb)), Float32(alpha))
end

function _square_frame_points(domain)
    xmin, xmax, ymin, ymax = domain.xmin, domain.xmax, domain.ymin, domain.ymax
    return CM.Point2f[
        CM.Point2f(xmin, ymin),
        CM.Point2f(xmax, ymin),
        CM.Point2f(xmax, ymax),
        CM.Point2f(xmin, ymax),
        CM.Point2f(xmin, ymin),
    ]
end

function decimated_points(points; max_points::Union{Nothing,Integer}=nothing)
    isnothing(max_points) && return points
    n = length(points)
    limit = Int(max_points)
    if limit <= 0 || n <= limit
        return points
    elseif limit == 1
        return [points[1]]
    end

    # Display by approximately uniform arclength, not uniform array index.
    # Uniform-index decimation is fragile when max_steps is very large because a
    # terminal boundary-walk or repeated near-goal jitter can occupy most of the
    # stored points and visually erase the interesting earlier part of the path.
    total = 0.0
    prev = points[1]
    @inbounds for k in 2:n
        z = points[k]
        total += abs(z - prev)
        prev = z
    end

    if !(isfinite(total)) || total <= eps(Float64)
        return [points[1], points[end]]
    end

    out = Vector{eltype(points)}()
    sizehint!(out, limit)
    push!(out, points[1])

    spacing = total / (limit - 1)
    next_s = spacing
    accum = 0.0
    prev = points[1]

    @inbounds for k in 2:n
        z = points[k]
        seg = abs(z - prev)
        while seg > 0 && accum + seg >= next_s && length(out) < limit - 1
            t = (next_s - accum) / seg
            push!(out, prev + t * (z - prev))
            next_s += spacing
        end
        accum += seg
        prev = z
    end

    if isempty(out) || abs(out[end] - points[end]) > eps(Float64)
        push!(out, points[end])
    end
    return out
end

function plot_ig_traces(field, traces;
    angles=nothing,
    title::AbstractString="IG/SLE proxy traces",
    show_field::Bool=false,
    show_seeds::Bool=true,
    figure_px::Integer=1400,
    linewidth::Real=1.8,
    line_alpha::Real=0.94,
    field_alpha::Real=0.80,
    max_points_per_trace::Union{Nothing,Integer}=nothing,
)
    fig = CM.Figure(size=(figure_px, figure_px), backgroundcolor=:black, fontsize=18)
    CM.Label(fig[0, 1], title; color=:white, fontsize=22, font=:bold, tellwidth=false)
    ax = CM.Axis(fig[1, 1]; aspect=CM.DataAspect(), backgroundcolor=:black)
    CM.hidedecorations!(ax; grid=false)
    CM.hidespines!(ax)

    xs, ys = RFG.domain_coordinates(field.domain)
    if show_field
        CM.heatmap!(ax, xs, ys, field.values'; colormap=:balance, alpha=Float32(field_alpha))
    end

    CM.lines!(ax, _square_frame_points(field.domain); color=CM.RGBAf(1, 1, 1, 0.86), linewidth=2.0)
    CM.scatter!(ax, [CM.Point2f(0, field.domain.ymin), CM.Point2f(0, field.domain.ymax)]; color=:white, markersize=7)

    angle_values = angles === nothing ? [trace.angle for trace in traces] : collect(angles)
    amin = isempty(angle_values) ? 0.0 : minimum(angle_values)
    amax = isempty(angle_values) ? 1.0 : maximum(angle_values)
    total = max(length(traces), 1)

    for (idx, trace) in enumerate(traces)
        isempty(trace.points) && continue
        display_points = decimated_points(trace.points; max_points=max_points_per_trace)
        pts = CM.Point2f[CM.Point2f(real(z), imag(z)) for z in display_points]
        color = _trace_color(idx, total, trace.angle, amin, amax, line_alpha)
        CM.lines!(ax, pts; color=color, linewidth=linewidth)
        show_seeds && CM.scatter!(ax, [first(pts)]; color=color, markersize=5)
    end

    CM.xlims!(ax, field.domain.xmin, field.domain.xmax)
    CM.ylims!(ax, field.domain.ymin, field.domain.ymax)
    return fig
end

function save_ig_figure(fig, output::AbstractString)
    path = isabspath(output) ? output : joinpath(IG_SCRIPT_ROOT, output)
    mkpath(dirname(path))
    CM.save(path, fig)
    println("saved: ", path)
    return path
end
