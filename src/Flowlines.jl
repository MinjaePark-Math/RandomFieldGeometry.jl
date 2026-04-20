module Flowlines

using FFTW
using Random
using Base.Threads
using ..RandomFieldGenerators

export FlowlineField, FlowlineTrace, MultiscaleFlowlineField
export flowline_field, multiscale_flowline_field
export trace_angle_fan, trace_flowline, trace_sle_fan

struct FlowlineField{T<:AbstractFloat}
    values::Matrix{T}
    phase::Matrix{Complex{T}}
    invchi::T
    goal::Complex{T}
    goal_radius::T
    domain::RandomFieldGenerators.SquareDomain{T}
end

function _precomputed_phase(values::AbstractMatrix{T}, invchi::T) where {T<:AbstractFloat}
    phase = Matrix{Complex{T}}(undef, size(values))
    @inbounds for idx in eachindex(values)
        s, c = sincos(values[idx] * invchi)
        phase[idx] = complex(c, s)
    end
    return phase
end

function FlowlineField(
    values::Matrix{T},
    invchi::T,
    goal::Complex{T},
    goal_radius::T,
    domain::RandomFieldGenerators.SquareDomain{T},
) where {T<:AbstractFloat}
    return FlowlineField{T}(values, _precomputed_phase(values, invchi), invchi, goal, goal_radius, domain)
end

struct MultiscaleFlowlineField{T<:AbstractFloat}
    levels::Vector{FlowlineField{T}}
    cutoffs::Vector{Int}
    domain::RandomFieldGenerators.SquareDomain{T}
end

struct FlowlineTrace{T<:AbstractFloat}
    points::Vector{Complex{T}}
    seed::Complex{T}
    angle::T
    termination::Symbol
    merged_into::Int
end

const _SMALL_DS_WARNING_SENT = Ref(false)

@inline function _northgoing_phase_offset(::Type{T}) where {T<:AbstractFloat}
    return T(pi) / T(2)
end

@inline function _phase_angle(angle::T) where {T<:AbstractFloat}
    return angle + _northgoing_phase_offset(T)
end

function _maybe_warn_small_ds(flow::FlowlineField{T}, requested_ds::T, effective_ds::T) where {T<:AbstractFloat}
    h = min(flow.domain.hx, flow.domain.hy)
    if requested_ds < T(0.02) * h && !_SMALL_DS_WARNING_SENT[]
        _SMALL_DS_WARNING_SENT[] = true
        @warn "Tracing with `ds << h` on a single mesh mostly over-resolves one fixed grid regularization. If `min_ds_factor` is set, the effective step may also be clamped upward." requested_ds effective_ds h requested_ratio=requested_ds / h effective_ratio=effective_ds / h
    end
    return nothing
end

@inline function _effective_ds(flow::FlowlineField{T}, requested_ds::T, min_ds_factor) where {T<:AbstractFloat}
    isnothing(min_ds_factor) && return requested_ds
    min_ds = max(zero(T), T(min_ds_factor)) * min(flow.domain.hx, flow.domain.hy)
    return max(requested_ds, min_ds)
end

"""
    flowline_field(values, domain, chi)
    flowline_field(field::IGField)

Store a mesh-regularized scalar field together with the square geometry
constants used to trace imaginary-geometry flowline proxies. The tracer
bilinearly interpolates the scalar field `h` and then evaluates `exp(i h / χ)`
by default, preserving the rapid local changes of the regularized field.

For chordal square fields, `phase_interpolation=:tip_phase` uses scalar
interpolation away from the north target but interpolates the precomputed unit
phase in a small north-tip strip; this avoids the `2πχ` branch artifact near
`i` without globally smoothing the direction field. Use
`phase_interpolation=:phase` only when you deliberately want globally circular
interpolated directions.

Important: these routines operate on a single mesh-regularized field and then
advance it by repeatedly following the local direction `exp(i h / χ)` with a
small mesh-scale stepping rule. They therefore produce regularized
discrete proxy curves, not true continuum Miller-Sheffield
imaginary-geometry flow lines, whose scaling limits are SLE-type fractal
curves. User-facing angles follow the square convention `θ = 0` north,
`θ = π/2` west, `θ = -π/2` east, and `θ = π` south.
"""
function flowline_field(values::AbstractMatrix{<:Real}, domain::RandomFieldGenerators.SquareDomain, chi::Real)
    size(values) == (domain.n, domain.n) || throw(ArgumentError("`values` must match the domain size."))
    T = promote_type(eltype(values), typeof(domain.hx), typeof(chi))
    scalar_values = Matrix{T}(undef, size(values))
    @inbounds for idx in eachindex(values)
        scalar_values[idx] = T(values[idx])
    end
    nogoal = complex(T(NaN), T(NaN))
    return FlowlineField(scalar_values, inv(T(chi)), nogoal, zero(T), domain)
end

function _chordal_goal(domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    radius = T(0.1) * min(domain.hx, domain.hy)
    # The continuum chordal target is the north marked boundary point +i.
    # The positive radius keeps numerical integration from evaluating the
    # singular endpoint itself; once captured, traces append this exact target.
    goal = complex(zero(T), domain.ymax)
    return goal, radius
end

function flowline_field(field::RandomFieldGenerators.IGField{T}) where {T<:AbstractFloat}
    flow = flowline_field(field.values, field.domain, field.constants.chi)
    if field.kind === :chordal_square
        goal, radius = _chordal_goal(field.domain)
        return FlowlineField(flow.values, flow.invchi, goal, radius, flow.domain)
    end
    return flow
end

function _deterministic_part(field::RandomFieldGenerators.IGField{T}) where {T<:AbstractFloat}
    return _deterministic_part(field, field.domain)
end

function _resampled_boundary(
    field::RandomFieldGenerators.IGField{T},
    domain::RandomFieldGenerators.SquareDomain{T},
) where {T<:AbstractFloat}
    if field.kind === :chordal_square
        return RandomFieldGenerators.square_chordal_boundary_data(
            domain,
            field.constants.kappa;
            boundary_mode=field.boundary_mode,
            shift=field.boundary_shift,
        )
    end

    return fill(field.boundary_shift, domain.n, domain.n)
end

function _deterministic_part(
    field::RandomFieldGenerators.IGField{T},
    domain::RandomFieldGenerators.SquareDomain{T},
) where {T<:AbstractFloat}
    size(field.deterministic) == (domain.n, domain.n) && return field.deterministic

    boundary = size(field.boundary) == (domain.n, domain.n) ? field.boundary : _resampled_boundary(field, domain)
    deterministic = RandomFieldGenerators._harmonic_extension(boundary, domain.hx, domain.hy; T=T)

    if field.kind === :interior
        RandomFieldGenerators.add_interior_singularity!(
            deterministic,
            domain;
            alpha=field.alpha,
            beta=field.beta,
            z0=field.z0,
        )
    end

    return deterministic
end

@inline function _square_halfwidth(domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    return (domain.xmax - domain.xmin) / T(2)
end

function _refined_domain(
    domain::RandomFieldGenerators.SquareDomain{T},
    oversample::Integer,
) where {T<:AbstractFloat}
    factor = Int(oversample)
    factor >= 1 || throw(ArgumentError("`spectral_oversample` must be at least 1."))
    factor == 1 && return domain

    refined_n = (domain.n - 1) * factor + 1
    return RandomFieldGenerators.square_domain(refined_n; halfwidth=_square_halfwidth(domain), T=T)
end

function _spectral_coefficients(values::AbstractMatrix{T}, domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    m = domain.n - 2
    coeffs = Matrix{T}(undef, m, m)
    @views coeffs .= values[2:(domain.n - 1), 2:(domain.n - 1)]
    FFTW.r2r!(coeffs, FFTW.RODFT00)
    return coeffs
end

function _spectral_to_field(coeffs::AbstractMatrix{T}, domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    m = domain.n - 2
    interior = Matrix{T}(undef, m, m)
    interior .= coeffs
    FFTW.r2r!(interior, FFTW.RODFT00)
    interior ./= T(4 * (m + 1) * (m + 1))

    out = zeros(T, domain.n, domain.n)
    @views out[2:(domain.n - 1), 2:(domain.n - 1)] .= interior
    return out
end

function _estimate_field_scale(coeffs::AbstractMatrix{T}, domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    m = size(coeffs, 1)
    sample_cutoff = min(m, 16)
    eig = RandomFieldGenerators._dirichlet_eigenvalues(m, domain.hx)
    accum = zero(T)
    count = 0

    @inbounds for j in 1:sample_cutoff, i in 1:sample_cutoff
        accum += coeffs[i, j] * coeffs[i, j] * (eig[i] + eig[j])
        count += 1
    end

    count == 0 && return sqrt(T(2pi))
    scale2 = accum / T(count)
    return sqrt(max(scale2, eps(T)))
end

function _extended_spectral_coefficients(
    field::RandomFieldGenerators.IGField{T},
    domain::RandomFieldGenerators.SquareDomain{T};
    rng::Union{Nothing,AbstractRNG}=nothing,
    extension_seed::Union{Nothing,Integer}=nothing,
) where {T<:AbstractFloat}
    base_coeffs = _spectral_coefficients(field.random, field.domain)

    domain.n == field.domain.n && return base_coeffs

    local_rng = isnothing(rng) ? (
        isnothing(extension_seed) ? Random.default_rng() : Xoshiro(extension_seed)
    ) : rng

    m_base = size(base_coeffs, 1)
    m_fine = domain.n - 2
    coeffs = zeros(T, m_fine, m_fine)
    @views coeffs[1:m_base, 1:m_base] .= base_coeffs

    eig = RandomFieldGenerators._dirichlet_eigenvalues(m_fine, domain.hx)
    scale = _estimate_field_scale(base_coeffs, field.domain)

    @inbounds for j in 1:m_fine, i in 1:m_fine
        if i > m_base || j > m_base
            coeffs[i, j] = randn(local_rng, T) * scale / sqrt(eig[i] + eig[j])
        end
    end

    return coeffs
end

function _cutoff_schedule(full_cutoff::Int; levels::Integer=6, min_cutoff::Int=8)
    levels = Int(levels)
    levels >= 2 || throw(ArgumentError("`levels` must be at least 2."))
    full_cutoff >= 1 || throw(ArgumentError("the field must have a non-empty interior."))

    low = clamp(Int(min_cutoff), 1, full_cutoff)
    raw = round.(Int, exp.(range(log(float(low)), log(float(full_cutoff)); length=levels)))
    cutoffs = unique(clamp.(raw, 1, full_cutoff))
    last(cutoffs) == full_cutoff || push!(cutoffs, full_cutoff)
    first(cutoffs) == low || pushfirst!(cutoffs, low)
    return unique(cutoffs)
end

"""
    multiscale_flowline_field(field; levels=6, min_cutoff=8)

Construct a sequence of progressively less-regularized square vector fields by
low-pass filtering only the random part of the IG field in the Dirichlet sine
basis. With `spectral_oversample > 1`, the random part is first extended to a
finer hidden square grid by embedding the resolved low modes and sampling new
high modes, which gives the tracer a genuine vanishing-regularization ladder
below the display mesh.
"""
function multiscale_flowline_field(
    field::RandomFieldGenerators.IGField{T};
    levels::Integer=6,
    min_cutoff::Int=8,
    spectral_oversample::Integer=1,
    rng::Union{Nothing,AbstractRNG}=nothing,
    extension_seed::Union{Nothing,Integer}=nothing,
) where {T<:AbstractFloat}
    domain = _refined_domain(field.domain, spectral_oversample)
    deterministic = _deterministic_part(field, domain)
    coeffs = _extended_spectral_coefficients(field, domain; rng=rng, extension_seed=extension_seed)
    cutoffs = _cutoff_schedule(domain.n - 2; levels=levels, min_cutoff=min_cutoff)
    goal, goal_radius = field.kind === :chordal_square ?
        _chordal_goal(domain) :
        (complex(T(NaN), T(NaN)), zero(T))

    level_fields = Vector{FlowlineField{T}}(undef, length(cutoffs))
    for (idx, cutoff) in enumerate(cutoffs)
        filtered = zeros(T, size(coeffs))
        @views filtered[1:cutoff, 1:cutoff] .= coeffs[1:cutoff, 1:cutoff]
        values = deterministic .+ _spectral_to_field(filtered, domain)
        base = flowline_field(values, domain, field.constants.chi)
        level_fields[idx] = FlowlineField(base.values, base.invchi, goal, goal_radius, domain)
    end

    return MultiscaleFlowlineField(level_fields, cutoffs, domain)
end

@inline function _level_step(
    domain::RandomFieldGenerators.SquareDomain{T},
    cutoff_full::Int,
    cutoff::Int;
    step_factor::Real=0.55,
) where {T<:AbstractFloat}
    multiplier = sqrt(max(one(T), T(cutoff_full) / T(max(cutoff, 1))))
    return T(step_factor) * min(domain.hx, domain.hy) * multiplier
end

@inline function _complex_nan(::Type{T}) where {T<:AbstractFloat}
    return complex(T(NaN), T(NaN))
end

@inline function _has_goal(flow::FlowlineField{T}) where {T<:AbstractFloat}
    return flow.goal_radius > zero(T) && isfinite(real(flow.goal)) && isfinite(imag(flow.goal))
end

@inline function _goal_reached(flow::FlowlineField{T}, z::Complex{T}) where {T<:AbstractFloat}
    return _has_goal(flow) && abs(z - flow.goal) <= flow.goal_radius
end

@inline function _near_goal_strip(flow::FlowlineField{T}, z::Complex{T}) where {T<:AbstractFloat}
    return _has_goal(flow) && imag(z) >= imag(flow.goal) - T(8) * flow.goal_radius
end

@inline function _default_step_budget(domain::RandomFieldGenerators.SquareDomain{T}, ds::T) where {T<:AbstractFloat}
    # Constant-speed tracing means step count is essentially arclength / ds.
    # The old budget used 32 diagonals and could silently request hundreds of
    # thousands of RK steps per curve on the default scripts.  Keep the low-level
    # fallback generous, but bounded at a mesh-scaled value so exploratory fans
    # return in finite time even when one angle wanders for a long time.
    diagonal = hypot(domain.xmax - domain.xmin, domain.ymax - domain.ymin)
    natural_budget = ceil(Int, T(6) * diagonal / ds)
    mesh_budget = max(20_000, 120 * max(1, domain.n - 1))
    return clamp(natural_budget, 2_000, mesh_budget)
end

@inline function _normalize_or_zero(z::Complex{T}) where {T<:AbstractFloat}
    n = abs(z)
    return n > eps(T) ? z / n : zero(Complex{T})
end

@inline function _bilinear_value(flow::FlowlineField{T}, z::Complex{T}) where {T<:AbstractFloat}
    domain = flow.domain
    ξ = (real(z) - domain.xmin) * domain.invhx
    η = (imag(z) - domain.ymin) * domain.invhy
    limit = T(domain.n - 1)

    if ξ < zero(T) || ξ > limit || η < zero(T) || η > limit
        return T(NaN)
    end

    i = floor(Int, ξ) + 1
    j = floor(Int, η) + 1
    if i >= domain.n
        i = domain.n - 1
    end
    if j >= domain.n
        j = domain.n - 1
    end

    tx = ξ - T(i - 1)
    ty = η - T(j - 1)

    @inbounds begin
        v11 = flow.values[i, j]
        v21 = flow.values[i + 1, j]
        v12 = flow.values[i, j + 1]
        v22 = flow.values[i + 1, j + 1]

        return (one(T) - tx) * (one(T) - ty) * v11 +
               tx * (one(T) - ty) * v21 +
               (one(T) - tx) * ty * v12 +
               tx * ty * v22
    end
end

@inline function _bilinear_phase(flow::FlowlineField{T}, z::Complex{T}) where {T<:AbstractFloat}
    domain = flow.domain
    ξ = (real(z) - domain.xmin) * domain.invhx
    η = (imag(z) - domain.ymin) * domain.invhy
    limit = T(domain.n - 1)

    if ξ < zero(T) || ξ > limit || η < zero(T) || η > limit
        return _complex_nan(T)
    end

    i = floor(Int, ξ) + 1
    j = floor(Int, η) + 1
    if i >= domain.n
        i = domain.n - 1
    end
    if j >= domain.n
        j = domain.n - 1
    end

    tx = ξ - T(i - 1)
    ty = η - T(j - 1)

    @inbounds begin
        v11 = flow.phase[i, j]
        v21 = flow.phase[i + 1, j]
        v12 = flow.phase[i, j + 1]
        v22 = flow.phase[i + 1, j + 1]

        return (one(T) - tx) * (one(T) - ty) * v11 +
               tx * (one(T) - ty) * v21 +
               (one(T) - tx) * ty * v12 +
               tx * ty * v22
    end
end

@inline function _direction_scalar(flow::FlowlineField{T}, z::Complex{T}, angle_offset::T) where {T<:AbstractFloat}
    h = _bilinear_value(flow, z)
    if !isfinite(h)
        return zero(Complex{T})
    end

    local_angle = muladd(h, flow.invchi, angle_offset)
    s, c = sincos(local_angle)
    return complex(c, s)
end

@inline function _direction_phase(flow::FlowlineField{T}, z::Complex{T}, angle_offset::T) where {T<:AbstractFloat}
    phase = _bilinear_phase(flow, z)
    n = abs(phase)
    if !isfinite(n)
        return zero(Complex{T})
    elseif n <= sqrt(eps(T))
        return _direction_scalar(flow, z, angle_offset)
    end

    s, c = sincos(angle_offset)
    return (phase / n) * complex(c, s)
end

@inline function _direction(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    if phase_interpolation === :scalar
        return _direction_scalar(flow, z, angle_offset)
    elseif phase_interpolation === :tip_phase
        return _near_goal_strip(flow, z) ?
            _direction_phase(flow, z, angle_offset) :
            _direction_scalar(flow, z, angle_offset)
    elseif phase_interpolation === :phase
        return _direction_phase(flow, z, angle_offset)
    end

    throw(ArgumentError("`phase_interpolation` must be `:scalar`, `:tip_phase`, or `:phase`."))
end

@inline function _euler_step(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    ds::T,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    k1 = _direction(flow, z, angle_offset, phase_interpolation)
    return k1 == 0 ? z : z + ds * k1
end

@inline function _rk2_step(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    ds::T,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    k1 = _direction(flow, z, angle_offset, phase_interpolation)
    if k1 == 0
        return z
    end

    k2 = _direction(flow, z + ds * k1 / T(2), angle_offset, phase_interpolation)
    return k2 == 0 ? z + ds * k1 : z + ds * k2
end

@inline function _rk4_step(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    ds::T,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    k1 = _direction(flow, z, angle_offset, phase_interpolation)
    if k1 == 0
        return z
    end

    k2 = _direction(flow, z + ds * k1 / T(2), angle_offset, phase_interpolation)
    k3 = _direction(flow, z + ds * k2 / T(2), angle_offset, phase_interpolation)
    k4 = _direction(flow, z + ds * k3, angle_offset, phase_interpolation)
    return z + ds * (k1 + T(2) * k2 + T(2) * k3 + k4) / T(6)
end

@inline function _step(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    ds::T,
    integrator::Symbol,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    if integrator === :euler
        return _euler_step(flow, z, angle_offset, ds, phase_interpolation)
    elseif integrator === :rk2
        return _rk2_step(flow, z, angle_offset, ds, phase_interpolation)
    elseif integrator === :rk4
        return _rk4_step(flow, z, angle_offset, ds, phase_interpolation)
    end

    throw(ArgumentError("`integrator` must be one of `:euler`, `:rk2`, or `:rk4`."))
end

@inline function _goal_capture_radius(flow::FlowlineField{T}, goal_capture_steps) where {T<:AbstractFloat}
    !_has_goal(flow) && return zero(T)
    isnothing(goal_capture_steps) && return flow.goal_radius
    return max(flow.goal_radius, T(goal_capture_steps) * min(flow.domain.hx, flow.domain.hy))
end

@inline function _trace_goal_capture_radius(
    flow::FlowlineField{T},
    goal_capture_steps,
    margin::T,
    ds::T,
) where {T<:AbstractFloat}
    radius = _goal_capture_radius(flow, goal_capture_steps)
    _has_goal(flow) || return radius

    # With positive boundary_margin, the active top edge is below the actual
    # marked point +i. Include that margin, plus one numerical step, so a trace
    # that has reached the active compactification of infinity can terminate
    # cleanly instead of oscillating forever below it when max_steps is huge.
    return max(radius, margin + ds)
end

@inline function _goal_captured(flow::FlowlineField{T}, z::Complex{T}, capture_radius::T) where {T<:AbstractFloat}
    return _has_goal(flow) && abs(z - flow.goal) <= capture_radius
end

function _append_goal_capture_segment!(
    points::Vector{Complex{T}},
    flow::FlowlineField{T},
    ztip::Complex{T},
) where {T<:AbstractFloat}
    push!(points, ztip)
    points[end] == flow.goal || push!(points, flow.goal)
    return nothing
end

@inline function _adaptive_step_update(
    z::Complex{T},
    anchor::Complex{T},
    current_ds::T,
    base_ds::T,
    stuck_steps::Int,
    adaptive_stuck_steps,
    adaptive_stuck_growth::Real,
    adaptive_max_scale::Real,
    adaptive_stuck_radius::Real,
) where {T<:AbstractFloat}
    isnothing(adaptive_stuck_steps) && return base_ds, z, 0

    radius = T(adaptive_stuck_radius) * current_ds
    if !isfinite(real(anchor)) || !isfinite(imag(anchor)) || abs(z - anchor) > radius
        return base_ds, z, 0
    end

    new_stuck_steps = stuck_steps + 1
    max_ds = T(adaptive_max_scale) * base_ds
    grown_ds = min(current_ds * T(adaptive_stuck_growth), max_ds)
    if new_stuck_steps >= Int(adaptive_stuck_steps) && grown_ds > current_ds + eps(T)
        return grown_ds, z, 0
    end

    return current_ds, anchor, new_stuck_steps
end

@inline function _active_bounds(domain::RandomFieldGenerators.SquareDomain{T}, margin::T) where {T<:AbstractFloat}
    return (
        domain.xmin + margin,
        domain.xmax - margin,
        domain.ymin + margin,
        domain.ymax - margin,
    )
end

@inline function _clamp_to_active_box(
    domain::RandomFieldGenerators.SquareDomain{T},
    z::Complex{T},
    margin::T,
) where {T<:AbstractFloat}
    xmin, xmax, ymin, ymax = _active_bounds(domain, margin)
    return complex(clamp(real(z), xmin, xmax), clamp(imag(z), ymin, ymax))
end

@inline function _box_boundary_tolerance(domain::RandomFieldGenerators.SquareDomain{T}) where {T<:AbstractFloat}
    scale = max(abs(domain.xmin), abs(domain.xmax), abs(domain.ymin), abs(domain.ymax), one(T))
    return max(T(16) * eps(T) * scale, sqrt(eps(T)) * min(domain.hx, domain.hy))
end

@inline function _project_to_box_tangent_cone(
    domain::RandomFieldGenerators.SquareDomain{T},
    z::Complex{T},
    v::Complex{T},
    margin::T,
) where {T<:AbstractFloat}
    xmin, xmax, ymin, ymax = _active_bounds(domain, margin)
    tol = _box_boundary_tolerance(domain)
    x = real(z)
    y = imag(z)
    dx = real(v)
    dy = imag(v)

    # Remove only the outward normal component.  This implements projected
    # dynamics on the active square: the path may slide along the clamped
    # boundary, but it does not bounce and it does not terminate merely because
    # a finite step tried to leave the box.
    if x <= xmin + tol && dx < zero(T)
        dx = zero(T)
    elseif x >= xmax - tol && dx > zero(T)
        dx = zero(T)
    end
    if y <= ymin + tol && dy < zero(T)
        dy = zero(T)
    elseif y >= ymax - tol && dy > zero(T)
        dy = zero(T)
    end

    return _normalize_or_zero(complex(dx, dy))
end

@inline function _fallback_motion_target(flow::FlowlineField{T}, margin::T) where {T<:AbstractFloat}
    xmin, xmax, ymin, ymax = _active_bounds(flow.domain, margin)
    return complex((xmin + xmax) / T(2), (ymin + ymax) / T(2))
end

@inline function _inward_box_normal(
    domain::RandomFieldGenerators.SquareDomain{T},
    z::Complex{T},
    margin::T,
) where {T<:AbstractFloat}
    xmin, xmax, ymin, ymax = _active_bounds(domain, margin)
    tol = _box_boundary_tolerance(domain)
    x = real(z)
    y = imag(z)
    dx = zero(T)
    dy = zero(T)

    x <= xmin + tol && (dx += one(T))
    x >= xmax - tol && (dx -= one(T))
    y <= ymin + tol && (dy += one(T))
    y >= ymax - tol && (dy -= one(T))

    return _normalize_or_zero(complex(dx, dy))
end

@inline function _on_active_boundary(
    domain::RandomFieldGenerators.SquareDomain{T},
    z::Complex{T},
    margin::T,
) where {T<:AbstractFloat}
    xmin, xmax, ymin, ymax = _active_bounds(domain, margin)
    tol = _box_boundary_tolerance(domain)
    x = real(z)
    y = imag(z)
    return x <= xmin + tol || x >= xmax - tol || y <= ymin + tol || y >= ymax - tol
end

function _unstall_clamped_step(
    flow::FlowlineField{T},
    z::Complex{T},
    angle_offset::T,
    ds::T,
    margin::T,
    phase_interpolation::Symbol,
) where {T<:AbstractFloat}
    # First try the actual local direction, but projected to the tangent cone of
    # the active square.  If the attempted motion is exactly outward normal, use
    # a deterministic fallback inward direction so the trace keeps moving
    # instead of reporting `:stalled`.
    v = _direction(flow, z, angle_offset, phase_interpolation)
    projected = _project_to_box_tangent_cone(flow.domain, z, v, margin)

    if projected == 0
        target = _fallback_motion_target(flow, margin)
        projected = _project_to_box_tangent_cone(flow.domain, z, target - z, margin)
    end

    if projected == 0
        projected = _inward_box_normal(flow.domain, z, margin)
    end

    projected == 0 && return z
    return _clamp_to_active_box(flow.domain, z + ds * projected, margin)
end

function _handle_boundary_exit!(
    flow::FlowlineField{T},
    z::Complex{T},
    znew::Complex{T},
    ds::T,
    margin::T,
) where {T<:AbstractFloat}
    if RandomFieldGenerators._inside_domain(flow.domain, znew; margin=margin)
        return znew, nothing
    end

    # Single boundary rule: clamp to the active square and let subsequent steps
    # follow the projected local direction field naturally.
    clamped = _clamp_to_active_box(flow.domain, znew, margin)
    return clamped, nothing
end

function _trace_flowline_impl(
    flow::FlowlineField{T},
    seed::Complex{T},
    angle::T;
    ds::Union{Nothing,Real}=nothing,
    max_steps::Union{Nothing,Integer}=nothing,
    boundary_margin::Union{Nothing,Real}=nothing,
    integrator::Symbol=:euler,
    stop_when::Union{Nothing,Function}=nothing,
    phase_interpolation::Symbol=:tip_phase,
    min_ds_factor::Union{Nothing,Real}=nothing,
    goal_capture_steps::Union{Nothing,Real}=2,
    adaptive_stuck_steps::Union{Nothing,Integer}=32,
    adaptive_stuck_growth::Real=2,
    adaptive_max_scale::Real=64,
    adaptive_stuck_radius::Real=24,
) where {T<:AbstractFloat}
    requested_ds = isnothing(ds) ? T(0.05) * min(flow.domain.hx, flow.domain.hy) : T(ds)
    requested_ds > zero(T) || throw(ArgumentError("`ds` must be positive."))
    local_ds = _effective_ds(flow, requested_ds, min_ds_factor)
    _maybe_warn_small_ds(flow, requested_ds, local_ds)
    local_max_steps = isnothing(max_steps) ? _default_step_budget(flow.domain, local_ds) : Int(max_steps)
    local_max_steps >= 1 || throw(ArgumentError("`max_steps` must be positive."))
    margin = isnothing(boundary_margin) ? T(1.1) * min(flow.domain.hx, flow.domain.hy) : T(boundary_margin)
    margin >= zero(T) || throw(ArgumentError("`boundary_margin` must be non-negative."))
    phase_angle = _phase_angle(angle)
    capture_radius = _trace_goal_capture_radius(flow, goal_capture_steps, margin, local_ds)

    points = Vector{Complex{T}}()
    sizehint!(points, min(local_max_steps + 1, 8192))
    push!(points, seed)

    if !RandomFieldGenerators._inside_domain(flow.domain, seed; margin=margin)
        return FlowlineTrace(points, seed, angle, :outside, 0)
    end

    z = seed
    current_ds = local_ds
    stuck_anchor = seed
    stuck_steps = 0
    for step in 1:local_max_steps
        capture_radius = _trace_goal_capture_radius(flow, goal_capture_steps, margin, current_ds)
        znew = _step(flow, z, phase_angle, current_ds, integrator, phase_interpolation)
        if znew == z
            znew = _unstall_clamped_step(flow, z, phase_angle, current_ds, margin, phase_interpolation)
        end
        reached_goal = _goal_reached(flow, znew)
        captured_goal = !reached_goal && _goal_captured(flow, znew, capture_radius)
        if reached_goal || captured_goal
            if captured_goal
                _append_goal_capture_segment!(points, flow, znew)
            else
                push!(points, znew)
            end
            return FlowlineTrace(points, seed, angle, :target, 0)
        end
        znew, boundary_status = _handle_boundary_exit!(flow, z, znew, current_ds, margin)
        reached_goal = _goal_reached(flow, znew)
        captured_goal = !reached_goal && _goal_captured(flow, znew, capture_radius)
        if reached_goal || captured_goal
            if captured_goal
                _append_goal_capture_segment!(points, flow, znew)
            else
                push!(points, znew)
            end
            return FlowlineTrace(points, seed, angle, :target, 0)
        end
        if boundary_status !== nothing
            return FlowlineTrace(points, seed, angle, boundary_status, 0)
        end
        if znew == z
            znew = _unstall_clamped_step(flow, z, phase_angle, current_ds, margin, phase_interpolation)
        end
        if znew == z
            return FlowlineTrace(points, seed, angle, :max_steps, 0)
        end

        push!(points, znew)
        current_ds, stuck_anchor, stuck_steps = _adaptive_step_update(
            znew,
            stuck_anchor,
            current_ds,
            local_ds,
            stuck_steps,
            adaptive_stuck_steps,
            adaptive_stuck_growth,
            adaptive_max_scale,
            adaptive_stuck_radius,
        )
        z = znew

        if !isnothing(stop_when) && stop_when(z, step, points)
            return FlowlineTrace(points, seed, angle, :stop_condition, 0)
        end
    end

    return FlowlineTrace(points, seed, angle, :max_steps, 0)
end

"""
    trace_flowline(flow, seed; angle=0, ...)
    trace_flowline(field::IGField, seed; angle=0, ...)

Trace a single discrete mesh-regularized imaginary-geometry proxy curve on a
square.

Boundary behavior is intentionally not configurable.  Each finite-step overshoot
is clamped back into the active square, after which subsequent motion follows
the projected local direction field.  The trace continues until it reaches the
chordal target, hits a custom stop condition, or exhausts `max_steps`.
"""
function trace_flowline(
    flow::FlowlineField{T},
    seed::Complex;
    angle::Real=0,
    ds::Union{Nothing,Real}=nothing,
    max_steps::Union{Nothing,Integer}=nothing,
    boundary_margin::Union{Nothing,Real}=nothing,
    integrator::Symbol=:euler,
    stop_when::Union{Nothing,Function}=nothing,
    phase_interpolation::Symbol=:tip_phase,
    min_ds_factor::Union{Nothing,Real}=nothing,
    goal_capture_steps::Union{Nothing,Real}=2,
    adaptive_stuck_steps::Union{Nothing,Integer}=32,
    adaptive_stuck_growth::Real=2,
    adaptive_max_scale::Real=64,
    adaptive_stuck_radius::Real=24,
) where {T<:AbstractFloat}
    zseed = complex(T(real(seed)), T(imag(seed)))
    return _trace_flowline_impl(
        flow,
        zseed,
        T(angle);
        ds=ds,
        max_steps=max_steps,
        boundary_margin=boundary_margin,
        integrator=integrator,
        stop_when=stop_when,
        phase_interpolation=phase_interpolation,
        min_ds_factor=min_ds_factor,
        goal_capture_steps=goal_capture_steps,
        adaptive_stuck_steps=adaptive_stuck_steps,
        adaptive_stuck_growth=adaptive_stuck_growth,
        adaptive_max_scale=adaptive_max_scale,
        adaptive_stuck_radius=adaptive_stuck_radius,
    )
end

"""
    trace_angle_fan(flow, seed, angles; kwargs...)

Trace several mesh-regularized proxy curves from a common starting point.
"""
function trace_angle_fan(flow::FlowlineField{T}, seed::Complex, angles; kwargs...) where {T<:AbstractFloat}
    angles_t = collect(angles)
    traces = Vector{FlowlineTrace{T}}(undef, length(angles_t))
    @threads for idx in eachindex(angles_t)
        traces[idx] = trace_flowline(flow, seed; angle=angles_t[idx], kwargs...)
    end
    return traces
end

function trace_angle_fan(
    field::RandomFieldGenerators.IGField,
    seed::Complex,
    angles;
    multiscale::Bool=false,
    levels::Union{Nothing,Integer}=nothing,
    min_cutoff::Union{Nothing,Integer}=nothing,
    spectral_oversample::Union{Nothing,Integer}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
    extension_seed::Union{Nothing,Integer}=nothing,
    kwargs...,
)
    clean_kwargs = (; kwargs...)
    if multiscale
        backend_kwargs = Pair{Symbol,Any}[]
        isnothing(levels) || push!(backend_kwargs, :levels => Int(levels))
        isnothing(min_cutoff) || push!(backend_kwargs, :min_cutoff => Int(min_cutoff))
        isnothing(spectral_oversample) || push!(backend_kwargs, :spectral_oversample => Int(spectral_oversample))
        isnothing(rng) || push!(backend_kwargs, :rng => rng)
        isnothing(extension_seed) || push!(backend_kwargs, :extension_seed => Int(extension_seed))
        backend = multiscale_flowline_field(field; (; backend_kwargs...)...)
        return trace_angle_fan(backend, seed, angles; clean_kwargs...)
    end
    return trace_angle_fan(flowline_field(field), seed, angles; clean_kwargs...)
end

function _refine_segment_points(
    flow::FlowlineField{T},
    zstart::Complex{T},
    ztarget::Complex{T},
    angle::T,
    ds::T,
    margin::T,
    integrator::Symbol,
    refinement_slack::T,
    join_factor::T,
    phase_interpolation::Symbol,
    goal_capture_steps::Union{Nothing,Real},
    adaptive_stuck_steps::Union{Nothing,Integer},
    adaptive_stuck_growth::Real,
    adaptive_max_scale::Real,
    adaptive_stuck_radius::Real,
) where {T<:AbstractFloat}
    seglen = abs(ztarget - zstart)
    seglen <= ds && return Complex{T}[ztarget], :linear_short

    join_tol = max(join_factor * ds, seglen / T(6))
    local_steps = max(4, ceil(Int, refinement_slack * seglen / ds))
    tangent = (ztarget - zstart) / seglen
    stop_when = (z, _, _) -> begin
        progress = real(conj(tangent) * (z - zstart))
        progress >= seglen - join_tol || abs(z - ztarget) <= join_tol
    end

    local_trace = _trace_flowline_impl(
        flow,
        zstart,
        angle;
        ds=ds,
        max_steps=local_steps,
        boundary_margin=margin,
        integrator=integrator,
        stop_when=stop_when,
        phase_interpolation=phase_interpolation,
        goal_capture_steps=goal_capture_steps,
        adaptive_stuck_steps=adaptive_stuck_steps,
        adaptive_stuck_growth=adaptive_stuck_growth,
        adaptive_max_scale=adaptive_max_scale,
        adaptive_stuck_radius=adaptive_stuck_radius,
    )

    if local_trace.termination === :stop_condition && length(local_trace.points) > 1
        refined = local_trace.points[2:end]
        if abs(refined[end] - ztarget) <= join_tol
            refined[end] = ztarget
        else
            _append_connector!(refined, refined[end], ztarget, ds)
        end
        return refined, :refined
    end

    refined = Complex{T}[]
    _append_connector!(refined, zstart, ztarget, ds)
    return refined, :fallback
end

function _refine_trace_points(
    points::AbstractVector{Complex{T}},
    flow::FlowlineField{T},
    angle::T;
    ds::T,
    boundary_margin::T,
    integrator::Symbol,
    refinement_slack::T,
    join_factor::T,
    phase_interpolation::Symbol,
    goal_capture_steps::Union{Nothing,Real},
    adaptive_stuck_steps::Union{Nothing,Integer},
    adaptive_stuck_growth::Real,
    adaptive_max_scale::Real,
    adaptive_stuck_radius::Real,
) where {T<:AbstractFloat}
    length(points) <= 1 && return Vector{Complex{T}}(points), :degenerate

    refined = Complex{T}[first(points)]
    status = :refined

    for idx in 1:(length(points) - 1)
        zstart = refined[end]
        ztarget = points[idx + 1]
        segment_points, segment_status = _refine_segment_points(
            flow,
            zstart,
            ztarget,
            angle,
            ds,
            boundary_margin,
            integrator,
            refinement_slack,
            join_factor,
            phase_interpolation,
            goal_capture_steps,
            adaptive_stuck_steps,
            adaptive_stuck_growth,
            adaptive_max_scale,
            adaptive_stuck_radius,
        )
        append!(refined, segment_points)
        segment_status === :fallback && (status = :fallback)
    end

    return refined, status
end

"""
    trace_flowline(multiscale, seed; angle=0, ...)

Trace a multiscale proxy by first integrating a coarse regularization and then
recursively refining the resulting polyline with progressively finer filtered
fields. This is still a proxy, but it reflects a vanishing-regularization
scheme better than the single-cutoff streamline.
"""
function trace_flowline(
    multiscale::MultiscaleFlowlineField{T},
    seed::Complex;
    angle::Real=0,
    max_steps::Union{Nothing,Integer}=nothing,
    boundary_margin::Union{Nothing,Real}=nothing,
    integrator::Symbol=:euler,
    step_factor::Real=0.05,
    refinement_slack::Real=2.75,
    join_factor::Real=1.25,
    phase_interpolation::Symbol=:tip_phase,
    min_ds_factor::Union{Nothing,Real}=nothing,
    goal_capture_steps::Union{Nothing,Real}=2,
    adaptive_stuck_steps::Union{Nothing,Integer}=32,
    adaptive_stuck_growth::Real=2,
    adaptive_max_scale::Real=64,
    adaptive_stuck_radius::Real=24,
) where {T<:AbstractFloat}
    zseed = complex(T(real(seed)), T(imag(seed)))
    angleT = T(angle)
    margin = isnothing(boundary_margin) ? T(1.1) * min(multiscale.domain.hx, multiscale.domain.hy) : T(boundary_margin)
    margin >= zero(T) || throw(ArgumentError("`boundary_margin` must be non-negative."))
    cutoff_full = multiscale.cutoffs[end]

    coarse_flow = multiscale.levels[1]
    coarse_ds = _level_step(multiscale.domain, cutoff_full, multiscale.cutoffs[1]; step_factor=step_factor)
    coarse_trace = _trace_flowline_impl(
        coarse_flow,
        zseed,
        angleT;
        ds=coarse_ds,
        max_steps=max_steps,
        boundary_margin=margin,
        integrator=integrator,
        phase_interpolation=phase_interpolation,
        min_ds_factor=min_ds_factor,
        goal_capture_steps=goal_capture_steps,
        adaptive_stuck_steps=adaptive_stuck_steps,
        adaptive_stuck_growth=adaptive_stuck_growth,
        adaptive_max_scale=adaptive_max_scale,
        adaptive_stuck_radius=adaptive_stuck_radius,
    )

    points = coarse_trace.points
    termination = coarse_trace.termination

    for level_idx in 2:length(multiscale.levels)
        ds = _level_step(multiscale.domain, cutoff_full, multiscale.cutoffs[level_idx]; step_factor=step_factor)
        points, refine_status = _refine_trace_points(
            points,
            multiscale.levels[level_idx],
            angleT;
            ds=ds,
            boundary_margin=margin,
            integrator=integrator,
            refinement_slack=T(refinement_slack),
            join_factor=T(join_factor),
            phase_interpolation=phase_interpolation,
            goal_capture_steps=goal_capture_steps,
            adaptive_stuck_steps=adaptive_stuck_steps,
            adaptive_stuck_growth=adaptive_stuck_growth,
            adaptive_max_scale=adaptive_max_scale,
            adaptive_stuck_radius=adaptive_stuck_radius,
        )
        refine_status === :fallback && (termination = :multiscale_fallback)
    end

    return FlowlineTrace(points, zseed, angleT, termination, 0)
end

function trace_flowline(
    field::RandomFieldGenerators.IGField{T},
    seed::Complex;
    angle::Union{Nothing,Real}=nothing,
    multiscale::Bool=false,
    levels::Union{Nothing,Integer}=nothing,
    min_cutoff::Union{Nothing,Integer}=nothing,
    spectral_oversample::Union{Nothing,Integer}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
    extension_seed::Union{Nothing,Integer}=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    default_angle = zero(T)
    angle_value = isnothing(angle) ? default_angle : T(angle)
    clean_kwargs = (; kwargs...)
    if multiscale
        backend_kwargs = Pair{Symbol,Any}[]
        isnothing(levels) || push!(backend_kwargs, :levels => Int(levels))
        isnothing(min_cutoff) || push!(backend_kwargs, :min_cutoff => Int(min_cutoff))
        isnothing(spectral_oversample) || push!(backend_kwargs, :spectral_oversample => Int(spectral_oversample))
        isnothing(rng) || push!(backend_kwargs, :rng => rng)
        isnothing(extension_seed) || push!(backend_kwargs, :extension_seed => Int(extension_seed))
        backend = multiscale_flowline_field(field; (; backend_kwargs...)...)
        return trace_flowline(backend, seed; angle=angle_value, clean_kwargs...)
    end
    return trace_flowline(flowline_field(field), seed; angle=angle_value, clean_kwargs...)
end

function trace_angle_fan(multiscale::MultiscaleFlowlineField{T}, seed::Complex, angles; kwargs...) where {T<:AbstractFloat}
    angles_t = collect(angles)
    traces = Vector{FlowlineTrace{T}}(undef, length(angles_t))
    @threads for idx in eachindex(angles_t)
        traces[idx] = trace_flowline(multiscale, seed; angle=angles_t[idx], kwargs...)
    end
    return traces
end

function _append_connector!(
    points::Vector{Complex{T}},
    z1::Complex{T},
    z2::Complex{T},
    max_segment::T,
) where {T<:AbstractFloat}
    dz = z2 - z1
    safe_segment = max(max_segment, eps(T))
    nseg = min(4096, max(1, ceil(Int, abs(dz) / safe_segment)))
    @inbounds for k in 1:nseg
        t = T(k) / T(nseg)
        push!(points, z1 + t * dz)
    end
    return nothing
end

function _collect_angles(::Type{T}, angles) where {T<:AbstractFloat}
    return T.(collect(angles))
end

function _centered_angles(::Type{T}, angle::Real, theta_left::Real, theta_right::Real, nangles::Integer) where {T<:AbstractFloat}
    n = Int(nangles)
    n >= 1 || throw(ArgumentError("`nangles` must be positive."))
    center = T(angle)
    return n == 1 ? T[center] : collect(range(center - T(theta_left), center + T(theta_right); length=n))
end

function _default_trace_spacing(
    domain::RandomFieldGenerators.SquareDomain{T};
    ds=nothing,
    step_factor=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    h = min(domain.hx, domain.hy)
    if !isnothing(ds)
        spacing = T(ds)
    elseif !isnothing(step_factor)
        spacing = T(step_factor) * h
    else
        spacing = T(0.02) * h
    end
    return max(spacing, eps(T))
end

function _default_fan_seed(
    field::RandomFieldGenerators.IGField{T};
    ds=nothing,
    step_factor=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    spacing = _default_trace_spacing(field.domain; ds=ds, step_factor=step_factor)
    xmid = (field.domain.xmin + field.domain.xmax) / T(2)
    return complex(xmid, field.domain.ymin + spacing)
end

"""
    trace_sle_fan(field, seed, angles; multiscale=false, kwargs...)

Trace the maintained square chordal IG/SLE proxy fan on top of a sampled IG
field. This is a thin convenience wrapper around `trace_angle_fan` that keeps
the public surface focused on the current fan workflow while still allowing the
same tracing keywords as the lower-level API. The return value is a named tuple
with `:traces`, `:angles`, `:seed`, and `:multiscale`.
"""
function trace_sle_fan(
    field::RandomFieldGenerators.IGField{T},
    seed::Complex,
    angles;
    multiscale::Bool=false,
    levels::Union{Nothing,Integer}=nothing,
    min_cutoff::Union{Nothing,Integer}=nothing,
    spectral_oversample::Union{Nothing,Integer}=nothing,
    rng::Union{Nothing,AbstractRNG}=nothing,
    extension_seed::Union{Nothing,Integer}=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    angle_values = _collect_angles(T, angles)
    zseed = complex(T(real(seed)), T(imag(seed)))
    traces = trace_angle_fan(
        field,
        zseed,
        angle_values;
        multiscale=multiscale,
        levels=levels,
        min_cutoff=min_cutoff,
        spectral_oversample=spectral_oversample,
        rng=rng,
        extension_seed=extension_seed,
        kwargs...,
    )

    return (
        kind=:fan,
        field=field,
        traces=traces,
        angles=angle_values,
        seed=zseed,
        multiscale=multiscale,
    )
end

function trace_sle_fan(
    field::RandomFieldGenerators.IGField{T};
    seed::Union{Nothing,Complex}=nothing,
    angle::Real=0,
    theta_left::Real=pi / 2,
    theta_right::Real=pi / 2,
    nangles::Integer=21,
    angles=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    zseed = isnothing(seed) ? _default_fan_seed(field; kwargs...) : seed
    angle_values = isnothing(angles) ? _centered_angles(T, angle, theta_left, theta_right, nangles) : _collect_angles(T, angles)
    return trace_sle_fan(field, zseed, angle_values; kwargs...)
end

end # module
