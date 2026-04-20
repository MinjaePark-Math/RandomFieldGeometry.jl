using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using DelimitedFiles
using Printf
using RandomFieldGeometry
using Statistics

# Visualization of the finite-grid two-sided phase law suggested by
# Theorem 1.1 in the mixed Liouville observable paper draft.
#
# This example stays on the current public package surface:
#   - `sample_chordal_square_ig_field`
#   - `trace_angle_fan`
#   - `free_square_gff`
#   - `domain_coordinates`
#
# It is still a numerical proxy. The theorem is about continuum / curve-supported
# limiting observables, while this script measures a finite-grid analogue on the
# sampled square-domain IG flow line. The curve `eta` is sampled from the same
# fan-tracing setup used by `examples/run_sle_fan.jl`, with `nflow=1` by default.
#
# Example:
#   julia --project examples/ig_mixed_phase_law.jl
#   julia --project examples/ig_mixed_phase_law.jl grid=257 kappa=2 alpha=0.35 beta=0.55 multiscale=true

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
parse_real_expr(text::AbstractString) = parse_real_expr(Meta.parse(text))

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

parse_symbol_text(text) = Symbol(replace(lowercase(strip(String(text))), r"^:" => ""))
get_opt(opts::Dict{String,String}, key::AbstractString, default) = get(opts, key, default)

function parse_int_expr(text::AbstractString)
    value = parse_real_expr(text)
    rounded = round(value)
    if !isfinite(value) || abs(value - rounded) > 100 * eps(Float64) * max(1.0, abs(value))
        throw(ArgumentError("expected an integer expression, got `$text`."))
    end
    return Int(rounded)
end

function centered_angles(center::Real, left::Real, right::Real, n::Integer)
    n = Int(n)
    n >= 1 || throw(ArgumentError("number of angles must be positive."))
    return n == 1 ? [Float64(center)] : collect(range(Float64(center - left), Float64(center + right); length=n))
end

struct CurveFrames
    s::Vector{Float64}
    z::Vector{ComplexF64}
    tangent::Vector{ComplexF64}
    normal::Vector{ComplexF64}
    ds::Vector{Float64}
    total_length::Float64
end

struct ArrowDiagnostics
    centers::Vector{ComplexF64}
    left_anchors::Vector{ComplexF64}
    right_anchors::Vector{ComplexF64}
    ZL::Vector{ComplexF64}
    ZR::Vector{ComplexF64}
    aligned_ZR::Vector{ComplexF64}
    phase_gap::Vector{Float64}
    unwrapped_gap::Vector{Float64}
    weights::Vector{Float64}
    predicted_phase::Float64
    empirical_phase::Float64
    empirical_error::Float64
end

wrap_to_pi(x::Real) = mod(Float64(x) + pi, 2pi) - pi

function weighted_circular_mean(phases::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    isempty(phases) && return NaN
    z = 0.0 + 0.0im
    for i in eachindex(phases)
        z += max(0.0, Float64(weights[i])) * cis(Float64(phases[i]))
    end
    abs(z) <= 1e-14 && return NaN
    return angle(z)
end

function unwrap_near(phases::AbstractVector{<:Real}, target::Real)
    isempty(phases) && return Float64[]
    out = Vector{Float64}(undef, length(phases))
    prev = Float64(phases[1])
    while prev - target > pi
        prev -= 2pi
    end
    while prev - target < -pi
        prev += 2pi
    end
    out[1] = prev
    for i in 2:length(phases)
        phi = Float64(phases[i])
        while phi - prev > pi
            phi -= 2pi
        end
        while phi - prev < -pi
            phi += 2pi
        end
        out[i] = phi
        prev = phi
    end
    return out
end

function smooth_vector(x::Vector{Float64}; radius::Int=2)
    n = length(x)
    n == 0 && return copy(x)
    y = similar(x)
    for i in 1:n
        lo = max(1, i - radius)
        hi = min(n, i + radius)
        y[i] = mean(@view x[lo:hi])
    end
    return y
end

function sample_independent_gff(n::Integer; boundary::Symbol=:neumann, seed::Integer=20260419, normalize_std::Bool=true)
    n = Int(n)
    if boundary === :neumann || boundary === :free
        out = Float64.(free_square_gff(n, seed; T=Float64))
    elseif boundary === :dirichlet
        out = zeros(Float64, n, n)
        interior = dirichlet_gff(2, n - 1, seed; T=Float64)
        @views out[2:(n - 1), 2:(n - 1)] .= Float64.(interior)
        out .-= mean(out)
    else
        throw(ArgumentError("gff_boundary must be :neumann/:free or :dirichlet."))
    end

    if normalize_std
        s = std(vec(out))
        s > 0 && (out ./= s)
    end
    return out
end

@inline function grid_index_from_point(z::ComplexF64, xs::AbstractVector, ys::AbstractVector)
    hx = xs[2] - xs[1]
    hy = ys[2] - ys[1]
    i = clamp(round(Int, (real(z) - xs[1]) / hx) + 1, 1, length(xs))
    j = clamp(round(Int, (imag(z) - ys[1]) / hy) + 1, 1, length(ys))
    return i, j
end

@inline function inside_box(z::ComplexF64, xs::AbstractVector, ys::AbstractVector; margin::Real=0.0)
    return xs[1] + margin <= real(z) <= xs[end] - margin &&
           ys[1] + margin <= imag(z) <= ys[end] - margin
end

function bilinear(mat::AbstractMatrix{<:Real}, xs::AbstractVector, ys::AbstractVector, z::ComplexF64; default::Float64=NaN)
    x = real(z)
    y = imag(z)
    nx, ny = size(mat)
    if x < xs[1] || x > xs[end] || y < ys[1] || y > ys[end]
        return default
    end
    hx = xs[2] - xs[1]
    hy = ys[2] - ys[1]
    tx = (x - xs[1]) / hx + 1
    ty = (y - ys[1]) / hy + 1
    i = clamp(floor(Int, tx), 1, nx - 1)
    j = clamp(floor(Int, ty), 1, ny - 1)
    ax = tx - i
    ay = ty - j
    v00 = Float64(mat[i, j])
    v10 = Float64(mat[i + 1, j])
    v01 = Float64(mat[i, j + 1])
    v11 = Float64(mat[i + 1, j + 1])
    return (1 - ax) * (1 - ay) * v00 + ax * (1 - ay) * v10 + (1 - ax) * ay * v01 + ax * ay * v11
end

function curve_arclength_data(curve::Vector{ComplexF64})
    length(curve) >= 2 || error("curve must contain at least two points")
    seglen = [abs(curve[k + 1] - curve[k]) for k in 1:(length(curve) - 1)]
    cum = zeros(Float64, length(curve))
    for k in 2:length(curve)
        cum[k] = cum[k - 1] + seglen[k - 1]
    end
    return seglen, cum
end

function frame_at_s(curve::Vector{ComplexF64}, seglen::Vector{Float64}, cum::Vector{Float64}, s::Real)
    total = cum[end]
    ss = clamp(Float64(s), 0.0, total)
    k = searchsortedlast(cum, ss)
    k = clamp(k, 1, length(seglen))
    len = max(seglen[k], 1e-14)
    t = clamp((ss - cum[k]) / len, 0.0, 1.0)
    z = curve[k] + t * (curve[k + 1] - curve[k])
    tangent = (curve[k + 1] - curve[k]) / len
    normal = 1im * tangent
    return z, tangent, normal
end

function curve_frames(curve::Vector{ComplexF64}; n_arrows::Integer=48, trim_fraction::Real=0.06)
    seglen, cum = curve_arclength_data(curve)
    total = cum[end]
    trim = clamp(Float64(trim_fraction), 0.0, 0.45)
    lo = trim * total
    hi = (1 - trim) * total
    n = Int(n_arrows)
    ss = n == 1 ? [(lo + hi) / 2] : collect(range(lo, hi; length=n))
    z = ComplexF64[]
    tangent = ComplexF64[]
    normal = ComplexF64[]
    for s in ss
        zz, tt, nn = frame_at_s(curve, seglen, cum, s)
        push!(z, zz)
        push!(tangent, tt)
        push!(normal, nn)
    end
    dss = fill((hi - lo) / max(n - 1, 1), n)
    return CurveFrames(Float64.(ss), z, tangent, normal, dss, total)
end

function linear_interp(sgrid::Vector{Float64}, vals::Vector{Float64}, s::Real)
    isempty(sgrid) && return 0.0
    ss = Float64(s)
    if ss <= sgrid[1]
        return vals[1]
    elseif ss >= sgrid[end]
        return vals[end]
    end
    k = searchsortedlast(sgrid, ss)
    k = clamp(k, 1, length(sgrid) - 1)
    t = (ss - sgrid[k]) / max(sgrid[k + 1] - sgrid[k], 1e-14)
    return (1 - t) * vals[k] + t * vals[k + 1]
end

function estimate_bsym_trace(psi::AbstractMatrix{<:Real}, xs, ys, frames::CurveFrames; offset::Real, smooth_radius::Integer=2)
    raw = Float64[]
    for k in eachindex(frames.z)
        zl = frames.z[k] + offset * frames.normal[k]
        zr = frames.z[k] - offset * frames.normal[k]
        vl = bilinear(psi, xs, ys, zl; default=NaN)
        vr = bilinear(psi, xs, ys, zr; default=NaN)
        if isnan(vl) && isnan(vr)
            push!(raw, 0.0)
        elseif isnan(vl)
            push!(raw, vr)
        elseif isnan(vr)
            push!(raw, vl)
        else
            push!(raw, 0.5 * (vl + vr))
        end
    end
    return smooth_vector(raw; radius=Int(smooth_radius))
end

function mark_disk!(mask::BitMatrix, values::Matrix{Float64}, counts::Matrix{Int}, i0::Int, j0::Int, radius_cells::Int, value::Float64)
    nx, ny = size(mask)
    r = max(0, radius_cells)
    for dj in -r:r, di in -r:r
        di^2 + dj^2 <= r^2 || continue
        i = i0 + di
        j = j0 + dj
        if 1 <= i <= nx && 1 <= j <= ny
            mask[i, j] = true
            values[i, j] += value
            counts[i, j] += 1
        end
    end
end

function rasterize_curve(curve::Vector{ComplexF64}, xs, ys, frames::CurveFrames, bsym::Vector{Float64}; radius_cells::Integer=1, samples_per_cell::Real=2.0)
    nx = length(xs)
    ny = length(ys)
    cut = falses(nx, ny)
    acc = zeros(Float64, nx, ny)
    cnt = zeros(Int, nx, ny)
    seglen, cum = curve_arclength_data(curve)
    h = min(xs[2] - xs[1], ys[2] - ys[1])
    r = Int(radius_cells)

    function rasterize_segment!(z1::ComplexF64, z2::ComplexF64, value::Float64)
        len = abs(z2 - z1)
        m = max(2, ceil(Int, samples_per_cell * len / h))
        for a in 0:m
            t = a / m
            z = (1 - t) * z1 + t * z2
            i, j = grid_index_from_point(z, xs, ys)
            mark_disk!(cut, acc, cnt, i, j, r, value)
        end
        return nothing
    end

    for k in 1:(length(curve) - 1)
        len = seglen[k]
        m = max(2, ceil(Int, samples_per_cell * len / h))
        for a in 0:m
            t = a / m
            z = (1 - t) * curve[k] + t * curve[k + 1]
            s = cum[k] + t * len
            b = linear_interp(frames.s, bsym, s)
            i, j = grid_index_from_point(z, xs, ys)
            mark_disk!(cut, acc, cnt, i, j, r, b)
        end
    end

    south_mid = complex((xs[1] + xs[end]) / 2, ys[1])
    north_mid = complex((xs[1] + xs[end]) / 2, ys[end])
    start_value = isempty(bsym) ? 0.0 : first(bsym)
    end_value = isempty(bsym) ? start_value : last(bsym)
    rasterize_segment!(south_mid, first(curve), start_value)
    rasterize_segment!(last(curve), north_mid, end_value)

    cutval = fill(NaN, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        if cnt[i, j] > 0
            cutval[i, j] = acc[i, j] / cnt[i, j]
        end
    end

    for _ in 1:16
        changed = false
        newval = copy(cutval)
        @inbounds for j in 1:ny, i in 1:nx
            if cut[i, j] && isnan(cutval[i, j])
                s = 0.0
                c = 0
                for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
                    ii = i + di
                    jj = j + dj
                    if 1 <= ii <= nx && 1 <= jj <= ny && !isnan(cutval[ii, jj])
                        s += cutval[ii, jj]
                        c += 1
                    end
                end
                if c > 0
                    newval[i, j] = s / c
                    changed = true
                end
            end
        end
        cutval = newval
        !changed && break
    end

    fallback = isempty(bsym) ? 0.0 : mean(bsym)
    @inbounds for j in 1:ny, i in 1:nx
        if cut[i, j] && isnan(cutval[i, j])
            cutval[i, j] = fallback
        end
    end
    return cut, cutval
end

function nearest_allowed_seed(z::ComplexF64, xs, ys, allowed::BitMatrix; max_radius::Int=40)
    i0, j0 = grid_index_from_point(z, xs, ys)
    nx, ny = size(allowed)
    if allowed[i0, j0]
        return i0, j0
    end
    for r in 1:max_radius
        for dj in -r:r, di in -r:r
            abs(di) == r || abs(dj) == r || continue
            i = i0 + di
            j = j0 + dj
            if 1 <= i <= nx && 1 <= j <= ny && allowed[i, j]
                return i, j
            end
        end
    end
    error("could not find an allowed seed cell near $z; try increasing `cut_radius_cells`.")
end

function flood_component(allowed::BitMatrix, seed::Tuple{Int,Int})
    nx, ny = size(allowed)
    seen = falses(nx, ny)
    q = Vector{Tuple{Int,Int}}()
    i0, j0 = seed
    if !(1 <= i0 <= nx && 1 <= j0 <= ny && allowed[i0, j0])
        return seen
    end
    push!(q, seed)
    seen[i0, j0] = true
    head = 1
    while head <= length(q)
        i, j = q[head]
        head += 1
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii = i + di
            jj = j + dj
            if 1 <= ii <= nx && 1 <= jj <= ny && allowed[ii, jj] && !seen[ii, jj]
                seen[ii, jj] = true
                push!(q, (ii, jj))
            end
        end
    end
    return seen
end

function left_right_masks(cut::BitMatrix, xs, ys, frames::CurveFrames; seed_offset::Real)
    allowed = .!cut
    mid = max(1, div(length(frames.z), 2))
    zc = frames.z[mid]
    nrm = frames.normal[mid]
    left_seed = nearest_allowed_seed(zc + seed_offset * nrm, xs, ys, allowed)
    right_seed = nearest_allowed_seed(zc - seed_offset * nrm, xs, ys, allowed)
    left = flood_component(allowed, left_seed)
    right = flood_component(allowed, right_seed)
    overlap = left .& right
    if any(overlap)
        @warn "left and right flood fills overlap; the cut may not separate the square cleanly. Try increasing `cut_radius_cells`."
    end
    return left .& .!overlap, right .& .!overlap
end

function initialize_harmonic_array(boundary::AbstractMatrix{<:Real}, cut::BitMatrix, cutval::Matrix{Float64}, lambda::Real, side::Symbol)
    nx, ny = size(boundary)
    H = Matrix{Float64}(boundary)
    sign = side === :left ? -1.0 : +1.0
    @inbounds for j in 1:ny, i in 1:nx
        if cut[i, j]
            H[i, j] = cutval[i, j] + sign * Float64(lambda)
        elseif 1 < i < nx && 1 < j < ny
            H[i, j] = 0.0
        end
    end
    return H
end

function relax_harmonic!(H::Matrix{Float64}, unknown::BitMatrix; iterations::Integer=700, omega::Real=1.55, tol::Real=1e-5)
    nx, ny = size(H)
    omega_f = Float64(omega)
    for it in 1:Int(iterations)
        maxdiff = 0.0
        @inbounds for color in 0:1
            for j in 2:(ny - 1), i in 2:(nx - 1)
                unknown[i, j] || continue
                ((i + j) & 1) == color || continue
                newv = 0.25 * (H[i + 1, j] + H[i - 1, j] + H[i, j + 1] + H[i, j - 1])
                upd = (1 - omega_f) * H[i, j] + omega_f * newv
                d = abs(upd - H[i, j])
                d > maxdiff && (maxdiff = d)
                H[i, j] = upd
            end
        end
        maxdiff < tol && return it, maxdiff
    end
    return Int(iterations), NaN
end

function harmonic_side_fields(field, xs, ys, curve::Vector{ComplexF64}, frames::CurveFrames, bsym::Vector{Float64};
        cut_radius_cells::Integer=2,
        seed_offset::Real=0.03,
        harmonic_iterations::Integer=700,
        harmonic_omega::Real=1.55,
        samples_per_cell::Real=2.0)
    cut, cutval = rasterize_curve(curve, xs, ys, frames, bsym; radius_cells=cut_radius_cells, samples_per_cell=samples_per_cell)
    leftmask, rightmask = left_right_masks(cut, xs, ys, frames; seed_offset=seed_offset)
    boundary = Matrix{Float64}(field.boundary)
    lambda = Float64(field.constants.lambda)
    HL = initialize_harmonic_array(boundary, cut, cutval, lambda, :left)
    HR = initialize_harmonic_array(boundary, cut, cutval, lambda, :right)
    nx, ny = size(cut)
    unknownL = copy(leftmask)
    unknownR = copy(rightmask)
    @inbounds for j in 1:ny, i in 1:nx
        if i == 1 || i == nx || j == 1 || j == ny || cut[i, j]
            unknownL[i, j] = false
            unknownR[i, j] = false
        end
    end
    itL, errL = relax_harmonic!(HL, unknownL; iterations=harmonic_iterations, omega=harmonic_omega)
    itR, errR = relax_harmonic!(HR, unknownR; iterations=harmonic_iterations, omega=harmonic_omega)
    return (
        HL=HL,
        HR=HR,
        cut=cut,
        cutval=cutval,
        leftmask=leftmask,
        rightmask=rightmask,
        iterations_left=itL,
        iterations_right=itR,
        residual_left=errL,
        residual_right=errR,
    )
end

function compute_harmonic_arrow_diagnostic(HL, HR, hfield, xs, ys, frames::CurveFrames; alpha_obs::Real, beta_obs::Real, lambda::Real, sample_offset::Real, anchor_offset::Real)
    alpha = Float64(alpha_obs)
    beta = Float64(beta_obs)
    predicted = 2 * alpha * Float64(lambda)
    centers = ComplexF64[]
    lefta = ComplexF64[]
    righta = ComplexF64[]
    ZL = ComplexF64[]
    ZR = ComplexF64[]
    gaps = Float64[]
    weights = Float64[]
    for k in eachindex(frames.z)
        zc = frames.z[k]
        nrm = frames.normal[k]
        zl = zc + sample_offset * nrm
        zr = zc - sample_offset * nrm
        inside_box(zl, xs, ys) && inside_box(zr, xs, ys) || continue
        hmid = bilinear(hfield, xs, ys, zc; default=NaN)
        hl = bilinear(HL, xs, ys, zl; default=NaN)
        hr = bilinear(HR, xs, ys, zr; default=NaN)
        if any(isnan, (hmid, hl, hr))
            continue
        end
        w = exp(beta * hmid) * max(frames.ds[k], 1e-12)
        zL = w * cis(-alpha * hl)
        zR = w * cis(-alpha * hr)
        push!(centers, zc)
        push!(lefta, zc + anchor_offset * nrm)
        push!(righta, zc - anchor_offset * nrm)
        push!(ZL, zL)
        push!(ZR, zR)
        push!(gaps, angle(zL / zR))
        push!(weights, abs(zL) * abs(zR))
    end
    aligned = [ZR[i] * cis(predicted) for i in eachindex(ZR)]
    empirical = weighted_circular_mean(gaps, weights)
    err = wrap_to_pi(empirical - predicted)
    return ArrowDiagnostics(
        centers,
        lefta,
        righta,
        ZL,
        ZR,
        aligned,
        gaps,
        unwrap_near(gaps, predicted),
        weights,
        predicted,
        empirical,
        err,
    )
end

function compute_collar_arrow_diagnostic(psi, HL, HR, hfield, xs, ys, frames::CurveFrames; alpha_obs::Real, beta_obs::Real, lambda::Real, width::Real, n_offsets::Integer=5, anchor_offset::Real)
    alpha = Float64(alpha_obs)
    beta = Float64(beta_obs)
    predicted = 2 * alpha * Float64(lambda)
    centers = ComplexF64[]
    lefta = ComplexF64[]
    righta = ComplexF64[]
    ZL = ComplexF64[]
    ZR = ComplexF64[]
    gaps = Float64[]
    weights = Float64[]
    rs = collect(range(0.35 * width, width; length=max(1, Int(n_offsets))))
    for k in eachindex(frames.z)
        zc = frames.z[k]
        nrm = frames.normal[k]
        zL = 0.0 + 0.0im
        zR = 0.0 + 0.0im
        for r in rs
            for side in (:left, :right)
                z = side === :left ? zc + r * nrm : zc - r * nrm
                inside_box(z, xs, ys) || continue
                hv = bilinear(hfield, xs, ys, z; default=NaN)
                psi_v = bilinear(psi, xs, ys, z; default=NaN)
                Hq = side === :left ? bilinear(HL, xs, ys, z; default=NaN) : bilinear(HR, xs, ys, z; default=NaN)
                any(isnan, (hv, psi_v, Hq)) && continue
                psi0 = psi_v - Hq
                local_weight = exp(beta * hv) * max(frames.ds[k], 1e-12) * width / length(rs)
                val = local_weight * cis(-alpha * (Hq + psi0))
                if side === :left
                    zL += val
                else
                    zR += val
                end
            end
        end
        if abs(zL) > 1e-14 && abs(zR) > 1e-14
            push!(centers, zc)
            push!(lefta, zc + anchor_offset * nrm)
            push!(righta, zc - anchor_offset * nrm)
            push!(ZL, zL)
            push!(ZR, zR)
            push!(gaps, angle(zL / zR))
            push!(weights, abs(zL) * abs(zR))
        end
    end
    aligned = [ZR[i] * cis(predicted) for i in eachindex(ZR)]
    empirical = weighted_circular_mean(gaps, weights)
    err = wrap_to_pi(empirical - predicted)
    return ArrowDiagnostics(
        centers,
        lefta,
        righta,
        ZL,
        ZR,
        aligned,
        gaps,
        unwrap_near(gaps, predicted),
        weights,
        predicted,
        empirical,
        err,
    )
end

function direction_samples(psi, xs, ys, chi::Real; angle::Real=0.0, stride::Integer=10, arrow_len::Real=0.06)
    pts = ComplexF64[]
    vecs = ComplexF64[]
    theta0 = Float64(angle) + pi / 2
    for j in 2:Int(stride):(length(ys) - 1), i in 2:Int(stride):(length(xs) - 1)
        z = complex(xs[i], ys[j])
        v = Float64(arrow_len) * cis(Float64(psi[i, j]) / Float64(chi) + theta0)
        push!(pts, z)
        push!(vecs, v)
    end
    return pts, vecs
end

function sample_eta_from_fan(field; theta_center::Real=0.0, theta_left::Real=0.0, theta_right::Real=0.0, nflow::Integer=1, ds::Real, ds_over_h::Real, max_steps::Integer, goal_capture_steps::Real=3.0, integrator::Symbol=:euler, multiscale::Bool=false)
    seed = complex((field.domain.xmin + field.domain.xmax) / 2, field.domain.ymin + ds)
    angles = centered_angles(theta_center, theta_left, theta_right, nflow)
    traces = if multiscale
        trace_angle_fan(
            field,
            seed,
            angles;
            max_steps=max_steps,
            boundary_margin=0.0,
            integrator=integrator,
            phase_interpolation=:tip_phase,
            goal_capture_steps=goal_capture_steps,
            multiscale=true,
            step_factor=ds_over_h,
        )
    else
        trace_angle_fan(
            field,
            seed,
            angles;
            ds=ds,
            max_steps=max_steps,
            boundary_margin=0.0,
            integrator=integrator,
            phase_interpolation=:tip_phase,
            goal_capture_steps=goal_capture_steps,
            multiscale=false,
        )
    end
    eta_index = argmin(abs.(angles .- Float64(theta_center)))
    return (
        seed=ComplexF64(seed),
        angles=angles,
        traces=traces,
        eta_index=eta_index,
        trace=traces[eta_index],
    )
end

function scaled_arrow_vectors(zs::Vector{ComplexF64}; base_length::Real, quantile::Real=0.85, max_factor::Real=1.8)
    mags = abs.(zs)
    if isempty(mags)
        return ComplexF64[]
    end
    idx = clamp(round(Int, quantile * (length(mags) - 1)) + 1, 1, length(mags))
    ref = max(sort(mags)[idx], 1e-12)
    out = ComplexF64[]
    for z in zs
        if abs(z) <= 1e-14
            push!(out, 0 + 0im)
        else
            push!(out, base_length * min(abs(z) / ref, max_factor) * z / abs(z))
        end
    end
    return out
end

function write_arrow_table(path::AbstractString, diag::ArrowDiagnostics)
    table = hcat(
        real.(diag.centers),
        imag.(diag.centers),
        real.(diag.left_anchors),
        imag.(diag.left_anchors),
        real.(diag.ZL),
        imag.(diag.ZL),
        real.(diag.right_anchors),
        imag.(diag.right_anchors),
        real.(diag.ZR),
        imag.(diag.ZR),
        real.(diag.aligned_ZR),
        imag.(diag.aligned_ZR),
        diag.phase_gap,
        diag.unwrapped_gap,
        diag.weights,
    )
    writedlm(path, table, '\t')
end

function save_field_heatmaps(outdir, xs, ys, psi, hfield, curve)
    fig = Figure(size=(1280, 540))
    ax1 = Axis(fig[1, 1], aspect=1, title="Imaginary-geometry field psi", xlabel="Re z", ylabel="Im z")
    hm1 = heatmap!(ax1, xs, ys, Matrix(psi)'; colormap=:balance)
    lines!(ax1, real.(curve), imag.(curve); color=:black, linewidth=2.5)
    scatter!(ax1, [0.0, 0.0], [ys[1], ys[end]]; color=:black, markersize=8)
    Colorbar(fig[1, 2], hm1)

    ax2 = Axis(fig[1, 3], aspect=1, title="Independent free-boundary GFF h", xlabel="Re z", ylabel="Im z")
    hm2 = heatmap!(ax2, xs, ys, Matrix(hfield)'; colormap=:balance)
    lines!(ax2, real.(curve), imag.(curve); color=:black, linewidth=2.5)
    Colorbar(fig[1, 4], hm2)

    save(joinpath(outdir, "field_heatmaps.png"), fig)
end

function save_direction_plot(outdir, xs, ys, psi, curve, family_curves, dir_pts, dir_vecs)
    fig = Figure(size=(900, 860))
    ax = Axis(fig[1, 1], aspect=1, title="Direction field and north-going proxy traces", xlabel="Re z", ylabel="Im z")
    hm = heatmap!(ax, xs, ys, Matrix(psi)'; colormap=:balance)
    Colorbar(fig[1, 2], hm)
    arrows2d!(ax, real.(dir_pts), imag.(dir_pts), real.(dir_vecs), imag.(dir_vecs); tiplength=9, tipwidth=9, shaftwidth=1.5, lengthscale=1.0, color=:white)
    for c in family_curves
        lines!(ax, real.(c), imag.(c); color=(:gray20, 0.65), linewidth=1.2)
    end
    lines!(ax, real.(curve), imag.(curve); color=:black, linewidth=3.0)
    save(joinpath(outdir, "direction_field_and_flowlines.png"), fig)
end

function save_phase_arrow_plot(outdir, xs, curve, diag::ArrowDiagnostics; filename::AbstractString, title_prefix::AbstractString)
    fig = Figure(size=(1280, 620))
    base = 0.055 * (xs[end] - xs[1])
    vL = scaled_arrow_vectors(diag.ZL; base_length=base)
    vR = scaled_arrow_vectors(diag.ZR; base_length=base)
    vRa = scaled_arrow_vectors(diag.aligned_ZR; base_length=base)

    ax1 = Axis(fig[1, 1], aspect=1, title=title_prefix * " - raw left/right", xlabel="Re z", ylabel="Im z")
    lines!(ax1, real.(curve), imag.(curve); color=:black, linewidth=2.8)
    arrows2d!(ax1, real.(diag.left_anchors), imag.(diag.left_anchors), real.(vL), imag.(vL); tiplength=10, tipwidth=10, shaftwidth=2.2, color=:dodgerblue)
    arrows2d!(ax1, real.(diag.right_anchors), imag.(diag.right_anchors), real.(vR), imag.(vR); tiplength=10, tipwidth=10, shaftwidth=2.2, color=:darkorange)

    ax2 = Axis(fig[1, 2], aspect=1, title=@sprintf("right arrows rotated by predicted gap %.3f", diag.predicted_phase), xlabel="Re z", ylabel="Im z")
    lines!(ax2, real.(curve), imag.(curve); color=:black, linewidth=2.8)
    arrows2d!(ax2, real.(diag.left_anchors), imag.(diag.left_anchors), real.(vL), imag.(vL); tiplength=10, tipwidth=10, shaftwidth=2.2, color=:dodgerblue)
    arrows2d!(ax2, real.(diag.right_anchors), imag.(diag.right_anchors), real.(vRa), imag.(vRa); tiplength=10, tipwidth=10, shaftwidth=2.2, color=:darkorange)

    Label(
        fig[2, 1:2],
        @sprintf(
            "empirical weighted circular mean = %.4f; predicted = %.4f; error = %.4f radians",
            diag.empirical_phase,
            diag.predicted_phase,
            diag.empirical_error,
        ),
    )
    save(joinpath(outdir, String(filename)), fig)
end

function save_phase_profile(outdir, diag::ArrowDiagnostics; filename::AbstractString)
    fig = Figure(size=(880, 480))
    ax = Axis(fig[1, 1], title="Local left/right phase gap along the curve", xlabel="window index", ylabel="phase gap (radians)")
    idx = collect(1:length(diag.unwrapped_gap))
    lines!(ax, idx, diag.unwrapped_gap; linewidth=2)
    scatter!(ax, idx, diag.unwrapped_gap; markersize=6)
    hlines!(ax, [diag.predicted_phase]; linestyle=:dash, linewidth=2)
    save(joinpath(outdir, String(filename)), fig)
end

function main(args=ARGS)
    opts = parse_cli_args(args)

    grid = parse_int_expr(get_opt(opts, "grid", "2^10+1"))
    kappa = parse_real_expr(get_opt(opts, "kappa", "2.0"))
    seed_ig = parse(Int, get_opt(opts, "seed_ig", "20260418"))
    seed_h = parse(Int, get_opt(opts, "seed_h", "20260419"))
    alpha_obs = parse_real_expr(get_opt(opts, "alpha", get_opt(opts, "alpha_obs", "0.35")))
    beta_obs = parse_real_expr(get_opt(opts, "beta", get_opt(opts, "beta_obs", "0.55")))
    gff_boundary = parse_symbol_text(get_opt(opts, "gff_boundary", "neumann"))
    boundary_mode = parse_symbol_text(get_opt(opts, "boundary_mode", "zero_force"))
    halfwidth = parse_real_expr(get_opt(opts, "halfwidth", "1.0"))
    theta_center = parse_real_expr(get_opt(opts, "theta_center", get_opt(opts, "angle", "0.0")))
    nflow = parse_int_expr(get_opt(opts, "nflow", "1"))
    theta_left = haskey(opts, "theta_left") ? parse_real_expr(opts["theta_left"]) : (nflow == 1 ? 0.0 : pi / 2)
    theta_right = haskey(opts, "theta_right") ? parse_real_expr(opts["theta_right"]) : (nflow == 1 ? 0.0 : pi / 2)
    ds_over_h = parse_real_expr(get_opt(opts, "ds_over_h", "0.02"))
    max_steps = parse_int_expr(get_opt(opts, "max_steps", "1.5e6"))
    goal_capture_steps = parse_real_expr(get_opt(opts, "goal_capture_steps", "3"))
    integrator = parse_symbol_text(get_opt(opts, "integrator", "euler"))
    multiscale = parse_bool(get_opt(opts, "multiscale", "false"), false)

    n_arrows = parse(Int, get_opt(opts, "n_arrows", "52"))
    trim_fraction = parse_real_expr(get_opt(opts, "trim_fraction", "0.07"))
    trace_offset_factor = parse_real_expr(get_opt(opts, "trace_offset_factor", "1.35"))
    sample_offset_factor = parse_real_expr(get_opt(opts, "sample_offset_factor", "1.35"))
    arrow_offset_factor = parse_real_expr(get_opt(opts, "arrow_offset_factor", "3.5"))
    cut_radius_cells = parse(Int, get_opt(opts, "cut_radius_cells", "2"))
    harmonic_iterations = parse(Int, get_opt(opts, "harmonic_iterations", "700"))
    harmonic_omega = parse_real_expr(get_opt(opts, "harmonic_omega", "1.55"))
    collar_width_factor = parse_real_expr(get_opt(opts, "collar_width_factor", "3.0"))
    collar_offsets = parse(Int, get_opt(opts, "collar_offsets", "5"))

    direction_stride = parse(Int, get_opt(opts, "direction_stride", "14"))
    direction_arrow_factor = parse_real_expr(get_opt(opts, "direction_arrow_factor", "2.2"))
    render_plots = parse_bool(get_opt(opts, "render_plots", "true"), true)
    output_dir = get_opt(opts, "output", "example_output/imaginary_geometry/mixed_phase_law")

    mkpath(output_dir)

    println("Sampling square chordal IG field...")
    field = sample_chordal_square_ig_field(
        grid,
        kappa;
        halfwidth=halfwidth,
        seed=seed_ig,
        boundary_mode=boundary_mode,
        T=Float64,
    )
    xs, ys = domain_coordinates(field.domain)
    hgrid = min(field.domain.hx, field.domain.hy)
    ds = ds_over_h * hgrid

    println("Tracing eta with the run_sle_fan-style fan setup...")
    eta_sample = sample_eta_from_fan(
        field;
        theta_center=theta_center,
        theta_left=theta_left,
        theta_right=theta_right,
        nflow=nflow,
        ds=ds,
        ds_over_h=ds_over_h,
        max_steps=max_steps,
        goal_capture_steps=goal_capture_steps,
        integrator=integrator,
        multiscale=multiscale,
    )
    seed = eta_sample.seed
    trace = eta_sample.trace
    curve = ComplexF64.(trace.points)
    length(curve) >= 20 || @warn "the traced flowline is short; diagnostics may be noisy." points=length(curve) termination=trace.termination

    println("Sampling independent square GFF...")
    hfield = sample_independent_gff(grid; boundary=gff_boundary, seed=seed_h, normalize_std=true)

    frames = curve_frames(curve; n_arrows=n_arrows, trim_fraction=trim_fraction)
    trace_offset = trace_offset_factor * hgrid
    sample_offset = sample_offset_factor * hgrid
    arrow_offset = arrow_offset_factor * hgrid
    seed_offset = max(8hgrid, 3arrow_offset)

    println("Building left/right slit-side harmonic approximations...")
    bsym = estimate_bsym_trace(Matrix{Float64}(field.values), xs, ys, frames; offset=trace_offset, smooth_radius=2)
    sides = harmonic_side_fields(
        field,
        xs,
        ys,
        curve,
        frames,
        bsym;
        cut_radius_cells=cut_radius_cells,
        seed_offset=seed_offset,
        harmonic_iterations=harmonic_iterations,
        harmonic_omega=harmonic_omega,
        samples_per_cell=2.0,
    )

    harmdiag = compute_harmonic_arrow_diagnostic(
        sides.HL,
        sides.HR,
        hfield,
        xs,
        ys,
        frames;
        alpha_obs=alpha_obs,
        beta_obs=beta_obs,
        lambda=field.constants.lambda,
        sample_offset=sample_offset,
        anchor_offset=arrow_offset,
    )

    collar_width = collar_width_factor * hgrid
    collardiag = compute_collar_arrow_diagnostic(
        Matrix{Float64}(field.values),
        sides.HL,
        sides.HR,
        hfield,
        xs,
        ys,
        frames;
        alpha_obs=alpha_obs,
        beta_obs=beta_obs,
        lambda=field.constants.lambda,
        width=collar_width,
        n_offsets=collar_offsets,
        anchor_offset=arrow_offset,
    )

    dir_pts, dir_vecs = direction_samples(
        Matrix{Float64}(field.values),
        xs,
        ys,
        field.constants.chi;
        angle=theta_center,
        stride=direction_stride,
        arrow_len=direction_arrow_factor * hgrid,
    )

    fan_curves = [ComplexF64.(fan_trace.points) for fan_trace in eta_sample.traces]

    writedlm(joinpath(output_dir, "curve.tsv"), hcat(real.(curve), imag.(curve)), '\t')
    writedlm(joinpath(output_dir, "eta_seed.tsv"), reshape([real(seed), imag(seed)], 1, 2), '\t')
    writedlm(joinpath(output_dir, "direction_field.tsv"), hcat(real.(dir_pts), imag.(dir_pts), real.(dir_vecs), imag.(dir_vecs)), '\t')
    write_arrow_table(joinpath(output_dir, "harmonic_phase_arrows.tsv"), harmdiag)
    write_arrow_table(joinpath(output_dir, "collar_phase_arrows.tsv"), collardiag)
    writedlm(joinpath(output_dir, "bsym_trace.tsv"), hcat(frames.s, real.(frames.z), imag.(frames.z), bsym), '\t')

    open(joinpath(output_dir, "summary.txt"), "w") do io
        println(io, "IG mixed phase-law visualization summary")
        println(io, "grid = ", grid)
        println(io, "kappa = ", kappa)
        println(io, "chi = ", field.constants.chi)
        println(io, "lambda = ", field.constants.lambda)
        println(io, "alpha = ", alpha_obs)
        println(io, "beta = ", beta_obs)
        println(io, "predicted phase gap = 2 alpha lambda = ", harmdiag.predicted_phase)
        println(io, "harmonic empirical weighted phase = ", harmdiag.empirical_phase)
        println(io, "harmonic error = ", harmdiag.empirical_error)
        println(io, "collar empirical weighted phase = ", collardiag.empirical_phase)
        println(io, "collar error = ", collardiag.empirical_error)
        println(io, "theta_center = ", theta_center)
        println(io, "theta_left = ", theta_left)
        println(io, "theta_right = ", theta_right)
        println(io, "nflow = ", nflow)
        println(io, "goal_capture_steps = ", goal_capture_steps)
        println(io, "eta_index = ", eta_sample.eta_index)
        println(io, "eta_angle = ", eta_sample.angles[eta_sample.eta_index])
        println(io, "seed = ", seed)
        println(io, "trace termination = ", trace.termination)
        println(io, "flowline points = ", length(curve))
        println(io, "gff boundary = ", gff_boundary)
        println(io, "boundary mode = ", boundary_mode)
        println(io, "multiscale = ", multiscale)
        println(io, "ds_over_h = ", ds_over_h)
        println(io, "cut cells = ", count(sides.cut))
        println(io, "left component cells = ", count(sides.leftmask))
        println(io, "right component cells = ", count(sides.rightmask))
        println(io, "harmonic relaxation iterations left/right = ", sides.iterations_left, "/", sides.iterations_right)
        println(io, "harmonic residual left/right = ", sides.residual_left, "/", sides.residual_right)
    end

    if render_plots
        println("Saving figures to: ", abspath(output_dir))
        save_field_heatmaps(output_dir, xs, ys, Matrix{Float64}(field.values), hfield, curve)
        save_direction_plot(output_dir, xs, ys, Matrix{Float64}(field.values), curve, fan_curves, dir_pts, dir_vecs)
        save_phase_arrow_plot(output_dir, xs, curve, harmdiag; filename="phase_arrows_harmonic.png", title_prefix="Harmonic-projection proxy")
        save_phase_arrow_plot(output_dir, xs, curve, collardiag; filename="phase_arrows_collar.png", title_prefix="Thin-collar proxy")
        save_phase_profile(output_dir, harmdiag; filename="phase_profile_harmonic.png")
        save_phase_profile(output_dir, collardiag; filename="phase_profile_collar.png")
    end

    println(@sprintf("Predicted gap: %.6f", harmdiag.predicted_phase))
    println(@sprintf("Harmonic empirical gap: %.6f", harmdiag.empirical_phase))
    println(@sprintf("Harmonic error: %.6f", harmdiag.empirical_error))
    println(@sprintf("Collar empirical gap: %.6f", collardiag.empirical_phase))
    println(@sprintf("Collar error: %.6f", collardiag.empirical_error))
    println("Summary written to: ", abspath(joinpath(output_dir, "summary.txt")))

    return (
        field=field,
        hfield=hfield,
        trace=trace,
        curve=curve,
        harmonic=harmdiag,
        collar=collardiag,
        sides=sides,
        output_dir=abspath(output_dir),
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
