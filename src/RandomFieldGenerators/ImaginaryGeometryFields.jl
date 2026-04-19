#
# Square-domain imaginary geometry field samplers built on the shared
# Dirichlet random-field generators in this module.
#

struct IGConstants{T<:AbstractFloat}
    kappa::T
    kappa_dual::T
    chi::T
    lambda::T
    lambda_dual::T
    critical_angle::T
end

struct SquareDomain{T<:AbstractFloat}
    n::Int
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    hx::T
    hy::T
    invhx::T
    invhy::T
end

struct IGField{T<:AbstractFloat}
    values::Matrix{T}
    random::Matrix{T}
    deterministic::Matrix{T}
    domain::SquareDomain{T}
    constants::IGConstants{T}
    boundary::Matrix{T}
    kind::Symbol
    boundary_mode::Symbol
    boundary_shift::T
    alpha::T
    beta::T
    z0::Complex{T}
end

"""
    ig_constants(kappa; T=Float32)

Return the basic imaginary-geometry constants for `κ ∈ (0, 4)`, including
`χ`, `λ`, `λ'`, and the critical angle `θ_c = πκ / (4 - κ)`.
"""
function ig_constants(kappa::Real; T::Type{<:AbstractFloat}=Float32)
    κ = T(kappa)
    zero(T) < κ < T(4) || throw(ArgumentError("imaginary-geometry flow lines require `kappa ∈ (0, 4)`, got $kappa."))

    χ = T(2) / sqrt(κ) - sqrt(κ) / T(2)
    λ = T(pi) / sqrt(κ)
    λdual = T(pi) * sqrt(κ) / T(4)
    κdual = T(16) / κ
    θc = T(pi) * κ / (T(4) - κ)

    return IGConstants(κ, κdual, χ, λ, λdual, θc)
end

ig_critical_angle(kappa::Real; T::Type{<:AbstractFloat}=Float32) = ig_constants(kappa; T=T).critical_angle

"""
    square_domain(n; halfwidth=1, T=Float32)

Construct the square `[-halfwidth, halfwidth]^2` with `n` grid points on each
side.
"""
function square_domain(n::Integer; halfwidth::Real=1, T::Type{<:AbstractFloat}=Float32)
    n = Int(n)
    n >= 3 || throw(ArgumentError("`n` must be at least 3 to include an interior."))

    L = T(halfwidth)
    xmin = -L
    xmax = L
    hx = (xmax - xmin) / T(n - 1)

    return SquareDomain(
        n,
        xmin,
        xmax,
        xmin,
        xmax,
        hx,
        hx,
        inv(hx),
        inv(hx),
    )
end

function domain_coordinates(domain::SquareDomain{T}) where {T}
    xs = collect(range(domain.xmin, domain.xmax; length=domain.n))
    ys = collect(range(domain.ymin, domain.ymax; length=domain.n))
    return xs, ys
end

@inline function _inside_domain(domain::SquareDomain{T}, z::Complex{T}; margin::T=zero(T)) where {T}
    x = real(z)
    y = imag(z)
    return (domain.xmin + margin <= x <= domain.xmax - margin) &&
           (domain.ymin + margin <= y <= domain.ymax - margin)
end

@inline function _gridpoint(domain::SquareDomain{T}, i::Int, j::Int) where {T}
    x = domain.xmin + T(i - 1) * domain.hx
    y = domain.ymin + T(j - 1) * domain.hy
    return complex(x, y)
end

"""
    boundary_seed(domain, side=:south; fraction=0.5, inset_steps=1.5)

Choose a seed a few mesh steps inside one side of the square.
"""
function boundary_seed(
    domain::SquareDomain{T},
    side::Symbol=:south;
    fraction::Real=0.5,
    inset_steps::Real=1.5,
) where {T}
    t = clamp(T(fraction), zero(T), one(T))
    inset = T(inset_steps) * min(domain.hx, domain.hy)
    x = domain.xmin + t * (domain.xmax - domain.xmin)
    y = domain.ymin + t * (domain.ymax - domain.ymin)

    if side === :south
        return complex(x, domain.ymin + inset)
    elseif side === :north
        return complex(x, domain.ymax - inset)
    elseif side === :west
        return complex(domain.xmin + inset, y)
    elseif side === :east
        return complex(domain.xmax - inset, y)
    end

    throw(ArgumentError("`side` must be one of `:south`, `:north`, `:west`, or `:east`."))
end

"""
    square_seed_grid(domain, nside; margin_steps=3)

Return an `nside × nside` grid of interior seeds for tracing families of
flowlines.
"""
function square_seed_grid(domain::SquareDomain{T}, nside::Integer; margin_steps::Integer=3) where {T}
    nside = Int(nside)
    nside >= 1 || throw(ArgumentError("`nside` must be positive."))
    margin_steps >= 1 || throw(ArgumentError("`margin_steps` must be positive."))

    low = T(margin_steps) * domain.hx
    high = (domain.xmax - domain.xmin) - low
    span = domain.xmax - domain.xmin
    high > low || throw(ArgumentError("the requested margin leaves no interior seeds."))

    ts = nside == 1 ? T[one(T) / T(2)] : collect(range(low / span, high / span; length=nside))
    seeds = Vector{Complex{T}}(undef, nside * nside)

    idx = 1
    @inbounds for ty in ts, tx in ts
        x = domain.xmin + tx * span
        y = domain.ymin + ty * span
        seeds[idx] = complex(x, y)
        idx += 1
    end

    return seeds
end

@inline function _dirichlet_eigenvalues(n::Int, h::T) where {T<:AbstractFloat}
    out = Vector{T}(undef, n)
    denom = T(2 * (n + 1))
    h2 = h * h

    @inbounds for k in 1:n
        out[k] = T(4) * sinpi(T(k) / denom)^2 / h2
    end

    return out
end

function _dirichlet_boundary_rhs(boundary::AbstractMatrix{<:Real}, hx::T, hy::T, ::Type{Tdest}) where {T<:AbstractFloat,Tdest<:AbstractFloat}
    nx, ny = size(boundary)
    mx = nx - 2
    my = ny - 2
    rhs = zeros(Tdest, mx, my)
    invhx2 = inv(Tdest(hx) * Tdest(hx))
    invhy2 = inv(Tdest(hy) * Tdest(hy))

    @inbounds for j in 1:my
        rhs[1, j] += Tdest(boundary[1, j + 1]) * invhx2
        rhs[mx, j] += Tdest(boundary[nx, j + 1]) * invhx2
    end

    @inbounds for i in 1:mx
        rhs[i, 1] += Tdest(boundary[i + 1, 1]) * invhy2
        rhs[i, my] += Tdest(boundary[i + 1, ny]) * invhy2
    end

    return rhs
end

function _harmonic_extension(boundary::AbstractMatrix{<:Real}, hx::Real, hy::Real; T::Type{<:AbstractFloat}=Float32)
    nx, ny = size(boundary)
    nx >= 3 && ny >= 3 || throw(ArgumentError("`boundary` must have at least one interior point."))

    mx = nx - 2
    my = ny - 2
    rhs = _dirichlet_boundary_rhs(boundary, T(hx), T(hy), T)
    eigx = _dirichlet_eigenvalues(mx, T(hx))
    eigy = _dirichlet_eigenvalues(my, T(hy))

    FFTW.r2r!(rhs, FFTW.RODFT00)

    @inbounds for j in 1:my, i in 1:mx
        rhs[i, j] /= eigx[i] + eigy[j]
    end

    FFTW.r2r!(rhs, FFTW.RODFT00)
    rhs ./= T(4 * (mx + 1) * (my + 1))

    out = Matrix{T}(undef, nx, ny)
    @inbounds for j in 1:ny, i in 1:nx
        out[i, j] = T(boundary[i, j])
    end
    @views out[2:(nx - 1), 2:(ny - 1)] .= rhs

    return out
end

function _sample_dirichlet_gff_square(
    domain::SquareDomain{T};
    rng::AbstractRNG,
    field_scale::Real=1,
) where {T<:AbstractFloat}
    out = zeros(T, domain.n, domain.n)
    @views out[2:(domain.n - 1), 2:(domain.n - 1)] .= dirichlet_gff(
        2,
        domain.n - 1,
        0;
        rng=rng,
        T=T,
        field_scale=field_scale,
    )

    return out
end

"""
    square_chordal_boundary_data(domain, kappa; boundary_mode=:zero_force, shift=0)

Square boundary data obtained by pulling half-plane boundary conditions back to
the square and then applying the coordinate-change correction from
Imaginary Geometry I,

`h_S = h_H ∘ ψ - χ arg ψ'`,

with `ψ : S → H` taking the south midpoint of the square to `0` and the north
midpoint to `∞`.

`boundary_mode = :zero_boundary` means the original half-plane boundary data is
identically zero on `∂H`.
`boundary_mode = :zero_force` means the original half-plane boundary data is
`-λ` on `(-∞, 0)` and `+λ` on `(0, ∞)`.
"""
@inline function _is_left_arc_x(x::T, tol::T) where {T<:AbstractFloat}
    return x < -tol
end

@inline function _is_right_arc_x(x::T, tol::T) where {T<:AbstractFloat}
    return x > tol
end

function _validate_boundary_mode(boundary_mode::Symbol)
    boundary_mode in (:zero_boundary, :zero_force) ||
        throw(ArgumentError("`boundary_mode` must be `:zero_boundary` or `:zero_force`, got `$boundary_mode`."))
    return boundary_mode
end

function _square_coordinate_change_correction(domain::SquareDomain{T}, χ::T) where {T<:AbstractFloat}
    corr = zeros(T, domain.n, domain.n)
    tol = min(domain.hx, domain.hy) / T(4)

    πχ = T(pi) * χ
    halfπχ = πχ / T(2)

    # bottom and top edges
    @inbounds for i in 2:(domain.n - 1)
        x = domain.xmin + T(i - 1) * domain.hx

        if x < -tol
            corr[i, 1] = zero(T)      # bottom-left arc
            corr[i, domain.n] = -πχ   # top-left arc
        elseif x > tol
            corr[i, 1] = zero(T)      # bottom-right arc
            corr[i, domain.n] =  πχ   # top-right arc
        else
            corr[i, 1] = zero(T)      # south midpoint average
            corr[i, domain.n] = zero(T)  # north midpoint average of ±πχ
        end
    end

    # vertical sides
    @inbounds for j in 2:(domain.n - 1)
        corr[1, j] = -halfπχ          # left side
        corr[domain.n, j] = halfπχ    # right side
    end

    # corners: use adjacent-edge averages
    corr[1, 1] = -πχ / T(4)
    corr[domain.n, 1] =  πχ / T(4)
    corr[1, domain.n] = -T(3) * πχ / T(4)
    corr[domain.n, domain.n] =  T(3) * πχ / T(4)

    return corr
end

function _square_halfplane_pullback_boundary(
    domain::SquareDomain{T},
    λ::T;
    boundary_mode::Symbol,
) where {T<:AbstractFloat}
    values = zeros(T, domain.n, domain.n)
    tol = min(domain.hx, domain.hy) / T(4)
    boundary_mode === :zero_boundary && return values

    @inbounds for i in 1:domain.n
        x = domain.xmin + T(i - 1) * domain.hx
        force = _is_left_arc_x(x, tol) ? -λ : (_is_right_arc_x(x, tol) ? λ : zero(T))
        values[i, 1] = force
        values[i, domain.n] = force
    end

    @inbounds for j in 2:(domain.n - 1)
        values[1, j] = -λ
        values[domain.n, j] = λ
    end

    return values
end

function square_chordal_boundary_data(
    domain::SquareDomain{T},
    kappa::Real;
    boundary_mode::Symbol=:zero_force,
    shift::Real=0,
) where {T}
    consts = ig_constants(kappa; T=T)
    boundary_mode = _validate_boundary_mode(boundary_mode)
    bc = zeros(T, domain.n, domain.n)
    s = T(shift)
    halfplane_values = _square_halfplane_pullback_boundary(domain, consts.lambda; boundary_mode=boundary_mode)
    coordinate_change = _square_coordinate_change_correction(domain, consts.chi)

    @inbounds for j in 1:domain.n, i in 1:domain.n
        if i == 1 || i == domain.n || j == 1 || j == domain.n
            bc[i, j] = halfplane_values[i, j] + coordinate_change[i, j] + s
        end
    end

    return bc
end

function square_chordal_boundary_data(
    n::Integer,
    kappa::Real;
    halfwidth::Real=1,
    boundary_mode::Symbol=:zero_force,
    shift::Real=0,
    T::Type{<:AbstractFloat}=Float32,
)
    domain = square_domain(n; halfwidth=halfwidth, T=T)
    return square_chordal_boundary_data(domain, kappa; boundary_mode=boundary_mode, shift=shift)
end

"""
    add_interior_singularity!(field, domain; alpha=0, beta=0, z0=0, rcut=min(hx,hy)/4)

Add the discrete analogue of `-α arg(z-z0) - β log|z-z0|` on a square grid.
"""
function add_interior_singularity!(
    field::AbstractMatrix{<:Real},
    domain::SquareDomain{T};
    alpha::Real=0,
    beta::Real=0,
    z0::Complex=zero(Complex{T}),
    rcut::Real=min(domain.hx, domain.hy) / 4,
) where {T<:AbstractFloat}
    size(field) == (domain.n, domain.n) || throw(ArgumentError("`field` shape must match the domain."))

    α = T(alpha)
    β = T(beta)
    zcenter = Complex{T}(z0)
    rmin = max(T(rcut), eps(T))

    @inbounds for j in 1:domain.n, i in 1:domain.n
        z = _gridpoint(domain, i, j) - zcenter
        r = max(abs(z), rmin)
        θ = atan(imag(z), real(z))
        field[i, j] += -α * θ - β * log(r)
    end

    return field
end

function _build_ig_field(
    values::Matrix{T},
    random_part::Matrix{T},
    deterministic::Matrix{T},
    domain::SquareDomain{T},
    boundary::Matrix{T},
    consts::IGConstants{T},
    kind::Symbol;
    boundary_mode::Symbol=:zero_boundary,
    boundary_shift::Real=0,
    alpha::Real=0,
    beta::Real=0,
    z0::Complex=zero(Complex{T}),
) where {T<:AbstractFloat}
    return IGField(
        values,
        random_part,
        deterministic,
        domain,
        consts,
        boundary,
        kind,
        boundary_mode,
        T(boundary_shift),
        T(alpha),
        T(beta),
        Complex{T}(z0),
    )
end

"""
    sample_chordal_square_ig_field(n, kappa; ...)

Sample a discrete square imaginary-geometry field with the chordal boundary
data from the south midpoint to the north midpoint.
"""
function sample_chordal_square_ig_field(
    n::Integer,
    kappa::Real;
    halfwidth::Real=1,
    seed::Integer=1234,
    rng::Union{Nothing,AbstractRNG}=nothing,
    boundary_mode::Symbol=:zero_force,
    shift::Real=0,
    field_scale::Real=1,
    T::Type{<:AbstractFloat}=Float32,
    return_parts::Bool=false,
)
    domain = square_domain(n; halfwidth=halfwidth, T=T)
    consts = ig_constants(kappa; T=T)
    boundary = square_chordal_boundary_data(domain, kappa; boundary_mode=boundary_mode, shift=shift)
    deterministic = _harmonic_extension(boundary, domain.hx, domain.hy; T=T)
    local_rng = isnothing(rng) ? Xoshiro(seed) : rng
    random_part = _sample_dirichlet_gff_square(domain; rng=local_rng, field_scale=field_scale)
    values = deterministic .+ random_part

    field = _build_ig_field(
        values,
        random_part,
        deterministic,
        domain,
        boundary,
        consts,
        :chordal_square;
        boundary_mode=boundary_mode,
        boundary_shift=shift,
    )
    return return_parts ? (field=field, deterministic=deterministic, random=random_part) : field
end

"""
    sample_interior_ig_field(n, kappa; alpha=0, beta=0, z0=0, ...)

Sample a discrete square imaginary-geometry field with an interior
`-α arg(z-z0) - β log|z-z0|` singularity.
"""
function sample_interior_ig_field(
    n::Integer,
    kappa::Real;
    halfwidth::Real=1,
    seed::Integer=1234,
    rng::Union{Nothing,AbstractRNG}=nothing,
    boundary_shift::Real=0,
    alpha::Real=0,
    beta::Real=0,
    z0::Complex=0 + 0im,
    field_scale::Real=1,
    T::Type{<:AbstractFloat}=Float32,
    return_parts::Bool=false,
)
    domain = square_domain(n; halfwidth=halfwidth, T=T)
    consts = ig_constants(kappa; T=T)
    boundary = fill(T(boundary_shift), domain.n, domain.n)
    deterministic = fill(T(boundary_shift), domain.n, domain.n)
    local_rng = isnothing(rng) ? Xoshiro(seed) : rng
    random_part = _sample_dirichlet_gff_square(domain; rng=local_rng, field_scale=field_scale)
    add_interior_singularity!(deterministic, domain; alpha=alpha, beta=beta, z0=z0)
    values = deterministic .+ random_part

    field = _build_ig_field(
        values,
        random_part,
        deterministic,
        domain,
        boundary,
        consts,
        :interior;
        boundary_mode=:zero_boundary,
        boundary_shift=boundary_shift,
        alpha=alpha,
        beta=beta,
        z0=z0,
    )
    return return_parts ? (field=field, deterministic=deterministic, random=random_part) : field
end
