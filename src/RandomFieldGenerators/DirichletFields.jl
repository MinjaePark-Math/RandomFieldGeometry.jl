#
# Spectral Dirichlet random field generators.
#

"""
    validate_L(L::Int)

Check whether `L` is a sensible size for the Dirichlet sine transform.
"""
function validate_L(L::Int)
    L > 1 || throw(ArgumentError("`L` must be at least 2."))

    if L % 2 != 0
        @warn "L=$L is odd. Discrete Sine Transform (RODFT00) is most efficient when L is even."
    end

    return nothing
end

"""
    validate_free_N(N::Int)

Check whether `N` is a sensible size for the free-boundary cosine transform.
"""
function validate_free_N(N::Int)
    N > 1 || throw(ArgumentError("`N` must be at least 2."))
    return nothing
end

"""
    dirichlet_fgf(dim, L, s, seed; rng=nothing, T=Float32, field_scale=1)
Generates a Fractional Gaussian Field (FGF) with exponent s.
- s = dim/2 : Log-Correlated Gradient Field (LGF)
- s = 1     : Gaussian Free Field (GFF)
"""
function dirichlet_fgf(
    dim::Int,
    L::Int,
    s::Real=1.0,
    seed::Int=1234;
    rng::Union{Nothing,AbstractRNG}=nothing,
    T::Type{<:AbstractFloat}=Float32,
    field_scale::Real=1,
)
    dim >= 1 || throw(ArgumentError("`dim` must be positive."))
    validate_L(L)
    local_rng = isnothing(rng) ? Xoshiro(seed) : rng

    # 1. Initialize Grid (size L-1)
    shape = ntuple(i -> L - 1, dim)
    W = randn(local_rng, T, shape...)

    # 2. Precompute 1D Eigenvalues to avoid redundant sinpi calls
    # λ_1d[i] = sin^2(π * i / 2L)
    λ_1d = Vector{T}(undef, L - 1)
    denom = T(2 * L)
    @inbounds for i in 1:(L - 1)
        λ_1d[i] = sinpi(T(i) / denom)^2
    end

    # 3. Apply Spectral Weights: λ^(-s/2)
    # We use Threads.@threads for multi-core CPU speedup
    # For higher dimensions, a CartesianIndices loop is cleaner
    indices = CartesianIndices(W)
    exponent = -T(s) / T(2)
    scale = T(field_scale)
    @threads for linear_idx in eachindex(W)
        idx = indices[linear_idx]

        # Sum up the eigenvalues for the specific frequency n
        λ_sum = zero(T)
        for d in 1:dim
            λ_sum += λ_1d[idx[d]]
        end

        # Scaling factor: (2 * sum(λ))^(-s/2)
        # Note: idx=1 is the lowest frequency, λ_sum > 0
        weight = (T(2) * λ_sum)^exponent
        W[idx] *= scale * weight
    end

    # 4. Transform from Frequency to Spatial Domain
    # RODFT00 is the Sine Transform (Dirichlet Boundary Conditions)
    FFTW.r2r!(W, FFTW.RODFT00)

    # 5. Normalize
    # Normalization by L^(dim/2) preserves the variance structure
    normalization = T(L)^(T(dim) / T(2))
    W ./= normalization

    return W
end

@inline function _free_1d_eigenvalues(N::Int, ::Type{T}) where {T<:AbstractFloat}
    λ = Vector{T}(undef, N)
    denom = T(2 * N)

    @inbounds for k in 1:N
        λ[k] = T(4) * sinpi(T(k - 1) / denom)^2
    end

    return λ
end

@inline function _free_mode_scales(N::Int, ::Type{T}) where {T<:AbstractFloat}
    scales = fill(inv(sqrt(T(2))), N)
    scales[1] = one(T)
    return scales
end

"""
    free_fgf(dim, N, s, seed; rng=nothing, T=Float32, field_scale=1)

Generate a free-boundary fractional Gaussian field on an `N`-point box in
`dim` dimensions. The sampler uses the spectral decomposition of the graph
Laplacian with free boundary conditions and fixes the additive constant by
subtracting the arithmetic mean.

- `s = dim/2` : free-boundary Log-Correlated Gaussian Field (LGF)
- `s = 1`     : free-boundary Gaussian Free Field (GFF)
"""
function free_fgf(
    dim::Int,
    N::Int,
    s::Real=1.0,
    seed::Int=1234;
    rng::Union{Nothing,AbstractRNG}=nothing,
    T::Type{<:AbstractFloat}=Float32,
    field_scale::Real=1,
)
    dim >= 1 || throw(ArgumentError("`dim` must be positive."))
    validate_free_N(N)
    local_rng = isnothing(rng) ? Xoshiro(seed) : rng

    shape = ntuple(_ -> N, dim)
    W = randn(local_rng, T, shape...)
    λ_1d = _free_1d_eigenvalues(N, T)
    mode_scales = _free_mode_scales(N, T)

    indices = CartesianIndices(W)
    exponent = -T(s) / T(2)
    scale = T(field_scale)
    @threads for linear_idx in eachindex(W)
        idx = indices[linear_idx]
        λ_sum = zero(T)
        coeff_scale = scale
        is_constant_mode = true

        for d in 1:dim
            mode = idx[d]
            λ_sum += λ_1d[mode]
            coeff_scale *= mode_scales[mode]
            is_constant_mode &= (mode == 1)
        end

        if is_constant_mode
            W[idx] = zero(T)
        else
            W[idx] *= coeff_scale * λ_sum^exponent
        end
    end

    # REDFT01 is the cosine-synthesis transform for the free-boundary basis.
    FFTW.r2r!(W, FFTW.REDFT01)

    normalization = T(N)^(T(dim) / T(2))
    W ./= normalization

    W .-= sum(W) / T(length(W))

    return W
end

"""
    dirichlet_lgf(dim, L, seed; kwargs...)
Log-correlated Gaussian Field. s = dim/2.
"""
function dirichlet_lgf(dim::Int, L::Int, seed::Int=1234; kwargs...)
    return dirichlet_fgf(dim, L, dim / 2, seed; kwargs...)
end

"""
    free_lgf(dim, N, seed; kwargs...)

Free-boundary log-correlated Gaussian field. The additive constant is fixed by
subtracting the arithmetic mean.
"""
function free_lgf(dim::Int, N::Int, seed::Int=1234; kwargs...)
    return free_fgf(dim, N, dim / 2, seed; kwargs...)
end

"""
    dirichlet_gff(dim, L, seed; kwargs...)
Gaussian Free Field. s = 1.
"""
function dirichlet_gff(dim::Int, L::Int, seed::Int=1234; kwargs...)
    return dirichlet_fgf(dim, L, 1, seed; kwargs...)
end

"""
    free_gff(dim, N, seed; kwargs...)

Free-boundary Gaussian Free Field. The additive constant is fixed by
subtracting the arithmetic mean.
"""
function free_gff(dim::Int, N::Int, seed::Int=1234; kwargs...)
    return free_fgf(dim, N, 1, seed; kwargs...)
end

"""
    free_square_gff(N, seed; kwargs...)

Convenience wrapper for `free_gff(2, N, seed; kwargs...)`.
"""
function free_square_gff(N::Int, seed::Int=1234; kwargs...)
    return free_gff(2, N, seed; kwargs...)
end
