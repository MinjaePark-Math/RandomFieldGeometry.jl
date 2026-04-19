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

"""
    dirichlet_lgf(dim, L, seed; kwargs...)
Log-correlated Gaussian Field. s = dim/2.
"""
function dirichlet_lgf(dim::Int, L::Int, seed::Int=1234; kwargs...)
    return dirichlet_fgf(dim, L, dim / 2, seed; kwargs...)
end

"""
    dirichlet_gff(dim, L, seed; kwargs...)
Gaussian Free Field. s = 1.
"""
function dirichlet_gff(dim::Int, L::Int, seed::Int=1234; kwargs...)
    return dirichlet_fgf(dim, L, 1, seed; kwargs...)
end
