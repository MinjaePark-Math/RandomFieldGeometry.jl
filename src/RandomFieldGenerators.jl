module RandomFieldGenerators

using FFTW
using Random
using Base.Threads

export dirichlet_fgf, dirichlet_lgf, dirichlet_gff

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
    dirichlet_fgf(dim, L, s, seed)
Generates a Fractional Gaussian Field (FGF) with exponent s.
- s = dim/2 : Log-Correlated Gradient Field (LGF)
- s = 1     : Gaussian Free Field (GFF)
"""
function dirichlet_fgf(dim::Int, L::Int, s::Real=1.0, seed::Int=1234)
    dim >= 1 || throw(ArgumentError("`dim` must be positive."))
    validate_L(L)
    rng = Xoshiro(seed)

    # 1. Initialize Grid (size L-1)
    shape = ntuple(i -> L - 1, dim)
    W = randn(rng, Float32, shape...)

    # 2. Precompute 1D Eigenvalues to avoid redundant sinpi calls
    # λ_1d[i] = sin^2(π * i / 2L)
    λ_1d = Float32[(sinpi(i / (2 * L)))^2 for i in 1:L-1]

    # 3. Apply Spectral Weights: λ^(-s/2)
    # We use Threads.@threads for multi-core CPU speedup
    # For higher dimensions, a CartesianIndices loop is cleaner
    indices = CartesianIndices(W)
    @threads for linear_idx in eachindex(W)
        idx = indices[linear_idx]

        # Sum up the eigenvalues for the specific frequency n
        λ_sum = 0.0f0
        for d in 1:dim
            λ_sum += λ_1d[idx[d]]
        end

        # Scaling factor: (2 * sum(λ))^(-s/2)
        # Note: idx=1 is the lowest frequency, λ_sum > 0
        weight = (2.0f0 * λ_sum)^(-Float32(s) / 2.0f0)
        W[idx] *= weight
    end

    # 4. Transform from Frequency to Spatial Domain
    # RODFT00 is the Sine Transform (Dirichlet Boundary Conditions)
    FFTW.r2r!(W, FFTW.RODFT00)

    # 5. Normalize
    # Normalization by L^(dim/2) preserves the variance structure
    normalization = Float32(L)^(Float32(dim) / 2.0f0)
    W ./= normalization

    return W
end

"""
    dirichlet_lgf(dim, L, seed)
Log-correlated Gaussian Field. s = dim/2.
"""
function dirichlet_lgf(dim::Int, L::Int, seed::Int=1234)
    return dirichlet_fgf(dim, L, Float32(dim) / 2.0f0, seed)
end

"""
    dirichlet_gff(dim, L, seed)
Gaussian Free Field. s = 1.
"""
function dirichlet_gff(dim::Int, L::Int, seed::Int=1234)
    return dirichlet_fgf(dim, L, 1.0f0, seed)
end

end # module
