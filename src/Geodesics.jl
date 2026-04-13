module Geodesics

using LinearAlgebra

export trace_path, trace_all_geodesics

"""
    trace_path(distances, start_idx)
Traces a single path from a starting point back to the source (0.0) 
using steepest descent on the distance field. Works for any dimension N.
"""
function trace_path(distances::AbstractArray{T,N}, start_idx::CartesianIndex{N}) where {T,N}
    path = [start_idx]
    curr = start_idx

    # Safety limit to prevent infinite loops in flat regions
    max_steps = maximum(size(distances)) * 3

    # Dynamically generate the standard basis vectors for N dimensions
    # e.g., in 2D: (CartesianIndex(1,0), CartesianIndex(0,1))
    dirs = ntuple(d -> CartesianIndex(ntuple(i -> i == d ? 1 : 0, N)), N)

    for _ in 1:max_steps
        d_curr = distances[curr]
        if d_curr <= 0.001f0
            break
        end # Reached source

        best_nb = curr
        min_d = d_curr

        # 2N-way connectivity check (4-way for 2D, 6-way for 3D)
        for dir in dirs
            # Check positive direction
            nb_pos = curr + dir
            if checkbounds(Bool, distances, nb_pos) && distances[nb_pos] < min_d
                min_d = distances[nb_pos]
                best_nb = nb_pos
            end

            # Check negative direction
            nb_neg = curr - dir
            if checkbounds(Bool, distances, nb_neg) && distances[nb_neg] < min_d
                min_d = distances[nb_neg]
                best_nb = nb_neg
            end
        end

        if best_nb == curr
            break
        end # Trapped in local minima

        push!(path, best_nb)
        curr = best_nb
    end
    return path
end

"""
    trace_all_geodesics(distances, step)
Generates a bundle of paths for visualization or statistical analysis.
Works natively in 2D, 3D, etc.
"""
function trace_all_geodesics(distances::AbstractArray{T,N}, step_size::Int) where {T,N}
    step_size >= 1 || throw(ArgumentError("`step_size` must be positive."))
    all_paths = Vector{Vector{CartesianIndex{N}}}()

    # Create stepped ranges for every dimension (e.g., 1:step:M, 1:step:M)
    ranges = ntuple(d -> 1:step_size:size(distances, d), N)

    # CartesianIndices automatically handles the N-dimensional nested loop
    for idx in CartesianIndices(ranges)
        if distances[idx] < Inf32 && distances[idx] > 0
            push!(all_paths, trace_path(distances, idx))
        end
    end

    return all_paths
end

end # module
