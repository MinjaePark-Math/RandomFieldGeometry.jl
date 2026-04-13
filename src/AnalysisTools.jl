module AnalysisTools

using LinearAlgebra

using ..Exporters
using ..Geodesics

export estimate_ball_growth_dimension,
       estimate_geodesic_dimension,
       estimate_shell_growth_exponent,
       geodesic_edge_weights,
       geodesic_overlay_segments,
       metric_ball_mask,
       metric_shell_mask,
       sample_distance_points,
       slice_distance_field

center_index(distances::AbstractArray{<:Real,N}) where {N} =
    ntuple(_ -> div(Exporters.validate_distance_field(distances), 2) + 1, N)

function centered_position(idx::NTuple{N,Int}, center::NTuple{N,Int}) where {N}
    return ntuple(i -> Float32(idx[i] - center[i]), N)
end

function canonical_edge_key(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N}
    lo, hi = a <= b ? (a, b) : (b, a)
    return (
        ntuple(i -> Int32(lo[i]), N),
        ntuple(i -> Int32(hi[i]), N),
    )
end

function edge_key_to_points(edge::Tuple{NTuple{N,Int32},NTuple{N,Int32}}) where {N}
    a = ntuple(i -> Int(edge[1][i]), N)
    b = ntuple(i -> Int(edge[2][i]), N)
    return a, b
end

function metric_ball_mask(distances::AbstractArray{<:Real,N}, radius::Real) where {N}
    Exporters.validate_distance_field(distances)
    rad = Float32(radius)
    rad >= 0.0f0 || throw(ArgumentError("`radius` must be non-negative."))

    mask = falses(size(distances))
    @inbounds for idx in eachindex(distances)
        value = Float32(distances[idx])
        mask[idx] = isfinite(value) && value <= rad
    end

    return mask
end

function metric_shell_mask(
    distances::AbstractArray{<:Real,N},
    radius::Real;
    half_width::Real=1.0f0,
) where {N}
    Exporters.validate_distance_field(distances)
    rad = Float32(radius)
    width = Float32(half_width)
    rad >= 0.0f0 || throw(ArgumentError("`radius` must be non-negative."))
    width > 0.0f0 || throw(ArgumentError("`half_width` must be positive."))

    lo = max(0.0f0, rad - width)
    hi = rad + width

    mask = falses(size(distances))
    @inbounds for idx in eachindex(distances)
        value = Float32(distances[idx])
        mask[idx] = isfinite(value) && lo <= value <= hi
    end

    return mask
end

function sample_distance_points(
    distances::AbstractArray{<:Real,N};
    step::Int=2,
    radius::Real=Exporters.boundary_cap(distances),
    min_radius::Real=0.0f0,
    shell_half_width::Union{Nothing,Real}=nothing,
) where {N}
    M = Exporters.validate_distance_field(distances)
    step >= 1 || throw(ArgumentError("`step` must be positive."))

    rad = Float32(radius)
    lo = Float32(min_radius)
    center = center_index(distances)

    positions = Vector{NTuple{N,Float32}}()
    point_distances = Float32[]

    shell_width = isnothing(shell_half_width) ? nothing : Float32(shell_half_width)
    ranges = ntuple(_ -> 1:step:M, N)

    for idx in CartesianIndices(ranges)
        point = Tuple(idx.I)
        value = Float32(distances[point...])
        if !isfinite(value) || value < lo || value > rad
            continue
        end
        if shell_width !== nothing && abs(value - rad) > shell_width
            continue
        end

        push!(positions, centered_position(point, center))
        push!(point_distances, value)
    end

    return (positions=positions, distances=point_distances, center=center, radius=rad)
end

function geodesic_edge_weights(
    distances::AbstractArray{<:Real,N};
    start_step::Int=8,
    radius::Real=Exporters.boundary_cap(distances),
    include_outside_cap::Bool=false,
) where {N}
    M = Exporters.validate_distance_field(distances)
    start_step >= 1 || throw(ArgumentError("`start_step` must be positive."))

    rad = Float32(radius)
    center = center_index(distances)

    edge_weights = Dict{Tuple{NTuple{N,Int32},NTuple{N,Int32}},Int32}()
    edge_distances = Dict{Tuple{NTuple{N,Int32},NTuple{N,Int32}},Float32}()
    node_weights = Dict{NTuple{N,Int32},Int32}()
    num_paths = 0

    ranges = ntuple(_ -> 1:start_step:M, N)
    for idx in CartesianIndices(ranges)
        curr = Tuple(idx.I)
        d_curr = Float32(distances[curr...])
        if !isfinite(d_curr) || d_curr <= 0.0f0
            continue
        end
        if !include_outside_cap && d_curr > rad
            continue
        end

        num_paths += 1

        while Float32(distances[curr...]) > 0.0f0
            node_key = ntuple(i -> Int32(curr[i]), N)
            node_weights[node_key] = get(node_weights, node_key, Int32(0)) + Int32(1)

            next_idx, _ = Exporters.trace_lowest_neighbor(distances, curr, M)
            if next_idx == curr
                break
            end

            edge_key = canonical_edge_key(curr, next_idx)
            edge_weights[edge_key] = get(edge_weights, edge_key, Int32(0)) + Int32(1)
            edge_distances[edge_key] = min(get(edge_distances, edge_key, Float32(Inf)), Float32(distances[curr...]))
            curr = next_idx
        end
    end

    segments = Vector{NTuple{2,NTuple{N,Float32}}}()
    counts = Int32[]
    edge_dists = Float32[]

    for (edge, count) in edge_weights
        a_idx, b_idx = edge_key_to_points(edge)
        push!(segments, (centered_position(a_idx, center), centered_position(b_idx, center)))
        push!(counts, count)
        push!(edge_dists, edge_distances[edge])
    end

    sort_order = sortperm(counts; rev=true)

    return (
        segments=segments[sort_order],
        counts=counts[sort_order],
        distances=edge_dists[sort_order],
        node_weights=node_weights,
        num_paths=num_paths,
        max_count=isempty(counts) ? Int32(0) : maximum(counts),
        center=center,
        radius=rad,
    )
end

function geodesic_overlay_segments(
    distances::AbstractArray{<:Real,N};
    start_step::Int=8,
    radius::Real=Exporters.boundary_cap(distances),
    include_outside_cap::Bool=false,
) where {N}
    M = Exporters.validate_distance_field(distances)
    start_step >= 1 || throw(ArgumentError("`start_step` must be positive."))

    rad = Float32(radius)
    center = center_index(distances)

    segments = Vector{NTuple{2,NTuple{N,Float32}}}()
    segment_distances = Float32[]
    num_paths = 0

    ranges = ntuple(_ -> 1:start_step:M, N)
    for idx in CartesianIndices(ranges)
        curr = Tuple(idx.I)
        d_curr = Float32(distances[curr...])
        if !isfinite(d_curr) || d_curr <= 0.0f0
            continue
        end
        if !include_outside_cap && d_curr > rad
            continue
        end

        num_paths += 1

        while Float32(distances[curr...]) > 0.0f0
            next_idx, _ = Exporters.trace_lowest_neighbor(distances, curr, M)
            if next_idx == curr
                break
            end

            push!(segments, (centered_position(curr, center), centered_position(next_idx, center)))
            push!(segment_distances, Float32(distances[curr...]))
            curr = next_idx
        end
    end

    return (
        segments=segments,
        distances=segment_distances,
        num_paths=num_paths,
        center=center,
        radius=rad,
    )
end

function slice_distance_field(
    distances::AbstractArray{<:Real,3};
    axis::Symbol=:x,
    index::Union{Nothing,Integer}=nothing,
    radius::Real=Exporters.boundary_cap(distances),
    fill_value::Real=NaN32,
)
    M = Exporters.validate_distance_field(distances)
    slice_index = something(index, div(M, 2) + 1)
    1 <= slice_index <= M || throw(ArgumentError("`index` must lie inside the grid."))

    rad = Float32(radius)
    fill32 = Float32(fill_value)
    slice = Matrix{Float32}(undef, M, M)

    if axis === :x
        for z in 1:M, y in 1:M
            value = Float32(distances[slice_index, y, z])
            slice[y, z] = isfinite(value) && value <= rad ? value : fill32
        end
    elseif axis === :y
        for z in 1:M, x in 1:M
            value = Float32(distances[x, slice_index, z])
            slice[x, z] = isfinite(value) && value <= rad ? value : fill32
        end
    elseif axis === :z
        for y in 1:M, x in 1:M
            value = Float32(distances[x, y, slice_index])
            slice[x, y] = isfinite(value) && value <= rad ? value : fill32
        end
    else
        throw(ArgumentError("`axis` must be one of `:x`, `:y`, or `:z`."))
    end

    return slice
end

function mean_float(values::AbstractVector{<:Real})
    isempty(values) && return NaN
    return sum(Float64(v) for v in values) / length(values)
end

function linear_fit(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real})
    length(xs) == length(ys) || throw(ArgumentError("`xs` and `ys` must have the same length."))
    length(xs) >= 2 || throw(ArgumentError("At least two points are required for a fit."))

    lx = log.(Float64.(xs))
    ly = log.(Float64.(ys))

    mean_x = mean_float(lx)
    mean_y = mean_float(ly)
    sxx = sum((x - mean_x)^2 for x in lx)
    sxx > 0.0 || throw(ArgumentError("Cannot fit a vertical log-log dataset."))

    slope = sum((x - mean_x) * (y - mean_y) for (x, y) in zip(lx, ly)) / sxx
    intercept = mean_y - slope * mean_x

    yhat = similar(ly)
    for i in eachindex(ly)
        yhat[i] = intercept + slope * lx[i]
    end

    sst = sum((y - mean_y)^2 for y in ly)
    ssr = sum((y - ŷ)^2 for (y, ŷ) in zip(ly, yhat))
    r2 = sst == 0.0 ? 1.0 : 1.0 - ssr / sst

    return (slope=slope, intercept=intercept, r2=r2)
end

function log_spaced_edges(min_value::Float64, max_value::Float64, num_bins::Int)
    num_bins >= 2 || throw(ArgumentError("`num_bins` must be at least 2."))
    min_value > 0.0 || throw(ArgumentError("`min_value` must be positive."))
    max_value > min_value || throw(ArgumentError("`max_value` must exceed `min_value`."))

    log_min = log(min_value)
    log_max = log(max_value)
    return exp.(range(log_min, log_max; length=num_bins + 1))
end

function estimate_geodesic_dimension(
    distances::AbstractArray{<:Real,N};
    step::Int=8,
    num_bins::Int=8,
) where {N}
    M = Exporters.validate_distance_field(distances)
    step >= 1 || throw(ArgumentError("`step` must be positive."))
    num_bins >= 3 || throw(ArgumentError("`num_bins` must be at least 3."))

    center = CartesianIndex(center_index(distances))
    paths = Geodesics.trace_all_geodesics(distances, step)

    euclidean_radii = Float64[]
    path_lengths = Float64[]
    occupied = Set{NTuple{N,Int}}()

    for path in paths
        isempty(path) && continue

        start_idx = first(path)
        displacement = Tuple(start_idx - center)
        euclidean_radius = norm(Float64.(displacement))
        euclidean_radius > 0.0 || continue

        push!(euclidean_radii, euclidean_radius)
        push!(path_lengths, length(path) - 1)

        for idx in path
            push!(occupied, Tuple(idx.I))
        end
    end

    isempty(euclidean_radii) && throw(ArgumentError("No positive-radius geodesics were available for dimension estimation."))

    min_radius = minimum(euclidean_radii)
    max_radius = maximum(euclidean_radii)
    bin_edges = log_spaced_edges(min_radius, max_radius, num_bins)

    binned_radii = Float64[]
    mean_lengths = Float64[]
    counts = Int[]

    for bin_idx in 1:num_bins
        lo = bin_edges[bin_idx]
        hi = bin_edges[bin_idx + 1]
        bucket = Float64[]

        for (radius, length_value) in zip(euclidean_radii, path_lengths)
            in_bin = bin_idx == num_bins ? (lo <= radius <= hi) : (lo <= radius < hi)
            in_bin || continue
            push!(bucket, length_value)
        end

        if length(bucket) >= 2
            push!(binned_radii, sqrt(lo * hi))
            push!(mean_lengths, mean_float(bucket))
            push!(counts, length(bucket))
        end
    end

    length(binned_radii) >= 2 || throw(ArgumentError("Not enough populated radius bins were found for a path-scaling fit."))
    path_fit = linear_fit(binned_radii, mean_lengths)

    box_sizes = Int[]
    box_counts = Int[]
    box_size = 1
    while box_size <= max(2, cld(M, 4))
        boxes = Set{NTuple{N,Int}}()
        for idx in occupied
            push!(boxes, ntuple(i -> div(idx[i] - 1, box_size), N))
        end
        push!(box_sizes, box_size)
        push!(box_counts, length(boxes))
        box_size *= 2
    end

    length(box_sizes) >= 2 || throw(ArgumentError("Not enough box scales were available for box-counting."))
    box_fit = linear_fit(box_sizes, box_counts)

    return (
        num_paths=length(paths),
        path_length_scaling=(
            dimension=path_fit.slope,
            r2=path_fit.r2,
            radii=binned_radii,
            mean_lengths=mean_lengths,
            counts=counts,
        ),
        union_box_counting=(
            dimension=-box_fit.slope,
            r2=box_fit.r2,
            box_sizes=box_sizes,
            box_counts=box_counts,
        ),
    )
end

function sample_profile_radii(valid_dists::AbstractVector{<:Real}, num_radii::Int)
    num_radii >= 4 || throw(ArgumentError("`num_radii` must be at least 4."))

    count_valid = length(valid_dists)
    count_valid >= 4 || throw(ArgumentError("At least four valid radii are required for a profile."))

    indices = unique(round.(Int, range(max(2, floor(Int, 0.08 * count_valid)), max(3, ceil(Int, 0.92 * count_valid)); length=num_radii)))
    radii = Float64[]
    for idx in indices
        push!(radii, Float64(valid_dists[clamp(idx, 1, count_valid)]))
    end
    return unique(radii)
end

function estimate_ball_growth_dimension(
    distances::AbstractArray{<:Real,N};
    num_radii::Int=24,
) where {N}
    Exporters.validate_distance_field(distances)
    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    radii = sample_profile_radii(valid_dists, num_radii)

    volumes = Int[]
    for radius in radii
        count_inside = 0
        @inbounds for value in distances
            distance = Float32(value)
            count_inside += isfinite(distance) && distance <= radius
        end
        push!(volumes, count_inside)
    end

    fit = linear_fit(radii, volumes)
    return (dimension=fit.slope, r2=fit.r2, radii=radii, volumes=volumes)
end

function estimate_shell_growth_exponent(
    distances::AbstractArray{<:Real,N};
    half_width::Real=2.0f0,
    num_radii::Int=24,
) where {N}
    Exporters.validate_distance_field(distances)
    width = Float32(half_width)
    width > 0.0f0 || throw(ArgumentError("`half_width` must be positive."))

    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    radii = sample_profile_radii(valid_dists, num_radii)

    shell_counts = Int[]
    for radius in radii
        lo = max(0.0f0, Float32(radius) - width)
        hi = Float32(radius) + width
        count_shell = 0
        @inbounds for value in distances
            distance = Float32(value)
            count_shell += isfinite(distance) && lo <= distance <= hi
        end
        push!(shell_counts, count_shell)
    end

    fit = linear_fit(radii, shell_counts)
    return (exponent=fit.slope, r2=fit.r2, radii=radii, shell_counts=shell_counts, half_width=width)
end

end
