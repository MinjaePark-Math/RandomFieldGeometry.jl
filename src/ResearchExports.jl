module ResearchExports

using JSON

using ..AnalysisTools
using ..Exporters

export export_confluence_web, export_metric_ball_web, export_sphere_web, export_slice_web

function write_point_cloud(
    filepath::AbstractString,
    points::AbstractVector{<:NTuple{3,<:Real}},
    distances::AbstractVector{<:Real},
    valid_dists::AbstractVector{<:Real},
    cap::Float32,
)
    buffer = Float32[]
    sizehint!(buffer, length(points) * 7)

    for (point, dist) in zip(points, distances)
        r, g, b = Exporters.rainbow_rgb_from_distance(valid_dists, cap, Float32(dist))
        push!(buffer, Float32(point[1]), Float32(point[2]), Float32(point[3]), r, g, b, Float32(dist))
    end

    open(filepath, "w") do io
        write(io, buffer)
    end

    return nothing
end

function write_slice(filepath::AbstractString, slice::AbstractMatrix{<:Real})
    open(filepath, "w") do io
        write(io, vec(Float32.(slice)))
    end

    return nothing
end

function safe_downsampled_distance_field(
    distances::AbstractArray{<:Real,3},
    valid_dists::AbstractVector{<:Real},
    mesh_downscale::Int,
)
    safe_dist = Float32.(distances)
    max_valid = Float32(valid_dists[end])
    for idx in eachindex(safe_dist)
        if !isfinite(safe_dist[idx])
            safe_dist[idx] = max_valid + 1.0f0
        end
    end

    return Exporters.downsample_minpool(safe_dist, mesh_downscale)
end

function write_shell_bundle(
    dir::AbstractString,
    distances::AbstractArray{<:Real,3},
    valid_dists::AbstractVector{<:Real},
    cap::Float32;
    num_frames::Int,
    mesh_downscale::Int,
)
    num_frames >= 1 || throw(ArgumentError("`num_frames` must be positive."))
    mesh_downscale >= 1 || throw(ArgumentError("`mesh_downscale` must be positive."))

    radii = Exporters.sample_frame_radii(valid_dists, num_frames)
    dist_small = safe_downsampled_distance_field(distances, valid_dists, mesh_downscale)

    Exporters.write_quantized_shell_frames(
        joinpath(dir, "shell_frames.bin"),
        dist_small,
        radii,
        valid_dists,
        cap;
        mesh_downscale=mesh_downscale,
        original_grid_size=size(distances, 1),
    )

    return radii
end

function export_metric_ball_web(
    distances::AbstractArray{<:Real,3},
    num_frames::Integer;
    mesh_downscale::Int=3,
    dir::AbstractString=".",
)
    M = Exporters.validate_distance_field(distances)
    mkpath(dir)

    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    radii = Exporters.sample_frame_radii(valid_dists, Int(num_frames))
    dist_small = safe_downsampled_distance_field(distances, valid_dists, mesh_downscale)
    Exporters.write_quantized_shell_frames(
        joinpath(dir, "ball_frames.bin"),
        dist_small,
        radii,
        valid_dists,
        cap;
        mesh_downscale=mesh_downscale,
        original_grid_size=size(distances, 1),
    )

    slice_z = AnalysisTools.slice_distance_field(distances; axis=:z, radius=cap)
    write_slice(joinpath(dir, "slice_z.bin"), slice_z)

    meta = Dict(
        "kind" => "metric_ball",
        "gridSize" => M,
        "boundaryCap" => Float64(cap),
        "meshDownscale" => mesh_downscale,
        "numFrames" => length(radii),
        "radii" => Float64.(radii),
        "sliceIndex" => div(M, 2) + 1,
        "sampleDistances" => Float64.(valid_dists[round.(Int, range(1, length(valid_dists); length=min(256, length(valid_dists))))]),
    )

    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end

    return nothing
end

function export_sphere_web(
    distances::AbstractArray{<:Real,3},
    num_frames::Integer;
    point_step::Int=3,
    mesh_downscale::Int=3,
    shell_half_width::Real=1.5f0,
    dir::AbstractString=".",
)
    M = Exporters.validate_distance_field(distances)
    mkpath(dir)

    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    radii = write_shell_bundle(dir, distances, valid_dists, cap; num_frames=Int(num_frames), mesh_downscale=mesh_downscale)

    points = AnalysisTools.sample_distance_points(distances; step=point_step, radius=cap)
    write_point_cloud(joinpath(dir, "metric_points.bin"), points.positions, points.distances, valid_dists, cap)

    meta = Dict(
        "kind" => "sphere",
        "gridSize" => M,
        "boundaryCap" => Float64(cap),
        "pointStep" => point_step,
        "meshDownscale" => mesh_downscale,
        "numFrames" => length(radii),
        "radii" => Float64.(radii),
        "shellHalfWidth" => Float64(shell_half_width),
    )

    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end

    return nothing
end

function export_confluence_web(
    distances::AbstractArray{<:Real,3};
    path_step::Int=8,
    dir::AbstractString=".",
)
    M = Exporters.validate_distance_field(distances)
    mkpath(dir)

    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    stats = AnalysisTools.geodesic_edge_weights(distances; start_step=path_step, radius=cap, include_outside_cap=true)

    buffer = Float32[]
    sizehint!(buffer, length(stats.segments) * 16)

    for (segment, count, dist) in zip(stats.segments, stats.counts, stats.distances)
        color = Exporters.confluence_rgb_from_weight(count, stats.max_count)
        for endpoint in segment
            push!(
                buffer,
                Float32(endpoint[1]),
                Float32(endpoint[2]),
                Float32(endpoint[3]),
                Float32(color[1]),
                Float32(color[2]),
                Float32(color[3]),
                Float32(dist),
                Float32(count),
            )
        end
    end

    open(joinpath(dir, "confluence_edges.bin"), "w") do io
        write(io, buffer)
    end

    meta = Dict(
        "kind" => "confluence",
        "gridSize" => M,
        "boundaryCap" => Float64(cap),
        "pathStep" => path_step,
        "numPaths" => stats.num_paths,
        "maxWeight" => Int(stats.max_count),
        "numEdges" => length(stats.counts),
        "sampleDistances" => Float64.(valid_dists[round.(Int, range(1, length(valid_dists); length=min(256, length(valid_dists))))]),
    )

    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end

    return nothing
end

function export_slice_web(
    distances::AbstractArray{<:Real,3};
    radius::Real=Exporters.boundary_cap(distances),
    dir::AbstractString=".",
)
    M = Exporters.validate_distance_field(distances)
    mkpath(dir)

    cap = Exporters.boundary_cap(distances)
    valid_dists = Exporters.valid_boundary_distances(distances, cap)
    rad = Float32(radius)

    slice_x = AnalysisTools.slice_distance_field(distances; axis=:x, radius=rad)
    slice_y = AnalysisTools.slice_distance_field(distances; axis=:y, radius=rad)
    slice_z = AnalysisTools.slice_distance_field(distances; axis=:z, radius=rad)

    write_slice(joinpath(dir, "slice_x.bin"), slice_x)
    write_slice(joinpath(dir, "slice_y.bin"), slice_y)
    write_slice(joinpath(dir, "slice_z.bin"), slice_z)

    open(joinpath(dir, "palette.json"), "w") do io
        JSON.print(io, Dict(
            "boundaryCap" => Float64(cap),
            "sampleDistances" => Float64.(valid_dists[round.(Int, range(1, length(valid_dists); length=min(256, length(valid_dists))))]),
        ))
    end

    meta = Dict(
        "kind" => "slice",
        "gridSize" => M,
        "boundaryCap" => Float64(cap),
        "radius" => Float64(rad),
        "sliceIndex" => div(M, 2) + 1,
    )

    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end

    return nothing
end

end
