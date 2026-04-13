module Exporters

using JSON
using Meshing
using WriteVTK

export export_web_binary, export_vtk

function validate_distance_field(distances::AbstractArray{<:Real,N}) where {N}
    N in (2, 3) || throw(ArgumentError("`distances` must be a square 2D or cubic 3D array."))
    all(size(distances, d) == size(distances, 1) for d in 2:N) ||
        throw(ArgumentError("`distances` must have the same side length in every dimension."))

    return size(distances, 1)
end

function boundary_cap(distances::AbstractArray{<:Real,N}) where {N}
    M = validate_distance_field(distances)
    cap = Inf32

    for idx in CartesianIndices(distances)
        if any(coord == 1 || coord == M for coord in Tuple(idx))
            d = Float32(distances[idx])
            if 0.0f0 < d < cap
                cap = d
            end
        end
    end

    isfinite(cap) || throw(ArgumentError("Could not determine a finite boundary cap from `distances`."))
    return cap
end

function valid_boundary_distances(distances::AbstractArray{<:Real,N}, cap::Float32) where {N}
    valid_dists = Float32[]
    sizehint!(valid_dists, length(distances))

    for d in distances
        value = Float32(d)
        if isfinite(value) && 0.0f0 < value <= cap
            push!(valid_dists, value)
        end
    end

    isempty(valid_dists) && throw(ArgumentError("No positive distances at or below the boundary cap were found."))
    sort!(valid_dists)
    return valid_dists
end

function rainbow_rgb_from_percentile(pct::Real)
    h = Float32(clamp(pct, 0.0f0, 1.0f0)) * 280.0f0
    c = 1.0f0
    x = c * (1.0f0 - abs((h / 60.0f0) % 2.0f0 - 1.0f0))

    if h < 60.0f0
        return (c, x, 0.0f0)
    elseif h < 120.0f0
        return (x, c, 0.0f0)
    elseif h < 180.0f0
        return (0.0f0, c, x)
    elseif h < 240.0f0
        return (0.0f0, x, c)
    elseif h < 300.0f0
        return (x, 0.0f0, c)
    end

    return (c, 0.0f0, x)
end

function rainbow_rgb_from_distance(valid_dists::AbstractVector{<:Real}, cap::Float32, dist::Float32)
    if dist <= 0.0f0
        return (0.0f0, 0.0f0, 0.0f0)
    elseif dist > cap
        return (1.0f0, 1.0f0, 1.0f0)
    end

    pct = clamp(Float32(searchsortedfirst(valid_dists, dist)) / Float32(length(valid_dists)), 0.0f0, 1.0f0)
    return rainbow_rgb_from_percentile(pct)
end

function confluence_rgb_from_weight(weight::Real, max_weight::Real)
    max_value = max(Float32(max_weight), 1.0f0)
    intensity = clamp(log1p(Float32(weight)) / log1p(max_value), 0.0f0, 1.0f0)

    if intensity < 0.45f0
        t = intensity / 0.45f0
        return (
            0.08f0 + 0.22f0 * t,
            0.16f0 + 0.58f0 * t,
            0.32f0 + 0.58f0 * t,
        )
    elseif intensity < 0.8f0
        t = (intensity - 0.45f0) / 0.35f0
        return (
            0.30f0 + 0.60f0 * t,
            0.74f0 + 0.16f0 * t,
            0.90f0 - 0.72f0 * t,
        )
    end

    t = (intensity - 0.8f0) / 0.2f0
    return (
        0.90f0 + 0.10f0 * t,
        0.90f0 + 0.08f0 * t,
        0.18f0 + 0.72f0 * t,
    )
end

function sample_frame_radii(valid_dists::AbstractVector{<:Real}, num_frames::Integer)
    num_frames >= 1 || throw(ArgumentError("`num_frames` must be positive."))

    radii = Float32[0.001f0]
    if num_frames > 1
        for i in 1:(num_frames - 1)
            idx = max(1, round(Int, (i / (num_frames - 1)) * length(valid_dists)))
            push!(radii, Float32(valid_dists[idx]))
        end
    end

    return radii
end

function downsample_minpool(distance_field::AbstractArray{<:Real,3}, factor::Int)
    factor >= 1 || throw(ArgumentError("`factor` must be positive."))

    grid_size = size(distance_field, 1)
    new_size = cld(grid_size, factor)
    pooled = fill(Inf32, new_size, new_size, new_size)

    for k in 1:new_size, j in 1:new_size, i in 1:new_size
        x_start = (i - 1) * factor + 1
        x_end = min(x_start + factor - 1, grid_size)
        y_start = (j - 1) * factor + 1
        y_end = min(y_start + factor - 1, grid_size)
        z_start = (k - 1) * factor + 1
        z_end = min(z_start + factor - 1, grid_size)

        min_val = Inf32
        for zz in z_start:z_end, yy in y_start:y_end, xx in x_start:x_end
            value = Float32(distance_field[xx, yy, zz])
            if value < min_val
                min_val = value
            end
        end
        pooled[i, j, k] = min_val
    end

    return pooled
end

function write_quantized_shell_frames(
    filepath::AbstractString,
    dist_small::AbstractArray{<:Real,3},
    radii::AbstractVector{<:Real},
    valid_dists::AbstractVector{<:Real},
    cap::Float32;
    mesh_downscale::Int,
    original_grid_size::Int,
)
    scale_factor = Float32(original_grid_size - 1) * 0.55f0

    open(filepath, "w") do io
        write(io, Int32(length(radii)))

        for radius in radii
            r = Float32(radius)
            verts, faces = isosurface(dist_small, MarchingCubes(iso=r))
            max_coord = isempty(verts) ? 0.0f0 : maximum(max(v[1], v[2], v[3]) for v in verts)
            is_normalized = max_coord <= 1.1f0

            r_col, g_col, b_col = rainbow_rgb_from_distance(valid_dists, cap, r)
            num_v = Int32(length(verts))
            num_f = Int32(length(faces))
            use_uint32 = num_v > 65535 ? Int32(1) : Int32(0)

            write(io, r, num_v, num_f, Float32(r_col), Float32(g_col), Float32(b_col), use_uint32)

            for i in 1:num_v
                vx, vy, vz = verts[i][1], verts[i][2], verts[i][3]

                if is_normalized
                    vx = vx * Float32(original_grid_size - 1) / 2.0f0
                    vy = vy * Float32(original_grid_size - 1) / 2.0f0
                    vz = vz * Float32(original_grid_size - 1) / 2.0f0
                else
                    vx *= Float32(mesh_downscale)
                    vy *= Float32(mesh_downscale)
                    vz *= Float32(mesh_downscale)
                end

                qx = round(Int16, clamp(vx / scale_factor, -1.0f0, 1.0f0) * 32767.0f0)
                qy = round(Int16, clamp(vy / scale_factor, -1.0f0, 1.0f0) * 32767.0f0)
                qz = round(Int16, clamp(vz / scale_factor, -1.0f0, 1.0f0) * 32767.0f0)

                write(io, qx, qy, qz)
            end

            for face in faces
                if use_uint32 == 1
                    write(io, UInt32(face[1] - 1), UInt32(face[2] - 1), UInt32(face[3] - 1))
                else
                    write(io, UInt16(face[1] - 1), UInt16(face[2] - 1), UInt16(face[3] - 1))
                end
            end
        end
    end

    return nothing
end

function trace_lowest_neighbor(distances::AbstractArray{<:Real,N}, curr::NTuple{N,Int}, M::Int) where {N}
    min_d = Float32(distances[curr...])
    best_nb = curr

    for dim in 1:N
        for δ in (-1, 1)
            nb = ntuple(i -> curr[i] + (i == dim ? δ : 0), N)
            if !all(1 <= nb[i] <= M for i in 1:N)
                continue
            end

            nb_dist = Float32(distances[nb...])
            if nb_dist < min_d
                min_d = nb_dist
                best_nb = nb
            end
        end
    end

    return best_nb, min_d
end

function ensure_parent_dir(path::AbstractString)
    parent = dirname(path)
    if parent != "." && !isempty(parent)
        mkpath(parent)
    end

    return nothing
end

function export_vtk(distances, filename="LFPP_Data"; path_step=5)
    path_step >= 1 || throw(ArgumentError("`path_step` must be positive."))

    println("Exporting Masked VTK Files & Colormap...")
    M = validate_distance_field(distances)
    N = ndims(distances)
    cap = boundary_cap(distances)
    valid_dists = valid_boundary_distances(distances, cap)
    ensure_parent_dir(filename)

    println(" -> Boundary Cap: $(round(cap, digits=2))")
    println(" -> Generating ParaView Distribution Colormap...")

    rgb_points = Float64[]
    dist_steps = range(0.0f0, Float64(cap), length=256)
    valid_count = Float32(length(valid_dists))

    for dist_val in dist_steps
        pct = clamp(Float32(searchsortedfirst(valid_dists, Float32(dist_val))) / valid_count, 0.0f0, 1.0f0)
        r, g, b = rainbow_rgb_from_percentile(pct)
        push!(rgb_points, dist_val, Float64(r), Float64(g), Float64(b))
    end

    preset = [Dict(
        "ColorSpace" => "RGB",
        "Name" => "LFPP_Distribution_Rainbow",
        "RGBPoints" => rgb_points,
    )]

    open(filename * "_ColorPreset.json", "w") do io
        JSON.print(io, preset)
    end

    masked_distances = Float32.(distances)
    for i in eachindex(masked_distances)
        if masked_distances[i] > cap
            masked_distances[i] = NaN32
        end
    end

    if N == 2
        vtk_grid(filename * "_Volume", 1:M, 1:M) do vtk
            vtk["DistanceField"] = masked_distances
        end
    else
        vtk_grid(filename * "_Volume", 1:M, 1:M, 1:M) do vtk
            vtk["DistanceField"] = masked_distances
        end
    end
    println(" -> Volume saved to $(filename)_Volume.vti")

    println(" -> Tracing masked grid geodesics and counting edge frequencies...")
    edge_counts = Dict{Any,Int32}()
    edge_dists = Dict{Any,Float32}()

    ranges = ntuple(_ -> 1:path_step:M, N)
    for start_idx in CartesianIndices(ranges)
        curr = Tuple(start_idx.I)
        d_curr = Float32(distances[curr...])
        if !(0.0f0 < d_curr <= cap)
            continue
        end

        while Float32(distances[curr...]) > 0.0f0
            d_start = Float32(distances[curr...])
            best_nb, _ = trace_lowest_neighbor(distances, curr, M)
            if best_nb == curr
                break
            end

            edge = (curr, best_nb)
            edge_counts[edge] = get(edge_counts, edge, Int32(0)) + Int32(1)

            if !haskey(edge_dists, edge)
                edge_dists[edge] = d_start
            end

            curr = best_nb
        end
    end

    pts_buffer = Float32[]
    dist_buffer = Float32[]
    weight_buffer = Int32[]

    for (edge, count) in edge_counts
        node_a, node_b = edge
        if N == 2
            push!(pts_buffer, Float32(node_a[1]), Float32(node_a[2]), 0.0f0)
            push!(pts_buffer, Float32(node_b[1]), Float32(node_b[2]), 0.0f0)
        else
            push!(pts_buffer, Float32(node_a[1]), Float32(node_a[2]), Float32(node_a[3]))
            push!(pts_buffer, Float32(node_b[1]), Float32(node_b[2]), Float32(node_b[3]))
        end
        push!(weight_buffer, count)
        push!(dist_buffer, edge_dists[edge])
    end

    num_points = div(length(pts_buffer), 3)
    coords = reshape(pts_buffer, 3, num_points)
    cells = [MeshCell(VTKCellTypes.VTK_LINE, [2i - 1, 2i]) for i in 1:div(num_points, 2)]

    vtk_grid(filename * "_Paths", coords, cells) do vtk
        vtk["PathWeight", VTKCellData()] = weight_buffer
        vtk["DistanceField", VTKCellData()] = dist_buffer
    end
    println(" -> Unique Edges: $(length(weight_buffer)). Paths saved to $(filename)_Paths.vtu")

    return nothing
end

function export_web_binary(distances, num_frames; path_step=5, mesh_downscale=3, dir=".")
    ndims(distances) == 3 || throw(ArgumentError("`export_web_binary` currently supports only 3D distance fields."))
    path_step >= 1 || throw(ArgumentError("`path_step` must be positive."))
    mesh_downscale >= 1 || throw(ArgumentError("`mesh_downscale` must be positive."))
    num_frames >= 1 || throw(ArgumentError("`num_frames` must be positive."))

    println("3. Initializing Web-Scaled Binary Export...")
    mkpath(dir)

    M = validate_distance_field(distances)
    N = M - 1
    cx = Float32(div(N, 2) + 1)
    cy = cx
    cz = cx

    cap = boundary_cap(distances)
    valid_dists = valid_boundary_distances(distances, cap)

    println("   -> Tracing High-Res Geodesics...")
    path_buffer = Float32[]
    for start_x in 1:path_step:M, start_y in 1:path_step:M, start_z in 1:path_step:M
        curr = (start_x, start_y, start_z)
        if Float32(distances[curr...]) <= 0.0f0
            continue
        end

        while Float32(distances[curr...]) > 0.0f0
            d_start = Float32(distances[curr...])
            next_idx, d_end = trace_lowest_neighbor(distances, curr, M)
            if next_idx == curr
                break
            end

            r, g, b = rainbow_rgb_from_distance(valid_dists, cap, d_start)
            push!(
                path_buffer,
                Float32(curr[1]) - cx,
                Float32(curr[2]) - cy,
                Float32(curr[3]) - cz,
                r,
                g,
                b,
                d_start,
            )

            r, g, b = rainbow_rgb_from_distance(valid_dists, cap, d_end)
            push!(
                path_buffer,
                Float32(next_idx[1]) - cx,
                Float32(next_idx[2]) - cy,
                Float32(next_idx[3]) - cz,
                r,
                g,
                b,
                d_end,
            )

            curr = next_idx
        end
    end

    open(joinpath(dir, "lfpp_paths.bin"), "w") do io
        write(io, path_buffer)
    end

    println("   -> Downsampling Volume by $mesh_downscale x (Min-Pooling)...")
    safe_dist = Float32.(distances)
    max_valid = valid_dists[end]
    for i in eachindex(safe_dist)
        if !isfinite(safe_dist[i])
            safe_dist[i] = max_valid + 1.0f0
        end
    end
    dist_small = downsample_minpool(safe_dist, mesh_downscale)
    radii = sample_frame_radii(valid_dists, num_frames)
    println("   -> Carving & Quantizing $num_frames Solid Isosurfaces...")
    write_quantized_shell_frames(
        joinpath(dir, "lfpp_shells.bin"),
        dist_small,
        radii,
        valid_dists,
        cap;
        mesh_downscale=mesh_downscale,
        original_grid_size=M,
    )

    meta = Dict("N" => N, "boundaryCap" => Float64(cap), "numFrames" => num_frames)
    open(joinpath(dir, "meta.json"), "w") do io
        JSON.print(io, meta)
    end

    println("=== Export Complete ===")
    return nothing
end

end
