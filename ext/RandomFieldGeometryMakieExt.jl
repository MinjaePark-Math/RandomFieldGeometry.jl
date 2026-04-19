module RandomFieldGeometryMakieExt

using Colors
using GLMakie
using GeometryBasics
using RandomFieldGeometry

const _exports = RandomFieldGeometry.Exporters
const _analysis = RandomFieldGeometry.AnalysisTools
const _flowlines = RandomFieldGeometry.Flowlines
const _fields = RandomFieldGeometry.RandomFieldGenerators

_transparent() = RGBAf(0f0, 0f0, 0f0, 0f0)

function _rgba_from_distance(valid_dists, boundary_cap::Float32, dist::Real, alpha::Float32)
    r, g, b = _exports.rainbow_rgb_from_distance(valid_dists, boundary_cap, Float32(dist))
    return RGBAf(r, g, b, alpha)
end

function _rgba_from_weight(weight::Real, max_weight::Real, alpha::Float32)
    r, g, b = _exports.confluence_rgb_from_weight(weight, max_weight)
    scaled_alpha = alpha * (0.18f0 + 0.82f0 * (log1p(Float32(weight)) / log1p(max(Float32(max_weight), 1.0f0))))
    return RGBAf(r, g, b, clamp(scaled_alpha, 0.0f0, 1.0f0))
end

function _distance_colormap(valid_dists, boundary_cap::Float32)
    valid_count = Float32(length(valid_dists))
    colors = RGBAf[]
    for value in range(0.0f0, boundary_cap, length=256)
        pct = clamp(Float32(searchsortedfirst(valid_dists, Float32(value))) / valid_count, 0.0f0, 1.0f0)
        r, g, b = _exports.rainbow_rgb_from_percentile(pct)
        push!(colors, RGBAf(r, g, b, 1.0f0))
    end
    return colors
end

_point3f(point::NTuple{3,<:Real}) = Point3f(Float32(point[1]), Float32(point[2]), Float32(point[3]))
_point2f(point::NTuple{2,<:Real}) = Point2f(Float32(point[1]), Float32(point[2]))

function _decimated_complex_points(points; max_points::Union{Nothing,Integer}=nothing)
    if isnothing(max_points)
        return points
    end
    n = length(points)
    limit = Int(max_points)
    if limit <= 0 || n <= limit
        return points
    elseif limit == 1
        return [points[1]]
    end

    total = 0.0
    prev = points[1]
    @inbounds for k in 2:n
        z = points[k]
        total += abs(z - prev)
        prev = z
    end

    if !(isfinite(total)) || total <= eps(Float64)
        return [points[1], points[end]]
    end

    out = Vector{eltype(points)}()
    sizehint!(out, limit)
    push!(out, points[1])
    spacing = total / (limit - 1)
    next_s = spacing
    accum = 0.0
    prev = points[1]

    @inbounds for k in 2:n
        z = points[k]
        seg = abs(z - prev)
        while seg > 0 && accum + seg >= next_s && length(out) < limit - 1
            t = (next_s - accum) / seg
            push!(out, prev + t * (z - prev))
            next_s += spacing
        end
        accum += seg
        prev = z
    end

    if isempty(out) || abs(out[end] - points[end]) > eps(Float64)
        push!(out, points[end])
    end
    return out
end

function _flowline_rgba(idx::Int, total::Int, angle::Real, alpha::Float32)
    hue = total <= 1 ? Float32(mod(angle / (2pi), 1)) : Float32((idx - 1) / max(total, 1))
    color = RGB(HSV(360f0 * hue, 0.92f0, 1.0f0))
    return RGBAf(Float32(red(color)), Float32(green(color)), Float32(blue(color)), alpha)
end

function _path_rgba_from_distance_and_weight(
    valid_dists,
    boundary_cap::Float32,
    dist::Real,
    weight::Real,
    max_weight::Real,
    exponent_n::Float32,
)
    r, g, b = if Float32(dist) > boundary_cap
        (1.0f0, 1.0f0, 1.0f0)
    else
        _exports.rainbow_rgb_from_distance(valid_dists, boundary_cap, Float32(dist))
    end

    normalized = clamp(Float32(weight) / max(Float32(max_weight), 1.0f0), 0.0f0, 1.0f0)
    alpha = normalized^((exponent_n * exponent_n) / 10.0f0)
    return RGBAf(r, g, b, clamp(alpha, 0.0f0, 1.0f0))
end

function RandomFieldGeometry.interactive_viewer(
    distances::AbstractArray{<:Real,2};
    path_step::Int=8,
)
    M = _exports.validate_distance_field(distances)
    path_step >= 1 || throw(ArgumentError("`path_step` must be positive."))

    boundary_cap = _exports.boundary_cap(distances)
    valid_dists = _exports.valid_boundary_distances(distances, boundary_cap)
    stats = _analysis.geodesic_edge_weights(distances; start_step=path_step, radius=boundary_cap, include_outside_cap=true)

    set_theme!(theme_dark())

    fig = Figure(size=(1640, 980), fontsize=18, backgroundcolor=:black)
    ax = Axis(fig[1, 1], aspect=DataAspect(), backgroundcolor=:black)
    hidedecorations!(ax)
    hidespines!(ax)

    ui_grid = fig[1, 2] = GridLayout()
    rowsize!(fig.layout, 1, Relative(1.0))
    colsize!(fig.layout, 1, Auto(1.0))
    colsize!(fig.layout, 2, Fixed(340))
    rowgap!(ui_grid, 16)

    Label(ui_grid[1, 1], "Minimal Makie Viewer", font=:bold, fontsize=20, halign=:center, tellwidth=false)

    slider_grid = GridLayout(ui_grid[2, 1])
    rowgap!(slider_grid, 12)
    colgap!(slider_grid, 10)

    Label(slider_grid[1, 1], "Radius", halign=:left)
    sl_radius = Slider(slider_grid[1, 2], range=0.0:0.01:boundary_cap, startvalue=boundary_cap, width=180)
    Label(slider_grid[2, 1], "Weight Alpha", halign=:left)
    sl_weight_alpha = Slider(slider_grid[2, 2], range=0.0:0.1:10.0, startvalue=2.0f0, width=180)
    Label(slider_grid[3, 1], "Ball Alpha", halign=:left)
    sl_ball_alpha = Slider(slider_grid[3, 2], range=0.0:0.01:1.0, startvalue=0.45f0, width=180)

    button_grid = GridLayout(ui_grid[3, 1])
    btn_reset = Button(button_grid[1, 1], label="Reset View", width=180)

    toggle_grid = GridLayout(ui_grid[4, 1])
    tgl_paths = Toggle(toggle_grid[1, 1], active=true)
    Label(toggle_grid[1, 2], "Show Paths", halign=:left)
    tgl_clip_paths = Toggle(toggle_grid[2, 1], active=false)
    Label(toggle_grid[2, 2], "Clip Paths to Radius", halign=:left)
    tgl_ball = Toggle(toggle_grid[3, 1], active=false)
    Label(toggle_grid[3, 2], "Show Metric Ball", halign=:left)
    tgl_center = Toggle(toggle_grid[4, 1], active=true)
    Label(toggle_grid[4, 2], "Show Center", halign=:left)

    obs_radius = sl_radius.value
    obs_weight_alpha = @lift(Float32($(sl_weight_alpha.value)))
    obs_ball_alpha = @lift(Float32($(sl_ball_alpha.value)))

    segment_points = Point2f[]
    segment_distances = Float32[]
    segment_weights = Float32[]
    for (segment, dist, weight) in zip(stats.segments, stats.distances, stats.counts)
        push!(segment_points, _point2f(segment[1]), _point2f(segment[2]))
        push!(segment_distances, Float32(dist), Float32(dist))
        push!(segment_weights, Float32(weight), Float32(weight))
    end

    active_path_colors = @lift begin
        radius = Float32($obs_radius)
        exponent_n = $obs_weight_alpha
        clip_paths = $(tgl_clip_paths.active)

        map(zip(segment_distances, segment_weights)) do (dist, weight)
            if clip_paths && dist > radius
                _transparent()
            else
                _path_rgba_from_distance_and_weight(valid_dists, boundary_cap, dist, weight, stats.max_count, exponent_n)
            end
        end
    end

    ball_rgba = @lift begin
        radius = Float32($obs_radius)
        alpha = Float32($obs_ball_alpha)
        rgba = Matrix{RGBAf}(undef, M, M)

        for y in 1:M, x in 1:M
            dist = Float32(distances[x, y])
            rgba[x, y] = isfinite(dist) && dist <= radius ?
                         _rgba_from_distance(valid_dists, boundary_cap, dist, alpha) :
                         _transparent()
        end

        rgba
    end

    lo = -Float32(M - 1) / 2.0f0
    hi = Float32(M - 1) / 2.0f0
    axis_interval = lo .. hi

    plt_center = scatter!(ax, [Point2f(0, 0)], color=:red, markersize=18)
    plt_ball = image!(ax, axis_interval, axis_interval, ball_rgba, interpolate=false)
    plt_paths = linesegments!(ax, segment_points, color=active_path_colors, transparency=true, linewidth=2)

    on(tgl_center.active) do is_active
        plt_center.visible = is_active
    end
    on(tgl_paths.active) do is_active
        plt_paths.visible = is_active
    end
    on(tgl_ball.active) do is_active
        plt_ball.visible = is_active
    end
    plt_ball.visible = false

    limits!(ax, lo, hi, lo, hi)
    on(btn_reset.clicks) do _
        limits!(ax, lo, hi, lo, hi)
    end

    display(fig)
    return fig
end

function RandomFieldGeometry.interactive_viewer(
    distances::AbstractArray{<:Real,3};
    path_step::Int=8,
)
    M = _exports.validate_distance_field(distances)
    path_step >= 1 || throw(ArgumentError("`path_step` must be positive."))

    boundary_cap = _exports.boundary_cap(distances)
    valid_dists = _exports.valid_boundary_distances(distances, boundary_cap)
    stats = _analysis.geodesic_edge_weights(distances; start_step=path_step, radius=boundary_cap, include_outside_cap=true)
    cmap_colors = _distance_colormap(valid_dists, boundary_cap)
    dist_field = Float32.(distances)

    set_theme!(theme_dark())

    fig = Figure(size=(1640, 980), fontsize=18, backgroundcolor=:black)
    ax = LScene(fig[1, 1], show_axis=false, tellwidth=true, tellheight=true, scenekw=(backgroundcolor=:black,))

    ui_grid = fig[1, 2] = GridLayout()
    rowsize!(fig.layout, 1, Relative(1.0))
    colsize!(fig.layout, 1, Auto(1.0))
    colsize!(fig.layout, 2, Fixed(340))
    rowgap!(ui_grid, 16)

    Label(ui_grid[1, 1], "Minimal Makie Viewer", font=:bold, fontsize=20, halign=:left, tellwidth=false)

    slider_grid = GridLayout(ui_grid[2, 1])
    rowgap!(slider_grid, 12)
    colgap!(slider_grid, 10)

    Label(slider_grid[1, 1], "Radius", halign=:left)
    sl_radius = Slider(slider_grid[1, 2], range=0.0:0.01:boundary_cap, startvalue=boundary_cap, width=180)
    Label(slider_grid[2, 1], "Weight Alpha", halign=:left)
    sl_weight_alpha = Slider(slider_grid[2, 2], range=0.0:0.1:10.0, startvalue=2.0f0, width=180)
    Label(slider_grid[3, 1], "Surface Alpha", halign=:left)
    sl_surface_alpha = Slider(slider_grid[3, 2], range=0.1:0.01:1.0, startvalue=0.45f0, width=180)
    Label(slider_grid[4, 1], "Shell Thickness", halign=:left)
    sl_isorange = Slider(slider_grid[4, 2], range=0.0:0.00125:0.05, startvalue=0.05f0, width=180)

    button_grid = GridLayout(ui_grid[3, 1])
    btn_reset = Button(button_grid[1, 1], label="Reset View", width=180)

    toggle_grid = GridLayout(ui_grid[4, 1])
    tgl_paths = Toggle(toggle_grid[1, 1], active=true)
    Label(toggle_grid[1, 2], "Show Paths", halign=:left)
    tgl_clip_paths = Toggle(toggle_grid[2, 1], active=false)
    Label(toggle_grid[2, 2], "Clip Paths to Radius", halign=:left)
    tgl_surface = Toggle(toggle_grid[3, 1], active=false)
    Label(toggle_grid[3, 2], "Show Metric Shell", halign=:left)
    tgl_center = Toggle(toggle_grid[4, 1], active=true)
    Label(toggle_grid[4, 2], "Show Center", halign=:left)

    obs_radius = sl_radius.value
    obs_weight_alpha = @lift(Float32($(sl_weight_alpha.value)))
    obs_surface_alpha = @lift(Float32($(sl_surface_alpha.value)))
    obs_isorange = @lift(Float32($(sl_isorange.value)))

    segment_points = Point3f[]
    segment_distances = Float32[]
    segment_weights = Float32[]
    for (segment, dist, weight) in zip(stats.segments, stats.distances, stats.counts)
        push!(segment_points, _point3f(segment[1]), _point3f(segment[2]))
        push!(segment_distances, Float32(dist), Float32(dist))
        push!(segment_weights, Float32(weight), Float32(weight))
    end

    active_path_colors = @lift begin
        radius = Float32($obs_radius)
        exponent_n = $obs_weight_alpha
        clip_paths = $(tgl_clip_paths.active)

        map(zip(segment_distances, segment_weights)) do (dist, weight)
            if clip_paths && dist > radius
                _transparent()
            else
                _path_rgba_from_distance_and_weight(valid_dists, boundary_cap, dist, weight, stats.max_count, exponent_n)
            end
        end
    end

    axis_interval = (-Float32(M - 1) / 2.0f0) .. (Float32(M - 1) / 2.0f0)
    surface_colormap = @lift [RGBAf(red(color), green(color), blue(color), $obs_surface_alpha) for color in cmap_colors]

    plt_center = scatter!(ax, [Point3f(0, 0, 0)], color=:red, markersize=16)
    plt_paths = linesegments!(ax, segment_points, color=active_path_colors, transparency=true, linewidth=2)
    plt_surface = volume!(
        ax,
        axis_interval,
        axis_interval,
        axis_interval,
        dist_field,
        algorithm=:iso,
        isovalue=obs_radius,
        isorange=obs_isorange,
        colormap=surface_colormap,
        colorrange=(0.0f0, boundary_cap),
        transparency=true,
    )

    on(tgl_center.active) do is_active
        plt_center.visible = is_active
    end
    on(tgl_paths.active) do is_active
        plt_paths.visible = is_active
    end
    on(tgl_surface.active) do is_active
        plt_surface.visible = is_active
    end
    plt_surface.visible = false
    center!(ax.scene)
    on(btn_reset.clicks) do _
        center!(ax.scene)
    end

    display(fig)
    return fig
end

function RandomFieldGeometry.confluence_viewer(
    distances::AbstractArray{<:Real,3};
    path_step::Int=8,
)
    stats = _analysis.geodesic_edge_weights(distances; start_step=path_step)
    set_theme!(theme_dark())

    segment_points = Point3f[]
    segment_weights = Float32[]
    for (segment, weight) in zip(stats.segments, stats.counts)
        push!(segment_points, _point3f(segment[1]), _point3f(segment[2]))
        push!(segment_weights, Float32(weight), Float32(weight))
    end

    fig = Figure(size=(1180, 820), fontsize=16, backgroundcolor=:black)
    ax = LScene(fig[1, 1], show_axis=false)

    ui = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 2, 320)

    Label(ui[1, 1], "Confluence Viewer", font=:bold, fontsize=20)
    Label(ui[2, 1], "Color encodes geodesic edge weight.", tellwidth=false)

    sliders = SliderGrid(
        ui[3, 1],
        (label="Opacity", range=0.05:0.01:1.0, startvalue=0.55),
        (label="Min Weight", range=0.0:0.01:1.0, startvalue=0.0, format=x -> string(round(x * max(stats.max_count, 1), digits=1))),
    )
    sl_alpha, sl_min_weight = sliders.sliders

    toggle_grid = GridLayout(ui[4, 1])
    tgl_center = Toggle(toggle_grid[1, 1], active=true)
    Label(toggle_grid[1, 2], "Show Center", halign=:left)

    weight_colors = @lift begin
        alpha = Float32($(sl_alpha.value))
        min_weight = Float32($(sl_min_weight.value)) * max(Float32(stats.max_count), 1.0f0)

        map(segment_weights) do weight
            weight < min_weight ? _transparent() : _rgba_from_weight(weight, stats.max_count, alpha)
        end
    end

    plt_center = scatter!(ax, [Point3f(0, 0, 0)], color=:red, markersize=18)
    plt_lines = linesegments!(ax, segment_points, color=weight_colors, transparency=true, linewidth=2)

    on(tgl_center.active) do is_active
        plt_center.visible = is_active
    end

    display(fig)
    return fig
end

function RandomFieldGeometry.metric_ball_viewer(
    distances::AbstractArray{<:Real,3};
    point_step::Int=3,
)
    boundary_cap = _exports.boundary_cap(distances)
    valid_dists = _exports.valid_boundary_distances(distances, boundary_cap)
    sample = _analysis.sample_distance_points(distances; step=point_step, radius=boundary_cap)
    pts = Point3f[_point3f(point) for point in sample.positions]

    set_theme!(theme_dark())

    fig = Figure(size=(1200, 840), fontsize=16, backgroundcolor=:black)
    ax = LScene(fig[1, 1], show_axis=false)

    ui = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 2, 320)

    Label(ui[1, 1], "Metric Ball Viewer", font=:bold, fontsize=20)
    sliders = SliderGrid(
        ui[2, 1],
        (label="Radius", range=0.0:0.01:boundary_cap, startvalue=boundary_cap),
        (label="Point Size", range=1.0:0.5:10.0, startvalue=4.0),
        (label="Ball Opacity", range=0.02:0.01:1.0, startvalue=0.22),
        (label="Shell Width", range=0.25:0.25:8.0, startvalue=1.5),
        (label="Shell Opacity", range=0.02:0.01:1.0, startvalue=0.8),
    )

    sl_radius, sl_point_size, sl_ball_alpha, sl_shell_width, sl_shell_alpha = sliders.sliders

    toggle_grid = GridLayout(ui[3, 1])
    tgl_shell = Toggle(toggle_grid[1, 1], active=true)
    Label(toggle_grid[1, 2], "Show Shell Accent", halign=:left)
    tgl_center = Toggle(toggle_grid[2, 1], active=true)
    Label(toggle_grid[2, 2], "Show Center", halign=:left)

    interior_colors = @lift begin
        radius = Float32($(sl_radius.value))
        alpha = Float32($(sl_ball_alpha.value))

        map(sample.distances) do dist
            dist <= radius ? _rgba_from_distance(valid_dists, boundary_cap, dist, alpha) : _transparent()
        end
    end

    shell_colors = @lift begin
        radius = Float32($(sl_radius.value))
        width = Float32($(sl_shell_width.value))
        alpha = Float32($(sl_shell_alpha.value))
        show_shell = $(tgl_shell.active)

        map(sample.distances) do dist
            show_shell && abs(dist - radius) <= width ? _rgba_from_distance(valid_dists, boundary_cap, dist, alpha) : _transparent()
        end
    end

    point_size = @lift Float32($(sl_point_size.value))

    plt_center = scatter!(ax, [Point3f(0, 0, 0)], color=:red, markersize=16)
    plt_ball = scatter!(ax, pts, color=interior_colors, markersize=point_size, transparency=true)
    plt_shell = scatter!(ax, pts, color=shell_colors, markersize=@lift($point_size * 1.2f0), transparency=true)

    on(tgl_center.active) do is_active
        plt_center.visible = is_active
    end

    display(fig)
    return fig
end

function RandomFieldGeometry.sphere_viewer(
    distances::AbstractArray{<:Real,3};
    point_step::Int=3,
    shell_half_width::Real=1.5f0,
)
    boundary_cap = _exports.boundary_cap(distances)
    valid_dists = _exports.valid_boundary_distances(distances, boundary_cap)
    sample = _analysis.sample_distance_points(distances; step=point_step, radius=boundary_cap)
    pts = Point3f[_point3f(point) for point in sample.positions]

    set_theme!(theme_dark())

    fig = Figure(size=(1200, 820), fontsize=16, backgroundcolor=:black)
    ax = LScene(fig[1, 1], show_axis=false)

    ui = fig[1, 2] = GridLayout()
    colsize!(fig.layout, 2, 320)

    Label(ui[1, 1], "Metric Sphere Viewer", font=:bold, fontsize=20)
    sliders = SliderGrid(
        ui[2, 1],
        (label="Radius", range=0.0:0.01:boundary_cap, startvalue=boundary_cap * 0.65),
        (label="Thickness", range=0.25:0.25:8.0, startvalue=Float32(shell_half_width)),
        (label="Point Size", range=1.0:0.5:10.0, startvalue=5.0),
        (label="Opacity", range=0.02:0.01:1.0, startvalue=0.88),
    )
    sl_radius, sl_width, sl_size, sl_alpha = sliders.sliders

    colors = @lift begin
        radius = Float32($(sl_radius.value))
        width = Float32($(sl_width.value))
        alpha = Float32($(sl_alpha.value))
        map(sample.distances) do dist
            abs(dist - radius) <= width ? _rgba_from_distance(valid_dists, boundary_cap, dist, alpha) : _transparent()
        end
    end

    scatter!(ax, pts, color=colors, markersize=@lift(Float32($(sl_size.value))), transparency=true)
    scatter!(ax, [Point3f(0, 0, 0)], color=:red, markersize=16)

    display(fig)
    return fig
end

function RandomFieldGeometry.slice_viewer(
    distances::AbstractArray{<:Real,3};
    radius::Real=_exports.boundary_cap(distances),
)
    boundary_cap = _exports.boundary_cap(distances)
    valid_dists = _exports.valid_boundary_distances(distances, boundary_cap)
    cmap = _distance_colormap(valid_dists, boundary_cap)

    set_theme!(theme_dark())

    fig = Figure(size=(1100, 820), fontsize=16, backgroundcolor=:black)
    ax = Axis(fig[1, 1], title="Central Slice", aspect=DataAspect())
    Colorbar(fig[1, 2], limits=(0.0f0, boundary_cap), colormap=cmap)

    ui = fig[2, 1:2] = GridLayout()
    sliders = SliderGrid(
        ui[1, 1],
        (label="Radius", range=0.0:0.01:boundary_cap, startvalue=Float32(radius)),
    )
    sl_radius = only(sliders.sliders)
    axis_menu = Menu(ui[1, 2], options=["x", "y", "z"], default="z")

    slice_obs = @lift begin
        axis_symbol = Symbol($(axis_menu.selection))
        _analysis.slice_distance_field(distances; axis=axis_symbol, radius=Float32($(sl_radius.value)))
    end

    heatmap!(
        ax,
        slice_obs;
        colormap=cmap,
        colorrange=(0.0f0, boundary_cap),
        nan_color=RGBAf(0f0, 0f0, 0f0, 0f0),
    )

    display(fig)
    return fig
end


function RandomFieldGeometry.plot_flowlines(
    field::RandomFieldGeometry.IGField,
    result::NamedTuple;
    kwargs...,
)
    if haskey(result, :traces)
        return RandomFieldGeometry.plot_flowlines(field, result.traces; kwargs...)
    elseif haskey(result, :path)
        return RandomFieldGeometry.plot_flowlines(field, result.path; kwargs...)
    end

    throw(ArgumentError("the result must have `:traces` or `:path`."))
end

function RandomFieldGeometry.plot_flowlines(
    field::RandomFieldGeometry.IGField,
    trace::_flowlines.FlowlineTrace;
    kwargs...,
)
    return RandomFieldGeometry.plot_flowlines(field, [trace]; kwargs...)
end

function RandomFieldGeometry.plot_flowlines(
    field::RandomFieldGeometry.IGField,
    traces::AbstractVector{<:_flowlines.FlowlineTrace};
    show_field::Bool=true,
    show_seeds::Bool=true,
    field_colormap=:balance,
    field_alpha::Real=0.92,
    line_alpha::Real=0.95,
    linewidth::Real=2.2,
    line_color=nothing,
    figure_size::Tuple{<:Integer,<:Integer}=(900, 900),
    fontsize::Integer=16,
    xlimits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    ylimits::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
    title::Union{Nothing,AbstractString}=nothing,
    max_plot_points::Union{Nothing,Integer}=nothing,
)
    xs, ys = _fields.domain_coordinates(field.domain)
    fig = Figure(size=figure_size, fontsize=fontsize, backgroundcolor=:black)
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        xlabel="Re z",
        ylabel="Im z",
        title=isnothing(title) ? "Imaginary Geometry Flowlines" : String(title),
        backgroundcolor=:black,
        xlabelcolor=:white,
        ylabelcolor=:white,
        xticklabelcolor=:white,
        yticklabelcolor=:white,
        xtickcolor=:white,
        ytickcolor=:white,
        xgridcolor=RGBAf(1f0, 1f0, 1f0, 0.08f0),
        ygridcolor=RGBAf(1f0, 1f0, 1f0, 0.08f0),
        leftspinecolor=:white,
        rightspinecolor=:white,
        topspinecolor=:white,
        bottomspinecolor=:white,
        titlecolor=:white,
    )

    if show_field
        heatmap!(ax, xs, ys, field.values'; colormap=field_colormap, alpha=Float32(field_alpha))
    end

    !isnothing(xlimits) && xlims!(ax, xlimits...)
    !isnothing(ylimits) && ylims!(ax, ylimits...)

    total = max(length(traces), 1)
    same_angle_family = total > 1 && all(trace -> abs(trace.angle - traces[1].angle) <= 1f-5, traces)
    single_color = RGBAf(1.0f0, 0.95f0, 0.18f0, Float32(line_alpha))
    for (idx, trace) in enumerate(traces)
        isempty(trace.points) && continue
        display_points = _decimated_complex_points(trace.points; max_points=max_plot_points)
        pts = Point2f[_point2f((real(z), imag(z))) for z in display_points]
        color = if !isnothing(line_color)
            line_color
        elseif total == 1
            single_color
        elseif same_angle_family
            RGBAf(0.22f0, 1.0f0, 0.78f0, Float32(line_alpha))
        else
            _flowline_rgba(idx, total, trace.angle, Float32(line_alpha))
        end
        lines!(ax, pts, color=color, linewidth=linewidth)

        if show_seeds
            scatter!(ax, [first(pts)], color=color, markersize=9)
        end
    end

    display(fig)
    return fig
end

end
