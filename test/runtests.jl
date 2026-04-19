using Test
using KernelAbstractions
using Random
using RandomFieldGeometry

@testset "RandomFieldGeometry.jl" begin
    @test size(dirichlet_gff(2, 8, 123)) == (7, 7)
    @test eltype(dirichlet_gff(2, 8, 123; T=Float64)) === Float64
    @test dirichlet_gff(2, 8, 0; rng=Xoshiro(123), T=Float32) ≈ dirichlet_gff(2, 8, 0; rng=Xoshiro(123), T=Float32)
    @test size(dirichlet_lgf(3, 6, 123)) == (5, 5, 5)

    weights2d = fill(1.0f0, 5, 5)
    distances2d = solve_fpp(weights2d; backend=KernelAbstractions.CPU())
    result2d = solve_fpp(weights2d; backend=KernelAbstractions.CPU(), return_info=true, sweep_factor=4)

    @test distances2d[3, 3] == 0.0f0
    @test distances2d[4, 3] ≈ 1.0f0
    @test distances2d[5, 3] ≈ 2.0f0
    @test distances2d[5, 5] ≈ 4.0f0
    @test result2d.info.converged
    @test result2d.info.max_iters == 20
    @test result2d.info.dimension == 2
    @test result2d.distances[5, 5] ≈ 4.0f0

    path2d = trace_path(distances2d, CartesianIndex(5, 5))
    @test first(path2d) == CartesianIndex(5, 5)
    @test last(path2d) == CartesianIndex(3, 3)

    ball_mask2d = metric_ball_mask(distances2d, 2.0f0)
    shell_mask2d = metric_shell_mask(distances2d, 2.0f0; half_width=0.5f0)
    sampled2d = sample_distance_points(distances2d; step=2, radius=2.0f0)
    confluence2d = geodesic_edge_weights(distances2d; start_step=2, radius=2.0f0)
    geodesic_dim2d = estimate_geodesic_dimension(distances2d; step=2, num_bins=4)
    ball_dim2d = estimate_ball_growth_dimension(distances2d; num_radii=8)
    shell_dim2d = estimate_shell_growth_exponent(distances2d; half_width=1.0f0, num_radii=8)

    @test ball_mask2d[3, 3]
    @test shell_mask2d[5, 3]
    @test !isempty(sampled2d.positions)
    @test confluence2d.max_count >= 1
    @test geodesic_dim2d.path_length_scaling.dimension > 0
    @test geodesic_dim2d.union_box_counting.dimension > 0
    @test ball_dim2d.dimension > 0
    @test shell_dim2d.exponent > 0

    weights3d = fill(1.0f0, 5, 5, 5)
    distances3d = solve_fpp(weights3d; backend=KernelAbstractions.CPU())
    result3d = solve_fpp(weights3d; backend=KernelAbstractions.CPU(), return_info=true, sweep_factor=4)

    @test distances3d[3, 3, 3] == 0.0f0
    @test distances3d[4, 3, 3] ≈ 1.0f0
    @test distances3d[5, 3, 3] ≈ 2.0f0
    @test distances3d[5, 5, 5] ≈ 6.0f0
    @test result3d.info.converged
    @test result3d.info.max_iters == 20
    @test result3d.info.dimension == 3
    @test result3d.distances[5, 5, 5] ≈ 6.0f0

    path3d = trace_path(distances3d, CartesianIndex(5, 5, 5))
    @test first(path3d) == CartesianIndex(5, 5, 5)
    @test last(path3d) == CartesianIndex(3, 3, 3)

    ball_mask3d = metric_ball_mask(distances3d, 2.0f0)
    shell_mask3d = metric_shell_mask(distances3d, 2.0f0; half_width=0.5f0)
    sampled3d = sample_distance_points(distances3d; step=2, radius=2.0f0)
    confluence3d = geodesic_edge_weights(distances3d; start_step=2, radius=2.0f0)
    geodesic_dim3d = estimate_geodesic_dimension(distances3d; step=2, num_bins=4)
    ball_dim3d = estimate_ball_growth_dimension(distances3d; num_radii=8)
    shell_dim3d = estimate_shell_growth_exponent(distances3d; half_width=1.0f0, num_radii=8)

    @test ball_mask3d[3, 3, 3]
    @test shell_mask3d[5, 3, 3]
    @test !isempty(sampled3d.positions)
    @test confluence3d.max_count >= 1
    @test geodesic_dim3d.path_length_scaling.dimension > 0
    @test geodesic_dim3d.union_box_counting.dimension > 0
    @test ball_dim3d.dimension > 0
    @test shell_dim3d.exponent > 0

    sim2d = run_lfpp_simulation(16, 1.0f0; dim=2, backend=KernelAbstractions.CPU(), sweep_factor=4)
    sim3d = run_lfpp_simulation(10, 1.0f0; dim=3, backend=KernelAbstractions.CPU(), sweep_factor=4)

    @test size(sim2d.distances) == (15, 15)
    @test size(sim2d.weights) == (15, 15)
    @test size(sim3d.distances) == (9, 9, 9)
    @test size(sim3d.weights) == (9, 9, 9)

    ig = sample_chordal_square_ig_field(33, 2.0f0; seed=321)
    @test size(ig.values) == (33, 33)
    @test size(ig.random) == size(ig.values)
    @test size(ig.deterministic) == size(ig.values)
    @test ig.random .+ ig.deterministic ≈ ig.values
    @test ig.constants.lambda ≈ Float32(pi / sqrt(2))
    @test ig.constants.critical_angle ≈ Float32(pi)
    @test ig.boundary_mode === :zero_force

    zero_boundary = square_chordal_boundary_data(33, 2.0f0; boundary_mode=:zero_boundary)
    zero_force = square_chordal_boundary_data(33, 2.0f0; boundary_mode=:zero_force)
    mid = cld(size(zero_boundary, 1), 2)
    @test zero_boundary[mid, 1] ≈ 0.0f0
    @test zero_force[mid, 1] ≈ zero_boundary[mid, 1]
    @test zero_boundary[1, mid] ≈ -ig.constants.chi * Float32(pi / 2)
    @test zero_boundary[end, mid] ≈ ig.constants.chi * Float32(pi / 2)
    @test zero_force[1, mid] - zero_boundary[1, mid] ≈ -ig.constants.lambda
    @test zero_force[end, mid] - zero_boundary[end, mid] ≈ ig.constants.lambda

    ig_zero_boundary = sample_chordal_square_ig_field(33, 2.0f0; boundary_mode=:zero_boundary, seed=321)
    @test ig_zero_boundary.boundary_mode === :zero_boundary
    @test ig_zero_boundary.random .+ ig_zero_boundary.deterministic ≈ ig_zero_boundary.values

    ig_seed = boundary_seed(ig.domain, :south; fraction=0.5, inset_steps=1.5)
    ig_flow = flowline_field(ig)
    ig_multiscale = multiscale_flowline_field(ig; levels=4, min_cutoff=4)
    ig_multiscale_oversampled = multiscale_flowline_field(
        ig;
        levels=4,
        min_cutoff=4,
        spectral_oversample=2,
        extension_seed=11,
    )
    ig_trace = trace_flowline(
        ig_flow,
        ig_seed;
        angle=0.0f0,
        ds=0.05f0 * ig.domain.hx,
        max_steps=6000,
        integrator=:euler,
        boundary_margin=0.0f0,
        stop_when=(z, _, _) -> imag(z) > 0.7f0,
    )
    @test length(ig_trace.points) > 10
    @test ig_trace.termination in (:boundary, :stop_condition, :max_steps, :target)
    @test sum(abs.(diff(ig_trace.points))) / max(abs(last(ig_trace.points) - first(ig_trace.points)), 1f-6) > 1.5f0

    ig_default_trace = trace_flowline(
        ig,
        ig_seed;
        ds=0.05f0 * ig.domain.hx,
        integrator=:euler,
        boundary_margin=0.0f0,
    )
    @test length(ig_default_trace.points) > 10
    @test ig_default_trace.termination in (:target, :boundary, :max_steps)
    if ig_default_trace.termination === :target
        @test abs(last(ig_default_trace.points) - ig_flow.goal) <= max(1.5f0 * ig_flow.goal_radius, 2.1f0 * ig.domain.hx)
    end
    @test length(ig_multiscale.levels) >= 2
    @test last(ig_multiscale.cutoffs) == ig.domain.n - 2
    @test ig_multiscale_oversampled.domain.n == 2 * (ig.domain.n - 1) + 1

    ig_multiscale_trace = trace_flowline(
        ig_multiscale,
        ig_seed;
        angle=0.0f0,
        max_steps=1500,
    )
    @test length(ig_multiscale_trace.points) > 10
    @test ig_multiscale_trace.termination in (:boundary, :max_steps, :multiscale_fallback, :stalled)

    ig_multiscale_oversampled_trace = trace_flowline(
        ig_multiscale_oversampled,
        ig_seed;
        angle=0.0f0,
        max_steps=1500,
    )
    @test length(ig_multiscale_oversampled_trace.points) > 10
    @test ig_multiscale_oversampled_trace.termination in (:boundary, :max_steps, :multiscale_fallback, :stalled)

    ig_multiscale_fan = trace_angle_fan(
        ig_multiscale,
        ig_seed,
        (-0.2f0, 0.0f0, 0.2f0);
        max_steps=1500,
    )
    @test length(ig_multiscale_fan) == 3

    ig_multiscale_fan_field = trace_angle_fan(
        ig,
        ig_seed,
        (-0.2f0, 0.0f0, 0.2f0);
        multiscale=true,
        levels=4,
        min_cutoff=4,
        max_steps=1500,
    )
    @test length(ig_multiscale_fan_field) == 3

    ig_fan = trace_angle_fan(
        ig,
        ig_seed,
        (-0.2f0, 0.0f0, 0.2f0);
        ds=0.05f0 * ig.domain.hx,
        max_steps=1500,
        boundary_margin=0.0f0,
    )
    @test length(ig_fan) == 3

    ig64 = sample_chordal_square_ig_field(33, 2.0; seed=321, T=Float64)
    sle_ds = 0.05 * ig64.domain.hx
    sle_fan = trace_sle_fan(ig64; angles=(0.0,), ds=sle_ds, boundary_margin=0.0, max_steps=nothing)
    @test length(sle_fan.traces) == 1
    @test sle_fan.seed ≈ complex(0.0, ig64.domain.ymin + sle_ds)
    @test sle_fan.traces[1].termination in (:target, :boundary, :max_steps)

    square = square_domain(33; T=Float32)
    constant_flow = flowline_field(zeros(Float32, 33, 33), square, 1.0f0)

    fan = trace_angle_fan(
        constant_flow,
        0.0f0 + 0.0f0im,
        (-Float32(pi) / 4, 0.0f0, Float32(pi) / 4);
        ds=0.5f0 * square.hx,
        max_steps=400,
        boundary_margin=0.0f0,
    )
    @test length(fan) == 3
    @test all(length(trace.points) > 5 for trace in fan)
    @test imag(fan[2].points[2] - fan[2].points[1]) > 0
    @test real(fan[1].points[2] - fan[1].points[1]) > 0
    @test real(fan[3].points[2] - fan[3].points[1]) < 0

    mktempdir() do tmp
        export_web_binary(distances3d, 3; path_step=2, mesh_downscale=2, dir=tmp)
        @test isfile(joinpath(tmp, "lfpp_paths.bin"))
        @test isfile(joinpath(tmp, "lfpp_shells.bin"))
        @test isfile(joinpath(tmp, "meta.json"))

        confluence_dir = joinpath(tmp, "confluence")
        export_confluence_web(distances3d; path_step=2, dir=confluence_dir)
        @test isfile(joinpath(confluence_dir, "confluence_edges.bin"))
        @test isfile(joinpath(confluence_dir, "meta.json"))

        sphere_dir = joinpath(tmp, "sphere")
        export_sphere_web(distances3d, 4; point_step=2, mesh_downscale=2, shell_half_width=1.0f0, dir=sphere_dir)
        @test isfile(joinpath(sphere_dir, "metric_points.bin"))
        @test isfile(joinpath(sphere_dir, "shell_frames.bin"))
        @test isfile(joinpath(sphere_dir, "meta.json"))

        ball_dir = joinpath(tmp, "metric_ball")
        export_metric_ball_web(distances3d, 4; mesh_downscale=2, dir=ball_dir)
        @test isfile(joinpath(ball_dir, "ball_frames.bin"))
        @test isfile(joinpath(ball_dir, "slice_z.bin"))
        @test isfile(joinpath(ball_dir, "meta.json"))

        slice_dir = joinpath(tmp, "slice")
        export_slice_web(distances3d; dir=slice_dir)
        @test isfile(joinpath(slice_dir, "slice_x.bin"))
        @test isfile(joinpath(slice_dir, "slice_y.bin"))
        @test isfile(joinpath(slice_dir, "slice_z.bin"))
        @test isfile(joinpath(slice_dir, "palette.json"))
        @test isfile(joinpath(slice_dir, "meta.json"))

        vtk2d_base = joinpath(tmp, "sample2d")
        export_vtk(distances2d, vtk2d_base; path_step=2)
        @test isfile(vtk2d_base * "_ColorPreset.json")
        @test any(isfile(vtk2d_base * suffix) for suffix in ("_Volume.vti", "_Volume.vtr"))
        @test isfile(vtk2d_base * "_Paths.vtu")

        vtk3d_base = joinpath(tmp, "sample3d")
        export_vtk(distances3d, vtk3d_base; path_step=2)
        @test isfile(vtk3d_base * "_ColorPreset.json")
        @test any(isfile(vtk3d_base * suffix) for suffix in ("_Volume.vti", "_Volume.vtr"))
        @test isfile(vtk3d_base * "_Paths.vtu")
    end
end
