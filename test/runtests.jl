using Test
using KernelAbstractions
using RandomFieldGeometry

@testset "RandomFieldGeometry.jl" begin
    @test size(dirichlet_gff(2, 8, 123)) == (7, 7)
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
