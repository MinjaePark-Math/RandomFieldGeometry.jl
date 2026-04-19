include(joinpath(@__DIR__, "ig_script_utils.jl"))

function main(args=ARGS)
    opts = parse_cli_args(args)
    field = sample_chordal_from_opts(opts)
    config = build_trace_config(opts, field.domain)
    trace_kwargs = config.trace_kwargs
    scale_kwargs = trace_scale_kwargs(config)

    theta_center = parse_real_expr(get_opt(opts, "theta_center", "0"))
    theta_left = parse_real_expr(get_opt(opts, "theta_left", "pi/2"))
    theta_right = parse_real_expr(get_opt(opts, "theta_right", "pi/2"))
    nflow = parse(Int, get_opt(opts, "nflow", "25"))
    angles = centered_angles(theta_center, theta_left, theta_right, nflow)
    seed_point = complex((field.domain.xmin + field.domain.xmax) / 2, field.domain.ymin + config.ds)

    println("tracing SLE/IG fan: grid=$(field.domain.n), curves=$(length(angles)), integrator=$(trace_kwargs.integrator), multiscale=$(config.multiscale)")
    println("  ds/h = $(config.ds_over_h)")
    flush(stdout)

    traces = RFG.trace_angle_fan(field, seed_point, angles; scale_kwargs..., trace_kwargs...)
    terminations = trace_termination_counts(traces)

    println("SLE/IG fan")
    println("  κ = ", field.constants.kappa, ", χ = ", field.constants.chi, ", λ = ", field.constants.lambda)
    println("  grid = ", field.domain.n, ", curves = ", length(traces), ", seed = ", seed_point)
    println("  ds = ", config.ds, " (ds/h = ", config.ds_over_h, ")")
    println("  terminations = ", terminations)

    title = "IG/SLE fan, κ=$(round(field.constants.kappa, digits=3)), $(field.boundary_mode)"
    max_plot_points = maybe_optional_int(opts, "max_plot_points")
    fig = plot_ig_traces(field, traces; angles=angles, title=title,
        show_field=parse_bool(get_opt(opts, "show_field", "false"), false),
        figure_px=parse(Int, get_opt(opts, "figure_px", "1400")),
        linewidth=parse_real_expr(get_opt(opts, "linewidth", "1.8")),
        line_alpha=parse_real_expr(get_opt(opts, "line_alpha", "0.94")),
        field_alpha=parse_real_expr(get_opt(opts, "field_alpha", "0.80")),
        show_seeds=parse_bool(get_opt(opts, "show_seeds", "true"), true),
        max_points_per_trace=max_plot_points)

    output = get_opt(opts, "output", "example_output/imaginary_geometry/fan/sle_fan.png")
    save_ig_figure(fig, output)
    return compact_trace_summary(
        output=output,
        terminations=terminations,
        ds=config.ds,
        ds_over_h=config.ds_over_h,
        attempts=1,
        retried=false,
    )
end

main(; kwargs...) = main(keyword_cli_args(kwargs))

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
