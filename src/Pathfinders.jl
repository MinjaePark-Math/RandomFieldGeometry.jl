module Pathfinders
using KernelAbstractions

export solve_fpp

@kernel function fim_active_block_kernel_2d!(distances, weights, active_blocks, next_active_blocks, M, blocks_dim, block_size)
    I = @index(Global, NTuple)
    i, j = I

    group_I = @index(Group, NTuple)
    bx, by = group_I

    local_I = @index(Local, NTuple)
    tx, ty = local_I

    if active_blocks[bx, by]
        if i <= M && j <= M
            d_old = distances[i, j]
            w = weights[i, j]
            d_new = d_old

            if i > 1
                d_new = min(d_new, distances[i - 1, j] + w)
            end
            if i < M
                d_new = min(d_new, distances[i + 1, j] + w)
            end
            if j > 1
                d_new = min(d_new, distances[i, j - 1] + w)
            end
            if j < M
                d_new = min(d_new, distances[i, j + 1] + w)
            end

            if d_new < d_old
                distances[i, j] = d_new
                next_active_blocks[bx, by] = true

                if tx == 1 && bx > 1
                    next_active_blocks[bx - 1, by] = true
                end
                if tx == block_size[1] && bx < blocks_dim[1]
                    next_active_blocks[bx + 1, by] = true
                end
                if ty == 1 && by > 1
                    next_active_blocks[bx, by - 1] = true
                end
                if ty == block_size[2] && by < blocks_dim[2]
                    next_active_blocks[bx, by + 1] = true
                end
            end
        end
    end
end

@kernel function fim_active_block_kernel_3d!(distances, weights, active_blocks, next_active_blocks, M, blocks_dim, block_size)
    I = @index(Global, NTuple)
    i, j, k = I

    group_I = @index(Group, NTuple)
    bx, by, bz = group_I

    local_I = @index(Local, NTuple)
    tx, ty, tz = local_I

    if active_blocks[bx, by, bz]
        if i <= M && j <= M && k <= M
            d_old = distances[i, j, k]
            w = weights[i, j, k]
            d_new = d_old

            if i > 1
                d_new = min(d_new, distances[i - 1, j, k] + w)
            end
            if i < M
                d_new = min(d_new, distances[i + 1, j, k] + w)
            end
            if j > 1
                d_new = min(d_new, distances[i, j - 1, k] + w)
            end
            if j < M
                d_new = min(d_new, distances[i, j + 1, k] + w)
            end
            if k > 1
                d_new = min(d_new, distances[i, j, k - 1] + w)
            end
            if k < M
                d_new = min(d_new, distances[i, j, k + 1] + w)
            end

            if d_new < d_old
                distances[i, j, k] = d_new
                next_active_blocks[bx, by, bz] = true

                if tx == 1 && bx > 1
                    next_active_blocks[bx - 1, by, bz] = true
                end
                if tx == block_size[1] && bx < blocks_dim[1]
                    next_active_blocks[bx + 1, by, bz] = true
                end
                if ty == 1 && by > 1
                    next_active_blocks[bx, by - 1, bz] = true
                end
                if ty == block_size[2] && by < blocks_dim[2]
                    next_active_blocks[bx, by + 1, bz] = true
                end
                if tz == 1 && bz > 1
                    next_active_blocks[bx, by, bz - 1] = true
                end
                if tz == block_size[3] && bz < blocks_dim[3]
                    next_active_blocks[bx, by, bz + 1] = true
                end
            end
        end
    end
end

function _solve_fpp_impl(
    weights32,
    backend,
    block_size,
    kernel_factory,
    init_distances,
    active_cpu,
    next_active_cpu,
    ndrange,
    grid_label::AbstractString;
    max_iters::Union{Nothing,Integer},
    sweep_factor::Integer,
    print_every::Integer,
    return_info::Bool,
)
    M = size(weights32, 1)
    iter_cap = isnothing(max_iters) ? M * sweep_factor : Int(max_iters)

    println("  -> Initializing Active-Block FIM Data...")
    init_start_time = time()

    d_dev = KernelAbstractions.allocate(backend, Float32, size(weights32)...)
    w_dev = KernelAbstractions.allocate(backend, Float32, size(weights32)...)
    KernelAbstractions.copyto!(backend, d_dev, init_distances)
    KernelAbstractions.copyto!(backend, w_dev, weights32)

    active_dev = KernelAbstractions.allocate(backend, Bool, size(active_cpu)...)
    next_active_dev = KernelAbstractions.allocate(backend, Bool, size(next_active_cpu)...)
    KernelAbstractions.copyto!(backend, active_dev, active_cpu)
    KernelAbstractions.copyto!(backend, next_active_dev, next_active_cpu)

    blocks_dim = size(active_cpu)
    kernel! = kernel_factory(backend, block_size)
    init_elapsed = round(time() - init_start_time, digits=2)
    println("  -> FIM initialization complete. Time: $(init_elapsed)s")

    println("  -> Igniting Wavefront... (Grid: $grid_label)")
    sweeps = 0
    converged = false
    start_time = time()
    final_active_blocks = count(active_cpu)

    for iter in 1:iter_cap
        kernel!(d_dev, w_dev, active_dev, next_active_dev, M, blocks_dim, block_size, ndrange=ndrange)
        KernelAbstractions.synchronize(backend)

        KernelAbstractions.copyto!(backend, next_active_cpu, next_active_dev)
        final_active_blocks = count(next_active_cpu)

        if !any(next_active_cpu)
            sweeps = iter
            converged = true
            break
        end

        KernelAbstractions.copyto!(backend, active_dev, next_active_cpu)
        fill!(next_active_cpu, false)
        KernelAbstractions.copyto!(backend, next_active_dev, next_active_cpu)

        if iter % print_every == 0
            print(".")
        end
    end

    elapsed = round(time() - start_time, digits=2)
    if converged
        println("\n  -> Converged in $sweeps sweeps. Time: $(elapsed)s")
    else
        sweeps = iter_cap
        @warn "Stopped after $iter_cap sweeps without a convergence certificate. Returning the current distance field."
    end

    final_distances = Array(d_dev)

    info = (
        converged=converged,
        sweeps=sweeps,
        initialization_seconds=Float64(init_elapsed),
        elapsed_seconds=Float64(elapsed),
        max_iters=iter_cap,
        sweep_factor=sweep_factor,
        active_blocks=final_active_blocks,
        backend=string(typeof(backend)),
        dimension=ndims(weights32),
    )

    return return_info ? (distances=final_distances, info=info) : final_distances
end

"""
    solve_fpp(weights; backend=KernelAbstractions.CPU(), max_iters=nothing, sweep_factor=8,
              print_every=100, return_info=false)

Solve the 2D or 3D first-passage percolation problem on a square or cubic lattice
with nearest-neighbor connectivity.
"""
function solve_fpp(
    weights::AbstractArray{<:Real,2};
    backend=KernelAbstractions.CPU(),
    max_iters::Union{Nothing,Integer}=nothing,
    sweep_factor::Integer=8,
    print_every::Integer=100,
    return_info::Bool=false,
)
    size(weights, 1) == size(weights, 2) ||
        throw(ArgumentError("`weights` must be a square 2D array."))

    M = size(weights, 1)
    M >= 1 || throw(ArgumentError("`weights` must be non-empty."))
    sweep_factor >= 1 || throw(ArgumentError("`sweep_factor` must be positive."))
    isnothing(max_iters) || max_iters >= 1 || throw(ArgumentError("`max_iters` must be positive when provided."))
    print_every >= 1 || throw(ArgumentError("`print_every` must be positive."))

    weights32 = Float32.(weights)
    dists_init = fill(Inf32, M, M)
    cx, cy = div(M, 2) + 1, div(M, 2) + 1
    dists_init[cx, cy] = 0.0f0

    block_size = (8, 8)
    blocks_x, blocks_y = cld(M, block_size[1]), cld(M, block_size[2])
    active_cpu = fill(false, blocks_x, blocks_y)
    next_active_cpu = fill(false, blocks_x, blocks_y)
    active_cpu[cld(cx, block_size[1]), cld(cy, block_size[2])] = true

    return _solve_fpp_impl(
        weights32,
        backend,
        block_size,
        fim_active_block_kernel_2d!,
        dists_init,
        active_cpu,
        next_active_cpu,
        (M, M),
        "$M x $M";
        max_iters=max_iters,
        sweep_factor=sweep_factor,
        print_every=print_every,
        return_info=return_info,
    )
end

function solve_fpp(
    weights::AbstractArray{<:Real,3};
    backend=KernelAbstractions.CPU(),
    max_iters::Union{Nothing,Integer}=nothing,
    sweep_factor::Integer=8,
    print_every::Integer=100,
    return_info::Bool=false,
)
    size(weights, 1) == size(weights, 2) == size(weights, 3) ||
        throw(ArgumentError("`weights` must be a cubic 3D array."))

    M = size(weights, 1)
    M >= 1 || throw(ArgumentError("`weights` must be non-empty."))
    sweep_factor >= 1 || throw(ArgumentError("`sweep_factor` must be positive."))
    isnothing(max_iters) || max_iters >= 1 || throw(ArgumentError("`max_iters` must be positive when provided."))
    print_every >= 1 || throw(ArgumentError("`print_every` must be positive."))

    weights32 = Float32.(weights)
    dists_init = fill(Inf32, M, M, M)
    cx, cy, cz = div(M, 2) + 1, div(M, 2) + 1, div(M, 2) + 1
    dists_init[cx, cy, cz] = 0.0f0

    block_size = (8, 8, 8)
    blocks_x, blocks_y, blocks_z = cld(M, block_size[1]), cld(M, block_size[2]), cld(M, block_size[3])
    active_cpu = fill(false, blocks_x, blocks_y, blocks_z)
    next_active_cpu = fill(false, blocks_x, blocks_y, blocks_z)
    active_cpu[cld(cx, block_size[1]), cld(cy, block_size[2]), cld(cz, block_size[3])] = true

    return _solve_fpp_impl(
        weights32,
        backend,
        block_size,
        fim_active_block_kernel_3d!,
        dists_init,
        active_cpu,
        next_active_cpu,
        (M, M, M),
        "$M x $M x $M";
        max_iters=max_iters,
        sweep_factor=sweep_factor,
        print_every=print_every,
        return_info=return_info,
    )
end

function solve_fpp(weights::AbstractArray{<:Real,N}; kwargs...) where {N}
    throw(ArgumentError("`solve_fpp` currently supports only 2D and 3D arrays, got dimension $N."))
end

end # module
