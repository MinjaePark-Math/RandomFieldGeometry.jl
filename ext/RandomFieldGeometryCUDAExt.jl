module RandomFieldGeometryCUDAExt

using CUDA
using RandomFieldGeometry

function RandomFieldGeometry._cuda_backend_impl()
    return CUDA.functional() ? CUDA.CUDABackend() : nothing
end

end
