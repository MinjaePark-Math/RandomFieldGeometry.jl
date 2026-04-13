module RandomFieldGeometryMetalExt

using Metal
using RandomFieldGeometry

function RandomFieldGeometry._metal_backend_impl()
    return Metal.functional() ? Metal.MetalBackend() : nothing
end

end
