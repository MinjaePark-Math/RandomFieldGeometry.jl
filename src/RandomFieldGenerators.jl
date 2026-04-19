module RandomFieldGenerators

using FFTW
using Random
using Base.Threads

export IGConstants, IGField, SquareDomain
export add_interior_singularity!, boundary_seed, domain_coordinates, ig_constants, ig_critical_angle
export sample_chordal_square_ig_field, sample_interior_ig_field, square_chordal_boundary_data
export square_domain, square_seed_grid
export dirichlet_fgf, dirichlet_gff, dirichlet_lgf

include("RandomFieldGenerators/DirichletFields.jl")
include("RandomFieldGenerators/ImaginaryGeometryFields.jl")

end # module
