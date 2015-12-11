module Gync

using JLD, DataFrames, MAT, Mamba, Distributions

include("utils.jl")
include("priors.jl")
include("loaddata.jl")
include("model.jl")
include("run.jl")

end
