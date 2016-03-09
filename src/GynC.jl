module Gync

using Mamba, Distributions
using JLD, HDF5
using DataFrames
#using Lumberjack

export mcmc, ModelConfig, Lausanne, Pfizer

const datadir = joinpath(dirname(@__FILE__), "..", "data")

type Subject
  data::Array{Float64}
  id::Any
end

data(s::Subject) = s.data

include("utils.jl")
include("../data/lausanne.jl")
include("../data/pfizer.jl")
include("model.jl")
include("io.jl")
include("reweight.jl")

end
