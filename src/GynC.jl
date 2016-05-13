module GynC

using Mamba, Distributions
using JLD, HDF5
using DataFrames
using Requires

import Sundials

#using Lumberjack

import Mamba.mcmc

export mcmc, batch
export ModelConfig, Lausanne, Pfizer
export load, save

export WeightedChain
export SimpleWeightedChain, GynCChain
export gradient_simplex!, reweight!

const datadir = joinpath(dirname(@__FILE__), "..", "data")

type Subject
  data::Array{Float64}
  id::Any
end

data(s::Subject) = s.data


type ModelConfig
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  sigma_y0::Real    # y0 prior mixture component std = ref. solution std * sigma_y0
  parms_bound::Vector # upper bound of flat parameter prior
end


type Sampling
  data
  logprior
  logllh
  logpost
  model
end


include("utils.jl")
include("../data/lausanne.jl")
include("../data/pfizer.jl")
include("gyncycle.jl")
include("model.jl")
include("io.jl")
include("reweight.jl")

@require PyPlot include("plot.jl")

end
