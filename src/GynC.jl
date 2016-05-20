module GynC

using Mamba, Distributions
using JLD, HDF5
using DataFrames
using Requires

import Sundials
import ForwardDiff # to compute derivative of objective

export Config, Lausanne, Pfizer

type Sampling
  samples::Array
  logprior::Vector
  loglikelihood::Vector
  logpost::Vector
  config::Config
end

include("gync/gync.jl")

export GynCConfig
export mcmc, batch
export load, save

include("projectsimplex.jl")
include("priorestimation.jl")

export WeightedChain
export emiteration!, euler_A!, euler_phih!

end
