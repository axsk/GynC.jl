using Mamba
using Distributions: pdf
using JLD, HDF5
using DataFrames
using Requires

import Sundials


export Lausanne, Pfizer
export Config
export mcmc, batch
export load, save


include("constants.jl")


type Subject
  data::Array{Float64}
  id::Any
end

data(s::Subject) = s.data


type Config
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  sigma_y0::Real    # y0 prior mixture component std = ref. solution std * sigma_y0
  parms_bound::Vector # upper bound of flat parameter prior
  relprop::Real     # relative proposal variance
  thin::Integer     # thinning intervall
  init::Vector      # initial sample
end

Config() = Config(Lausanne(1))

Config(s::Subject; args...) = Config(data(s); args...)

Config(data; sigma_rho=0.1, sigma_y0=1, parms_bound::Real=5, relprop=0.1, thin=1, init=refinit) =
  Config(data, sigma_rho, sigma_y0, parms_bound * refparms, relprop, thin, init)


type Sampling
  samples::Array
  logprior::Vector
  loglikelihood::Vector
  logpost::Vector
  config::Config
  model
end


include("data/lausanne.jl")
include("data/pfizer.jl")

include("utils.jl")
include("distributions.jl")
include("rhs.jl")
include("model.jl")
include("simulate.jl")

@require PyPlot include("plot.jl")
