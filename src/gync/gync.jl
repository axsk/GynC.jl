export Lausanne, Pfizer
export GynCConfig
export mcmc, batch
export load, save

### Constants ### 

# indices for measured variables: LH, FSH, E2, P4
const measuredinds = [2,7,24,25]
const hillinds     = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampledinds  = deleteat!(collect(1:103), hillinds)

const refy0       = include("refy0.jl")
const refallparms = include("refparms.jl")
const refparms    = refallparms[sampledinds]
const refinit     = vcat(refparms, refy0)


const speciesnames   = include("speciesnames.jl")
const parameternames = include("parameternames.jl")[sampledinds]
const samplednames   = [parameternames; speciesnames]

### types ###

abstract Config

type GynCConfig <: Config
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  sigma_y0::Real    # y0 prior mixture component std = ref. solution std * sigma_y0
  parms_bound::Vector # upper bound of flat parameter prior
  relprop::Real     # relative proposal variance
  thin::Integer     # thinning intervall
  init::Vector      # initial sample
end

GynCConfig() = GynCConfig(Lausanne(1))

GynCConfig(s::Subject; args...) = GynCConfig(data(s); args...)

GynCConfig(data; sigma_rho=0.1, sigma_y0=1, parms_bound::Real=5, relprop=0.1, thin=1, init=refinit) =
  GynCConfig(data, sigma_rho, sigma_y0, parms_bound * refparms, relprop, thin, init)

type Sampling
  samples::Array
  logprior::Vector
  loglikelihood::Vector
  logpost::Vector
  config::Config
end

const datadir = joinpath(dirname(@__FILE__), "..", "data")

type Subject
  data::Array{Float64}
  id::Any
end

data(s::Subject) = s.data

include("../../data/lausanne.jl")
include("../../data/pfizer.jl")
include("utils.jl")
include("distributions.jl")
include("rhs.jl")
include("model.jl")
include("simulate.jl")

@require PyPlot include("plot.jl")
