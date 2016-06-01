const measuredinds = [2,7,24,25]
const hillinds     = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampledinds  = deleteat!(collect(1:103), hillinds)

const refy0       = include("data/refy0.jl")
const refallparms = include("data/refparms.jl")
const refparms    = refallparms[sampledinds]


const speciesnames   = include("data/speciesnames.jl")
const parameternames = include("data/parameternames.jl")[sampledinds]
const samplednames   = [parameternames; speciesnames]

allparms(parms::Vector) = (p = copy(refallparms); p[sampledinds] = parms; p)


type Subject
  data::Array{Float64}
  id::Any
end

data(s::Subject) = s.data

type Config
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  relprop::Real     # relative proposal variance
  thin::Integer     # thinning intervall
  initparms::Vector      # initial sample
  inity0::Vector
  priorparms
  priory0
end


function Config(data; sigma_rho=0.1, relprop=0.1, thin=1, initparms=refparms, inity0=refy0, p_parms=priorparms(5 * initparms), p_y0=priory0(1) ) 
  Config(data, sigma_rho, relprop, thin, initparms, inity0, p_parms, p_y0)
end

Config() = Config(Lausanne(1))
Config(s::Subject; args...) = Config(data(s); args...)

function Base.show(io::IO, c::Config)
  print(io, "GynC mcmc sampling config
 sigma:   $(c.sigma_rho)
 relprop: $(c.relprop)
 thin:    $(c.thin)
 init:    $(hash((c.initparms, c.inity0)))
 prior:   $(hash((priorparms, priory0)))")
end


### Priors  ###

import Distributions: UnivariateDistribution, Truncated

priory0(sigma::Real) = gaussianmixture(referencesolution(), sigma)
priorparms(αs)       = Distributions.UnivariateDistribution[
  Distributions.Truncated(Mamba.Flat(), 0, α) for α in αs]

function referencesolution(resolution=1)
  sol = gync(refy0, allparms(refparms), collect(0:resolution:30.))
  # since we get a (small) negative value for OvF, impeding the log transformation for the prior, set this to the next minimal value
  for i in 1:size(sol,1)
    sol[i, sol[i,:] .<= 0] = minimum(sol[i, sol[i,:] .> 0])
  end
  sol
end

function gaussianmixture(y::Matrix, stdfactor=1)
   stds = mapslices(std, y, 2) * stdfactor |> vec
   vars = abs2(stds)
   normals = mapslices(yt->Distributions.MvNormal(yt, vars), y, 1) |> vec
   Distributions.MixtureModel(normals)
end


### Sampling specifics ###

parms(x::Vector) = x[1:82]
y0(x::Vector)    = x[83:end]

list(x::Vector) = log(x)
unlist(x::Vector) = exp(x)
# TODO: fix transformation in mcmc

init(c::Config) = (vcat(c.initparms, c.inity0))

function SamplerVariate(c::Config)
  linit          = list(init(c))
  #cachedlogpost = cache(x -> post(c,unlist(x)), 3)
  
  logf = x -> post(c, unlist(x))
  sigma         = eye(length(linit)) * log(1+(c.relprop^2))

  Mamba.SamplerVariate(linit, Mamba.AMMTune(linit, sigma, cache(logf, 3);
    beta = 0.05,
    scale = 2.38))
end

### Density functions)

function prior(c::Config, x::Vector)
  l = Distributions.logpdf(c.priory0, y0(x))
  for i in 1:82
    l += Distributions.logpdf(c.priorparms[i], x[i])
  end
  l
end

function post(c::Config, x::Vector)
  l = prior(c, x)
  l == -Inf || (l += llh(c, x))
  #rand() < 0.05 && println("$(x[1]) $l")
  l
end

function llh(c::Config, x::Vector) 
  tspan = collect(0:30.)
  y = gync(c, x, collect(0:30.))[measuredinds,:]

  if any(isnan(y)) > 0
    #Base.warn("encountered NaN in gync result")
    return -Inf
  end
  sre = l2(c.data, y)
  -1/(2*c.sigma_rho^2) * sre
end

" sundials cvode solution to the gyncycle model "
gync(c::Config, x::Vector, tspan) = gync(y0(x), allparms(parms(x)), tspan)

gync(y0, p, t) = Sundials.cvode((t,y,dy) -> gyncycle_rhs!(y,p,dy), y0, t)'

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  reldiff = diff ./ data1
  return sumabs2(reldiff[!isnan(reldiff)])
end

function l2(data1, data2)
  # TODO: think about the scales
  # NOTE: dependence on amount of measured data
  diff = (data1 - data2) ./ [120, 10, 400, 15]
  sumabs2(diff[!isnan(diff)])
end

