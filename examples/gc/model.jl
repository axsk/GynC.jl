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


type Patient
  data::Array{Float64}
  id::Any
end

data(p::Patient) = p.data
Base.show(io::IO, p::Patient) = print(io,p.id)

type Config
  patient::Patient  # patient measurements
  sigma_rho::Float64   # measurement error / std for likelihood gaussian 
  relprop::Float64     # relative proposal variance
  thin::Int     # thinning intervall
  initparms::Vector{Float64}      # initial sample
  inity0::Vector{Float64}
  priorparms
  priory0
end

data(c::Config) = data(c.patient)


function Config(patient=Lausanne(1); sigma_rho=0.1, relprop=0.1, thin=1, initparms=refparms, inity0=refy0, p_parms=priorparms(5 * initparms), p_y0=priory0(1) ) 
  Config(patient, sigma_rho, relprop, thin, initparms, inity0, p_parms, p_y0)
end

function Base.show(io::IO, c::Config)
  print(io, "Config:
 patient: ", c.patient, "
 sigma:   $(c.sigma_rho)
 relprop: $(c.relprop)
 thin:    $(c.thin)
 init:    $(hash((c.initparms, c.inity0)))
 prior:   $(typeof((c.priorparms, c.priory0)))")
end


### Priors  ###

import Distributions: UnivariateDistribution, Truncated

priory0(sigma::Real) = gaussianmixture(referencesolution(), sigma)
priorparms(αs)       = Distributions.UnivariateDistribution[
  Distributions.Truncated(Mamba.Flat(), 0, α) for α in αs]

function referencesolution(resolution=1)
  sol = gync(refy0, allparms(refparms), collect(0:resolution:30.))
  # since we get a (small) negative value for OvF, impeding the log transformation for the prior, set this to the next minimal value
  for j in 1:size(sol, 2)
    toosmall = sol[:,j] .<= 0
    sol[toosmall, j] = minimum(sol[!toosmall, j])
  end
  sol
end

function gaussianmixture(y::Matrix, stdfactor=1)
   stds = mapslices(std, y, 1) * stdfactor |> vec
   vars = abs2(stds)
   normals = mapslices(yt->Distributions.MvNormal(yt, stds), y, 2) |> vec
   Distributions.MixtureModel(normals)
end


### Sampling specifics ###

parms(x::Vector) = x[1:82]
y0(x::Vector)    = x[83:end]

list(x::Vector) = log(x)
unlist(x::Vector) = exp(x)
# TODO: fix transformation in mcmc

init(c::Config) = (vcat(c.initparms, c.inity0))
dim(c::Config)  = length(c.variate[:])

function SamplerVariate(c::Config)
  linit          = list(init(c))
  
  # note the adjustment to balance out the jump density transformation
  logf = cache(x -> logpost(c, unlist(x)) + sum(x), 3)
  sigma         = eye(length(linit)) * log(1+(c.relprop^2))

  Mamba.SamplerVariate(linit, Mamba.AMMTune(linit, sigma, cache(logf, 3);
    beta = 0.05,
    scale = 2.38))
end

### Density functions)

function logprior(c::Config, x::Vector)
  l = Distributions.logpdf(c.priory0, y0(x))
  for i in 1:82
    l += Distributions.logpdf(c.priorparms[i], x[i])
  end
  l
end

function logpost(c::Config, x::Vector)
  l = logprior(c, x)
  l == -Inf || (l += llh(c, x))
  #rand() < 0.05 && println("$(x[1]) $l")
  l
end

function llh(c::Config, x::Vector) 
  y = gync(c, x, 0:30)[:,measuredinds]

  if any(isnan(y)) > 0
    #Base.warn("encountered NaN in gync result")
    return -Inf
  end
  sre = l2(data(c), y)
  -1/(2*c.sigma_rho^2) * sre
end

" sundials cvode solution to the gyncycle model "
gync(c::Config, x::Vector, tspan) = gync(y0(x), allparms(parms(x)), tspan)

gync(y0, p, t) = Sundials.cvode((t,y,dy) -> gyncycle_rhs!(y,p,dy), y0, convert(Array{Float64, 1}, t))

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  reldiff = diff ./ data1
  return sumabs2(reldiff[!isnan(reldiff)])
end

function l2(data1, data2)
  # TODO: think about the scales
  # NOTE: dependence on amount of measured data
  diff = (data1 - data2) ./ [120 10 400 15]
  sumabs2(diff[!isnan(diff)])
end

