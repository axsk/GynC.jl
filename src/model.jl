const measuredinds = [2,7,24,25]

const hillinds     = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampledinds  = deleteat!(collect(1:103), hillinds)

const refparms    = refallparms[sampledinds]

const model_measerrors = [120, 10, 400, 15.]

#const defaultpropvar = include("data/proposals/allcovs.jl") * 2.38^2 / 115
# shouldnt the squares be taken after the log?
uniformpropvar(relprop) = eye(116) * log(1+(relprop^2))

const defaultpropvar = uniformpropvar(0.1)

const samplednames   = [parameternames[sampledinds]; speciesnames; "periodlength"]

function allparms(parms::Vector)
  p = copy(refallparms)
  p[sampledinds] = parms
  p
end


type Patient
  data::Array{Float64,2}
  id::Any
end

data(p::Patient) = p.data
Base.show(io::IO, p::Patient) = print(io,p.id)

type Config
  patient::Patient  # patient measurements
  sigma_rho::Float64   # measurement error / std for likelihood gaussian 
  propvar::Matrix{Float64}     # covariance of guassian proposal in log-parameter space
  adapt::Bool
  thin::Int     # thinning intervall
  initparms::Vector{Float64}      # initial sample
  inity0::Vector{Float64}
  priorparms
  priory0
end

data(c::Config) = data(c.patient)
filename(c::Config) = "p$(c.patient.id)s$(c.sigma_rho)r$(c.propvar|>trace)t$(c.thin)a$(c.adapt).jld"


function Config(patient=Lausanne(1); sigma_rho=0.1, propvar=defaultpropvar, adapt=false, thin=1, initparms=refparms, inity0=refy0, p_parms=priorparms(5 * initparms), p_y0=priory0(1) ) 
  Config(patient, sigma_rho, propvar, adapt, thin, initparms, inity0, p_parms, p_y0)
end

function Base.show(io::IO, c::Config)
  print(io, "Config:
 patient: ", c.patient, "
 sigma:   $(c.sigma_rho)
 tr(initprop): $(c.propvar |> trace)
 adapt:   $(c.adapt)
 thin:    $(c.thin)
 init:    $(hash((c.initparms, c.inity0)))
 prior:   $(typeof((c.priorparms, c.priory0)))")
end


### Priors  ###

priory0(sigma::Real) = gaussianmixture(referencesolution(), sigma)
priorparms(αs)       = Distributions.UnivariateDistribution[
  Distributions.Truncated(Mamba.Flat(), 0, α) for α in αs]

function gaussianmixture(y::Matrix, stdfactor=1)
   stds = mapslices(std, y, 1) * stdfactor |> vec
   normals = mapslices(yt->Distributions.MvNormal(yt, stds), y, 2) |> vec
   Distributions.MixtureModel(normals)
end


### Sampling specifics ###

parms(x::Vector) = x[1:82]
y0(x::Vector)    = x[83:115]
period(x::Vector) = x[116]

list(x::Vector) = log(x)
unlist(x::Vector) = exp(x)

" sundials cvode solution to the gyncycle model "
forwardsol(x::Vector, tspan=0:30) = try
  gync(y0(x), allparms(parms(x)), tspan)
catch
  fill(NaN, length(tspan), length(y0(x)))
end

# TODO: fix transformation in mcmc

init(c::Config) = (vcat(c.initparms, c.inity0, 28.))
dim(c::Config)  = length(c.variate[:])

function SamplerVariate(c::Config)
  linit          = list(init(c))
  
  # note the adjustment to balance out the jump density transformation
  # justified by g(x|x')/g(x'|x) = lnN(x';ln(x),s) / lnN(x;ln(x'),s) = x'/x
  logf = cache(x -> logpost(c, unlist(x)) + sum(x), 3)

  Mamba.SamplerVariate(linit, Mamba.AMMTune(linit, c.propvar, logf;
    beta = 0.05,
    scale = 2.38))
end

### Density functions

function logprior(c::Config, x::Vector)
  l = Distributions.logpdf(c.priory0, y0(x))
  for i in 1:82
    l += Distributions.logpdf(c.priorparms[i], x[i])
  end
  l += Distributions.logpdf(Distributions.Normal(28, 2), period(x))
end

function logpost(c::Config, x::Vector)
  l = logprior(c, x)
  l == -Inf || (l += llh(c, x))
  #rand() < 0.05 && println("$(x[1]) $l")
  l
end

meastimes(days, periods, periodlength) = vcat([(1:days) + periodlength * p for p in 0:periods-1]...)

function llh(c::Config, x::Vector, periods::Int=2)
  ndata = size(data(c), 1)

  t = meastimes(ndata, periods, period(x))

  # simulate the trajectory
  local y
  try
    # sort the times for the ode solver, and resort the results
    perm = sortperm(t)
    y = forwardsol(x, t[perm])[invperm(perm),measuredinds]
  catch e
    Base.warn("forward solution solver threw: $e")
    return -Inf
  end

  if any(isnan(y)) > 0
    Base.warn("encountered NaN in solution result")
    return -Inf
  end


  # TODO: check index/time relations
  # simulate periodic data by comparing it to the shifted simulated trajectory reapeatedy
  l = 0.

  for p = 0:periods-1
    periodtimes = (1:ndata) + ndata * p
    yperiod = y[periodtimes, :]
    l += llh_measerror(data(c) - yperiod)
  end
  
  l
end

function llh_measerror(error::Matrix)
  scaled = error ./ model_measerrors'
  -sumabs2(scaled[!isnan(scaled)])
end
