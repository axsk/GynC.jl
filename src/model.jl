# indices for measured variables: LH, FSH, E2, P4
const MEASURED = [2,7,24,25]
const hillinds = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampleparms = deleteat!(collect(1:103), hillinds)
const mleparms, mley0 = loadmles()

type ModelConfig
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  mle_y0::Vector    # prior mles
  mle_parms::Vector
  sigma_y0::Real    # prior sigmas
  sigma_parms::Real
  sampleparms::Vector  # indices of parameters to sample
end

ModelConfig(person=1; kwargs...) = ModelConfig(pfizerdata(person); kwargs...)

function ModelConfig(data::Matrix; sigma_rho=0.1, sigma_y0=1, sigma_parms=20)
  parms, y0 = loadmles()
  ModelConfig(data, sigma_rho, y0, parms, sigma_y0, sigma_parms, sampleparms)
end

""" Return the Bayesian Model with priors y0 ~ LN(y0), parms' ~ LN(parms'). Here parms' denotes the sampled parameters, while `parms` are all parameters. """
function model(c::ModelConfig)

  # copy for mutating via mergeparms!
  tparms = copy(c.mle_parms)
  mle_sparms = c.mle_parms[c.sampleparms]
  mley = mlegync()

  Model(
    y0 = Stochastic(1,
      () -> independentmixtureprior(mley, c.sigma_y0)), 
      
    sparms = Stochastic(1,
      () -> UnivariateDistribution[Truncated(Flat(),0,p*c.sigma_parms) for p in mle_sparms]),
      
    parms = Logical(1,
      (sparms) -> (tparms[c.sampleparms] = sparms; tparms), false),
      
    data = Stochastic(2,
      (y0, parms) -> DensityDistribution(
        size(c.data),
        data -> cachedllh(data, parms.value, y0.value, c.sigma_rho), 
        log=true),
      false))
end

""" likelihood (up to proport.) for the parameters given the patientdata """
function llh(data::Matrix{Float64}, parms::Vector{Float64}, y0::Vector{Float64}, sigma::Real)
  negparms = collect(1:length(parms))[parms.<0]
  negy0    = collect(1:length(y0)   )[y0   .<0]
  if length(negparms)+length(negy0) > 0
    println("negative parms: ", negparms, ", y0: ", negy0)
    return -Inf
  end

  tspan = Array{Float64}(collect(1:31))
  y = gync(y0, tspan, parms)[MEASURED,:]
  any(isnan(y)) > 0 && return -Inf
  sre = squaredrelativeerror(data, y)
  -1/(2*sigma^2) * sre
end

""" cached loglikelihood to evade double evaluation """
# see https://github.com/brian-j-smith/Mamba.jl/issues/68
cachedllh = cache(llh,3)

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  # TODO: divide by data1 or data2?
  rdiff = diff ./ data2
  sre   = sumabs2(rdiff[!isnan(rdiff)]) / length(!isnan(rdiff))
end
