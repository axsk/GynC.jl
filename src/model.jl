const datadir = joinpath(dirname(@__FILE__), "..", "data")

# indices for measured variables: LH, FSH, E2, P4
const MEASURED = [2,7,24,25]
const hillinds = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampleparms = deleteat!(collect(1:103), hillinds)

const mleparms, mley0 = loadmles()
const sampledmles = [mleparms[sampleparms]; mley0]

const speciesnames = open(readlines, joinpath(datadir, "speciesnames.txt"))
const parameternames = open(readlines, joinpath(datadir, "parameternames.txt"))
const samplednames = [parameternames[sampleparms]; speciesnames]

type ModelConfig
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  sigma_y0::Real    # prior sigma for y0 kernels
  parms_bound::Vector # upper bound of flat parameter prior
end

ModelConfig(s::Subject; kwargs...) = ModelConfig(data(s); kwargs...)

function ModelConfig(data::Matrix; sigma_rho=0.1, sigma_y0=1, parms_bound=5)
  isa(parms_bound, Real) && (parms_bound = parms_bound * mleparms)
  ModelConfig(data, sigma_rho, sigma_y0, parms_bound)
end

""" Return the Bayesian Model with priors y0 ~ LN(y0), parms' ~ LN(parms'). Here parms' denotes the sampled parameters, while `parms` are all parameters. """
function model(c::ModelConfig)

  tparms = copy(mleparms)

  Model(
    y0 = Stochastic(1,
      () -> independentmixtureprior(mlegync(), c.sigma_y0)), 
      
    sparms = Stochastic(1,
      () -> UnivariateDistribution[Truncated(Flat(), 0, parbound) for parbound in c.parms_bound[sampleparms]]),
      
    parms = Logical(1,
      (sparms) -> (tparms[sampleparms] = sparms; tparms), false),
      
    data = Stochastic(2,
      (y0, parms) -> DensityDistribution(
        size(c.data),
        data -> cachedllh(data, parms.value, y0.value, c.sigma_rho), 
        log=true),
      false))
end

""" likelihood (up to proport.) for the parameters given the patientdata """
function llh(data::Matrix{Float64}, parms::Vector{Float64}, y0::Vector{Float64}, sigma::Real)
  tspan = Array{Float64}(collect(1:31))
  y = gync(y0, tspan, parms)[MEASURED,:]
  if any(isnan(y)) > 0
    #Base.warn("encountered nan in gync result")
    #try
      #save("llhdebug.jld", "data", data, "parms", parms, "y0", y0, "sigma", sigma)
    #catch
      #println("caught saveexception, worth the effort :)")
    #end
    return -Inf
  end
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
