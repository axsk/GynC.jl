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

ModelConfig(person::Int=1; kwargs...) = ModelConfig(pfizerdata(person); kwargs...)

function ModelConfig(data::Matrix; sigma_rho=0.05, sigma_y0=1, parms_bound=5)
  isa(parms_bound, Real) && parms_bound = parms_bound * mleparms
  ModelConfig(data, sigma_rho, sigma_y0, parms_bound)
end

""" Return the Bayesian Model with priors y0 ~ LN(y0), parms' ~ LN(parms'). Here parms' denotes the sampled parameters, while `parms` are all parameters. """
function model(c::ModelConfig)

  tparms = copy(mleparms)

  Model(
    y0 = Stochastic(1,
      () -> independentmixtureprior(mlegync(), c.sigma_y0)), 
      
    sparms = Stochastic(1,
      () -> UnivariateDistribution[Truncated(Flat(), 0, parbound) for parbound in c.parms_bound]),
      
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
    warn("encountered nan in gync result")
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

""" load the (externally computed) maximal likelihood estimates """
function loadmles()
  parmat = matread(joinpath(datadir, "parameters.mat"))
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  parms, y0
end

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function pfizerdata(person)
  data = readtable(joinpath(datadir,"pfizer_normal.txt"), separator='\t')
  results = Vector()
  map(groupby(data, 6)) do subject
    p = fill(NaN, 4, 31)
    for measurement in eachrow(subject)
      # map days to 1-31
      day = (measurement[1]+30)%31+1
      for i = 1:4
        val = measurement[i+1]
        p[i,day] = isa(val, Number) ? val : NaN
      end
    end
    push!(results,p)
  end
  results[person]
end
