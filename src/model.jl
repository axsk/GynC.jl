# indices for measured variables: LH, FSH, E2, P4
const measuredinds = [2,7,24,25]
const hillinds     = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
const sampledinds  = deleteat!(collect(1:103), hillinds)

const refy0       = include("const/refy0.jl")
const refallparms = include("const/refparms.jl")
const refparms    = refallparms[sampledinds]

function allparms(parms::Vector)
  p = copy(refallparms)
  p[sampledinds] = parms
  p
end

const speciesnames   = include("const/speciesnames.jl")
const parameternames = include("const/parameternames.jl")[sampledinds]
const samplednames   = [parameternames; speciesnames]

type ModelConfig
  data::Matrix      # measurements
  sigma_rho::Real   # measurement error / std for likelihood gaussian 
  sigma_y0::Real    # prior sigma for y0 kernels
  parms_bound::Vector # upper bound of flat parameter prior
end

ModelConfig(s::Subject, args...) = ModelConfig(data(s), args...)

ModelConfig(data, sigma_rho=0.1, sigma_y0=1, parms_bound::Real=5) =
  ModelConfig(data, sigma_rho, sigma_y0, parms_bound * refparms)

function gaussianmixture(y::Matrix, stdfactor=1)
   covariances = mapslices(std, y, 2) * stdfactor |> vec
   normals = mapslices(yt->MvNormal(yt, covariances), y, 1) |> vec
   MixtureModel(normals)
end

# HOTFIX for missing insupport method for MixtureModel
insupport(d::Distributions.MultivariateMixture, x::AbstractVector) = true

""" construct the mamba model """
function model(c::ModelConfig)
  cachedllh = cache(llh,3)

  Model(
    logy0 = Stochastic(1,
      () -> gaussianmixture(log(referencesolution()), c.sigma_y0),
      false),

    y0 = Logical(1,
      (logy0) -> exp(logy0)),
      
    parms = Stochastic(1,
      () -> UnivariateDistribution[Truncated(Flat(), 0, parbound) for parbound in c.parms_bound]),
      
    data = Stochastic(2,
      (y0, parms) -> DensityDistribution(size(c.data),
                       data -> cachedllh(data, allparms(parms.value), y0.value, c.sigma_rho),
                       log=true),
      false),

    loglikelihood = Logical(
      (y0, parms, data) -> cachedllh(data.value, allparms(parms.value), y0.value, c.sigma_rho))
  )
end

function referencesolution(resolution=1)
  sol = gync(refy0, collect(1:resolution:31.), allparms(refparms))
  # since we get a (small) negative value for OvF, impeding the log transformation for the prior, set this to the next minimal value
  for i in 1:size(sol,1)
    sol[i, sol[i,:] .<= 0] = minimum(sol[i, sol[i,:] .> 0])
  end
  sol
end

""" loglikelihood (up to proport.) for the parameters given the patientdata """
function llh(data::Matrix{Float64}, parms::Vector{Float64}, y0::Vector{Float64}, sigma::Real)
  tspan = Array{Float64}(collect(0:30))
  y = gync(y0, tspan, parms)[measuredinds,:]
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

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  reldiff = diff ./ data1
  return sumabs2(reldiff[!isnan(reldiff)])
end
