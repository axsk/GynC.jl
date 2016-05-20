""" construct the mamba model """
function model(c::GynCConfig)
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
      false))
end

function referencesolution(resolution=1)
  sol = gync(refy0, allparms(refparms), collect(0:resolution:30.))
  # since we get a (small) negative value for OvF, impeding the log transformation for the prior, set this to the next minimal value
  for i in 1:size(sol,1)
    sol[i, sol[i,:] .<= 0] = minimum(sol[i, sol[i,:] .> 0])
  end
  sol
end

""" loglikelihood (up to proport.) for the parameters given the patientdata """
function llh(data::Matrix{Float64}, parms::Vector{Float64}, y0::Vector{Float64}, sigma::Real)
  tspan = collect(0:30.)
  y = gync(y0, parms, tspan)[measuredinds,:]
  if any(isnan(y)) > 0
    #Base.warn("encountered nan in gync result")
    #try
      #save("llhdebug.jld", "data", data, "parms", parms, "y0", y0, "sigma", sigma)
    #catch
      #println("caught saveexception, worth the effort :)")
    #end
    return -Inf
  end
  sre = distsquared(data, y)
  -1/(2*sigma^2) * sre
end

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

distsquared = l2



### Solve the GynCycle model

" sundials cvode solution to the gyncycle model "
gync_cvode(y0, p, t) = Sundials.cvode((t,y,dy) -> gyncycle_rhs!(y,p,dy), y0, t)'

const fortranpath = joinpath(dirname(@__FILE__),"..","deps","limex","GynC.so")

""" Wrapper to the LIMEX solver for the GynC model 
solve the model for the times t given initial condition y0 and parameters parms, and store the result in y"""
function gync_limex(y0::Vector{Float64}, parms::Vector{Float64}, tspan::Vector{Float64})
  n = 33
  m = length(tspan)
  y = Array{Float64}(n,m)

  @assert length(y0)==33
  @assert length(parms)==114

  ccall((:limstep_, fortranpath), Ptr{Array{Float64,2}},
    (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int64}, Ptr{Int64}, Ptr{Float64}),
    y, copy(y0), tspan, &n, &m, parms)
  y
end
