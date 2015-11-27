# indices for measured variables: LH, FSH, E2, P4
MEASURED = [2,7,24,25]
# indices for the parameters to be sampled
# veraltetes modell [4, 6, 10, 17, 21, 22, 26, 32, 34, 51, 52, 53, 54, 55, 62, 65, 68, 95, 98, 101, 103]
hillind = [4, 6, 10, 18, 20, 22, 26, 33, 36, 39, 43, 47, 49, 52, 55, 59, 65, 95, 98, 101, 103]
SAMPLEPARMS = deleteat!(collect(1:103), hillind)



speciesnames = open(readlines, "data/model/speciesnames.txt")
parameternames = open(readlines, "data/model/parameternames.txt")

""" load the patient data and return a vector of Arrays, each of shape 4x31 denoting the respective concentration or NaN if not available """
function loadpfizer(path = "data/pfizer_normal.txt")
  data = readtable(path, separator='\t')
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
  results
end

function loadparms()
  parmat = matread("parameters.mat")
  parms  = vec(parmat["para"])
  y0     = vec(parmat["y0_m16"])
  parms, y0
end

""" load parameters and initial data and run a cycle with gync """
function test_gync()
  parms, y0 = loadparms() 
  tspan = collect(1:0.1:56.0)
  @time y = gync(y0, tspan, parms)
  
  plot(
    melt(DataFrame(vcat(tspan', y[MEASURED,:])'), :x1),
    x = :x1, y = :value, color = :variable, Geom.line)
end

""" likelihood (up to proport.) for the parameters given the patientdata """
function loglikelihood(data::Matrix{Float64}, parms::Vector{Float64}, y0::Vector{Float64})
  negparms = collect(1:length(parms))[parms.<0]
  negy0    = collect(1:length(y0)   )[y0   .<0]
  if length(negparms)+length(negy0) > 0
    println("negative parms: ", negparms, ", y0: ", negy0)
    return -Inf
  end

  tspan = Array{Float64}(collect(1:31))
  y = gync(y0, tspan, parms)[MEASURED,:]
  sum(isnan(y)) > 0 && return -Inf
  #sre = minimum(
  #  [squaredrelativeerror(data, translatecol(y, transl)) 
  #  for transl in 0:30])
  sre = squaredrelativeerror(data, y)
  llh = -1/(2*SIGMA_RHO^2) * sre
  #rand()<0.01 && println("$llh   $(y0[1]) $(parms[SAMPLEPARMS][1])")
  llh
end

# cache loglikelihood to evade double evaluation, see https://github.com/brian-j-smith/Mamba.jl/issues/68
cachedllh = cache(loglikelihood,3)

""" componentwise squared relative difference of two matrices """
function squaredrelativeerror(data1::Matrix, data2::Matrix)
  diff = data1 - data2
  # TODO: divide by data1 or data2?
  rdiff = diff ./ data2
  sre   = sumabs2(rdiff[!isnan(rdiff)]) / length(!isnan(rdiff))
end

""" cyclic translation of the columns of `data` by `transl` to the left """
function translatecol(data::Matrix, transl::Integer)
  hcat(data[:, 1+transl:31], data[:, 1:transl])
end

""" given the desired mean and std, return the mu and scale parms for the lognormal distribtuion. """
function lognormalparms(mean, std)
  var = std^2
  s2 = log(1 + var / mean^2)
  mu = log(mean) - s2 / 2
  (mu, sqrt(s2))
end

""" Return the Bayesian Model with priors y0 ~ LN(y0), parms' ~ LN(parms'). Here parms' denotes the sampled parameters, while `parms` are all parameters. """
function gyncmodel(data::Matrix, parms::Vector, y0::Vector)
  # copy for mutating via mergeparms!
  tparms      = copy(parms)
  sparms      = parms[SAMPLEPARMS]

  m = Model(
    y0 = Stochastic(1,
    () -> independentmixtureprior(SIGMA_Y0)), 
      
    sparms = Stochastic(1,
    () -> UnivariateDistribution[Truncated(Flat(),0,p*SIGMA_PARMS) for p in sparms]),
      
    parms = Logical(1,
      (sparms) -> (tparms[SAMPLEPARMS] = sparms; tparms), false),
      
    data = Stochastic(2,
      (y0, parms) -> DensityDistribution(size(data),
        data -> cachedllh(data, parms.value, y0.value), log=true),  false))

  inputs = Dict{Symbol,Any}()
  inits  = Dict{Symbol,Any}(:y0 => y0, :sparms => sparms, :data => data)
  m, inputs, [inits]
end

function run_mcmc(person=1 ;scheme=:AMM, iters=10, relpropvariance=0.01, burnin=0)
  parms, y0 = loadparms()
  data = loadpfizer()[person]

  samplingnodes = [:sparms, :y0]
  proposalvariance = diagm(abs2(vcat(parms[SAMPLEPARMS], y0) * relpropvariance));

  schemes = Dict{Any,Array{Mamba.Sampler,1}}(
    :NUTS => [NUTS(samplingnodes)],
    :AMM  => [AMM (samplingnodes, proposalvariance)],
    :MALA => [MALA(samplingnodes, 1, proposalvariance)],
    :AMWG => [AMWG(samplingnodes, diag(proposalvariance))])

  m, inp, ini = gyncmodel(data, parms, y0)
  setsamplers!(m, schemes[scheme])
  mcmc(m, inp, ini, iters, burnin=burnin)
end

function mcchannel(chains::ModelChains, iters, block=100)
  c=Channel(1)
  @schedule begin
    for i=1:ceil(Int,iters/block)
      ref = @spawn mcmc(chains, min(iters - (i-1)*block, block), verbose=false)
      wait(ref)
      chain = fetch(ref)
      isopen(c) || break
      isready(c) && take!(c)
      put!(c,chains)
    end
  end
  c
end
    
function mcsignal(chains::ModelChains, iters=100_000, block=100)
  x = Input(chains)
  @schedule begin
    for i=1:ceil(Int,iters/block)
      ref = @spawn mcmc(chains, min(iters - (i-1)*block, block), verbose=false)
      wait(ref)
      chains = fetch(ref)
      push!(x, chains)
    end
  end
  x
end

