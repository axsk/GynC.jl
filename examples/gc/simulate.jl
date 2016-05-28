function Sampling(c::Config, v::SamplerVariate)
  Sampling(Matrix{Float64}(0,115), Float64[], Float64[], Float64[], c, v)
end

function prior(c::Config, x::Vector)
  pi_y = gaussianmixture(log(referencesolution()), c.sigma_y0)
  pi_p = UnivariateDistribution[Truncated(Flat(), 0, b) for b in c.parms_bound]

  l = logpdf(pi_y, x[83:end])
  for i in 1:82
    l += logpdf(pi_p[i], x[i])
  end
  l
end

function post(c::Config, x::Vector)
  l = prior(c, x)
  l == -Inf && return l
  l += llh(c.data, allparms(x[1:82]), exp(x[83:end]), c.sigma_rho)
  l
end

function simulate(c::Config, iters::Int)
  init = vcat(refparms, log(refy0))
  cpost = cache(x -> post(c, x), 3)
  sigma = eye(length(c.init)) * log(1+(c.relprop^2))

  v = SamplerVariate(init, Mamba.AMMTune(init, sigma, cpost; beta=0.05, scale=2.38))
  simulate!(Sampling(c, v), iters)
end

function simulate!(s::Sampling, iters::Int)
  thin  = s.config.thin
  n     = round(Int, iters/thin, RoundDown)
  v     = s.model
  x     = Array(Float64, n, length(v[:]))
  priors = Array(Float64, n)
  posts  = Array(Float64, n)

  cpost  = get(v.tune.logf)

  for i in 1:n
    for j in 1:thin
      sample!(v)
    end
    x[i,:] = v[:]
    priors[i] = prior(s.config, v[:])
    posts[i]  = cpost(v[:])
  end

  Sampling(
    vcat(s.samples, x),
    vcat(s.logprior, priors),
    vcat(s.loglikelihood, posts),
    vcat(s.logpost, posts),
    s.config,
    s.model)
end


### old code ###

" simulate `iters` iteration of the markov chain corresponding to the model specified by the `Config`, with initial values `inity0` and `initparms` (defaulting to the reference solution). The initial proposal density at point x is a Log-normal distribution with median x standard deviation x*`relprop` " 
function mcmc(c::Config, iters)
  m = model(c)

  initparms, inity0 = sampletoparms(c.init)
  initparms = initparms[sampledinds]

  nparms = length(inity0) + length(initparms)
  prop = log(1+(c.relprop^2)) * eye(nparms)

  inp = Dict{Symbol,Any}()
  inits = [Dict{Symbol,Any}(:logy0 => log(inity0), :parms => initparms, :data => c.data)]

  setinputs!(m, inp)
  setinits!(m, inits)
  setsamplers!(m, [AMM([:parms, :logy0], prop, adapt=:all)])

  Sampling(sample(m, iters, c.thin)..., c, m)
end

function sample(m::Mamba.Model, n::Int, thin::Int)
  n = round(Int, n/thin, RoundDown)
  samples = Array(Float64, n, length(unlist(m, true)))
  logprior = Array(Float64, n)
  logllh = Array(Float64, n)
  logpost = Array(Float64, n)

  for i in 1:n
    for j in 1:thin
      sample!(m)
    end
    samples[i,:] = unlist(m, true)

    # store the posterior and prior densities
    logprior[i] = logpdf(m[:logy0]) + logpdf(m[:parms])
    logllh[i]  = logpdf(m[:data])
    logpost[i] = logpdf(m)
  end

  samples, logprior, logllh, logpost
end


function mcmc(s::Sampling, iters)
  s = deepcopy(s)
  
  samples, prior, llh, post = sample(s.model, iters, s.config.thin)

  Sampling(
    vcat(s.samples, samples),
    vcat(s.logprior, prior),
    vcat(s.loglikelihood, llh),
    vcat(s.logpost, post),
    s.config,
    s.model)
end


### batch computation ###

" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."
function batch(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{Config, Void}=nothing, overwrite=false)

  local s

  if !isfile(path) || overwrite
    isa(config, Config) || Base.error("Need to give a config")
    s = mcmc(config, batchiters)
    save(path, s)
  else
    s = load(path, all=false)
  end

  thin = s.config.thin

  while (iters = min(batchiters, maxiters-(thin*size(s.samples, 1)))) >= thin
    s = mcmc(s, iters)
    save(path, s)
  end
  s
end
