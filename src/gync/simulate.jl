" simulate `iters` iteration of the markov chain corresponding to the model specified by the `GynCConfig`, with initial values `inity0` and `initparms` (defaulting to the reference solution). The initial proposal density at point x is a Log-normal distribution with median x standard deviation x*`relprop` " 
function mcmc(c::GynCConfig, iters)
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

  n = round(Int, iters/c.thin, RoundDown)
  sim = Array(Float64, n, length(unlist(m, true)))
  logprior = Array(Float64, n)
  logllh = Array(Float64, n)
  logpost = Array(Float64, n)

  for i in 1:n
    for j in 1:c.thin
      sample!(m)
    end
    sim[i,:] = unlist(m, true)

    # store the posterior and prior densities
    logprior[i] = logpdf(m[:logy0]) + logpdf(m[:parms])
    logllh[i]  = logpdf(m[:data])
    logpost[i] = logpdf(m)
  end

  Sampling(sim, logprior, logllh, logpost, c)
end

function mcmc(s::Sampling, iters)
  c = deepcopy(s.config)
  c.init = s.samples[end,:] |> vec
  s1 = s
  s2 = mcmc(c, iters)
  Sampling(
    vcat(s1.samples, s2.samples),
    vcat(s1.logprior, s2.logprior),
    vcat(s1.loglikelihood, s2.loglikelihood),
    vcat(s1.logpost, s2.logpost),
    s1.config)
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
