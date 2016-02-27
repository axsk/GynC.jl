function runsim(c, inity0, initparms, iters, thin, customtune=nothing)
  m = model(c)

  relprop = 0.1
  nparms = length(sampledmles)
  prop = log(1+(relprop^2)) * eye(nparms) # TODO: check

  setsamplers!(m, [AMM([:sparms, :y0], prop, adapt=:all)])

  inp = Dict{Symbol,Any}()
  inits = [Dict{Symbol,Any}(:y0 => inity0, :sparms => initparms, :data => c.data)]

  # TODO: fix this hack
  customtune != nothing && settune!(m, [customtune])
  sim = mcmc(m, inp, inits, iters, verbose=true, chains=1, thin=thin)
end

function startmcmc(c::ModelConfig, iters::Int, path::AbstractString, thin=100)
  sim = runsim(c, mley0, mleparms[sampleparms], iters)

  mkpath(dirname(path))
  jldopen(path, "w") do j
    hdfchunksize = 1000
    d_create(j.plain, "chains", Float64, ((size(sim.value,1),115,1),(-1,115,-1)), "chunk", (hdfchunksize,115,1))
    j["chains"][:,:,:] = sim.value
    j["tune"] = sim.model.samplers[1].tune
    j["modelconfig"] = c
    j["thin"] = thin
  end
  sim
end

function continuemcmc(path::AbstractString, iters::Int)
  local last, tune, thin, c

  jldopen(path, "r") do j
    last = j["chains"][end, :, :]
    tune = read(j["tune"])
    thin = read(j["thin"])
    c = read(j["modelconfig"])
  end

  lasty0 = last[1, 83:115, chain] |> vec
  lastthetha = last[1, 1:82, chain] |> vec

  sim = runsim(c, lasty0, lasttheta, iters, thin, tune)

  jldopen(path, "r+") do j
    s = size(j["chains"])
    set_dims!(j.plain["chains"], (s[1]+size(sim.value,1), s[2], s[3]))
    j["chains"][s[1]+1:end, :, :] = sim.value
    delete!(j["tune"])
    j["tune"] = sim.model.samplers[1].tune
  end
  sim
end

nsamples(path) = jldopen(j->size(j["chains"],1), path, "r")

function run(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, thin=100)
  if !isfile(path)
    isa(config, ModelConfig) || Base.error("not given a config")
    startmcmc(config, batchiters, path, thin)
  end

  while (iters = min(batchiters, maxiters-nsamples(path)*thin)) > 0
    continuemcmc(path, iters)
  end
end
