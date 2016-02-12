const hdf_chunksize = 1_000
const relprop = 0.1 # relative standard deviation of proposal density

nsamples(path) = jldopen(j->size(j["chains"],1), path, "r")

scaledprop(relprop::Float64, n::Int) = log(1+(relprop^2)) * eye(n)

function startmcmc(c::ModelConfig, iters::Int, path::AbstractString, thin=100)
  # create sampler
  prop = scaledprop(relprop, length(sampledmles))
  samplers = [AMM([:sparms, :y0], prop, adapt=:all)]
 
  # create model
  m = model(c)
  setsamplers!(m, samplers)
  inp = Dict{Symbol,Any}()
  inits = [Dict(:y0 => mley0, :sparms => mleparms[sampleparms], :data => c.data)] |> Array{Dict{Symbol,Any}}
  
  # initial run
  #debug("starting initial run", Dict(:inits => size(inits)))
  sim = mcmc(m, inp, inits, iters, verbose=true, chains=1, thin=thin)

  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((size(sim.value,1),115,1),(-1,115,-1)), "chunk", (hdf_chunksize,115,1))
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

  d = size(last, 2)
  prop = scaledprop(relprop, length(sampledmles))
  samplers = [AMM([:sparms, :y0], prop, adapt=:all)]
  m = model(c)
  setsamplers!(m, samplers)
  inp = Dict{Symbol,Any}()
  inits = [Dict{Symbol,Any}(:sparms => vec(last[1, 1:82, chain]), :y0 => vec(last[1, 83:115, chain]), :data => c.data) for chain in 1:size(last,3)]
  m.samplers[1].tune = tune

  sim = mcmc(m, inp, inits, iters, verbose=true, thin=thin)

  jldopen(path, "r+") do j
    s = size(j["chains"])
    set_dims!(j.plain["chains"], (s[1]+size(sim.value,1), s[2], s[3]))
    j["chains"][s[1]+1:end, :, :] = sim.value
    delete!(j["tune"])
    j["tune"] = sim.model.samplers[1].tune
  end
  sim
end

function run(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, thin=100)
  if !isfile(path)
    isa(config, ModelConfig) || Base.error("not given a config")
    startmcmc(config, batchiters, path, thin)
  end

  while (iters = min(batchiters, maxiters-nsamples(path)*thin)) > 0
    continuemcmc(path, iters)
  end
end
