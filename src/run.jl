const hdf_chunksize = 1_000

scaledprop(relprop::Float64, n::Int) = log(1+(relprop^2)) * eye(n)

function startmcmc(subj::Subject, iters::Int, chains::Int, path::AbstractString, relprop::Float64=0.1; thin=100)
  c = ModelConfig(subj)

  # create sampler
  prop = scaledprop(relprop, length(sampledmles))
  samplers = [AMM([:sparms, :y0], prop, adapt=:all)]
 
  # create model
  m = model(c)
  setsamplers!(m, samplers)
  inp = Dict{Symbol,Any}()
  inits = [Dict(:y0 => mley0, :sparms => mleparms[sampleparms], :data => c.data) for i=1:chains] |> Array{Dict{Symbol,Any}}
  
  # initial run
  #debug("starting initial run", Dict(:inits => size(inits)))
  sim = mcmc(m, inp, inits, iters, verbose=true, chains=chains, thin=thin)

  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((size(sim.value,1),115,chains),(-1,115,-1)), "chunk", (hdf_chunksize,115,1))
    j["chains"][:,:,:] = sim.value
    j["tune"] = sim.model.samplers[1].tune
    j["subj"] = subj
    j["modelconfig"] = c
    j["thin"] = thin
  end
  sim
end

nsamples(path) = jldopen(j->size(j["chains"],1), path, "r")

function run(path::AbstractString; batchiters=100_000, maxiters=1_000_000, subj::Union{Subject, Void}=nothing)
  if !isfile(path)
    subj == nothing && error("not given any subject")
    startmcmc(subj, batchiters, 1, path)
  end

  while iters = min(batchiters, maxiters-nsamples(path)) > 0
    continuemcmc(path, iters)
  end
end

function continuemcmc(path::AbstractString, iters::Int)
  local last, tune, subj, thin

  jldopen(path, "r") do j
    last = j["chains"][end, :, :]
    tune = read(j["tune"])
    subj = read(j["subj"])
    thin = read(j["thin"])
  end

  c = ModelConfig(subj)
  d = size(last, 2)
  samplers = [AMM([:sparms, :y0], eye(d,d), adapt=:all)]
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
