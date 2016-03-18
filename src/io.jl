" Load a saved ModelChain from `path`. If `all=false` only load the last iteration "
function load(path; all::Bool=true)
  c = deserialize_mc(JLD.load(path, "modelchains"))
  if all
    val = JLD.load(path, "chains")
    range = c.range.step:c.range.step:c.range.stop
    c = ModelChains(Chains(val, range, c.names, c.chains), c.model)
  end
  c
end

" Save the simulated ModelChain `sim` to `path`. "
function save(path, sim::ModelChains; force=false)
  s = size(sim.value)
  s[1] < 1 && error("cannot save empty chain")
  !force && isfile(path) && error("$path already exists. Use `append!` or force overwriting using `force`.")

  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((s[1],s[2],s[3]), (-1,s[2],-1)), "chunk", (1000, s[2], 1))
    j["chains"][:,:,:] = sim.value
    j["modelchains"] = serialize_mc(sim[end:end,:,:])
  end
  sim
end

" Serialize the ModelChain `mc`. Used for storage via JLD " 
function serialize_mc(mc::ModelChains)
  io = IOBuffer()
  serialize(io, mc)
  io.data
end

" Deserialize a serialized ModelChain "
deserialize_mc(ser) = deserialize(IOBuffer(ser))

" Append the surplus iterations from `sim` to the file specified by `path`."
function append!(path, sim::ModelChains)
  jldopen(path, "r+") do j
    oldrows = size(j["chains"], 1)
    oldmc   = deserialize_mc(read(j["modelchains"]))
    diffmc = sim[(last(oldmc) + step(oldmc)):end, :, :]
    s = size(diffmc.value)

    set_dims!(j.plain["chains"], (oldrows+s[1], s[2], s[3]))
    j["chains"][(oldrows+1):end, :, :] = sim.value 
    delete!(j["modelchains"])
    j["modelchains"] = serialize_mc(sim[end:end,:,:])
  end
end

" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."
function batch(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, force=false, mcmcargs...)

  if !isfile(path) || force
    isa(config, ModelConfig) || Base.error("Need to give a config")
    sim = mcmc(config, batchiters; mcmcargs...)
    save(path, sim, force=force)
  end

  sim = load(path, all=false)

  # only verbose is allowed in mcmc(::ModelChains,...)
  mcmcargs = filter(x->x[1]==:verbose, mcmcargs)

  while (iters = min(batchiters, maxiters-last(sim))) >= step(sim)
    sim = Mamba.mcmc(sim, iters; mcmcargs...)
    append!(path, sim)
    sim = load(path)
  end
  sim
end
