function load(path; all::Bool=false)
  sim = deserialize_mc(JLD.load(path, "modelchains"))
  # TODO: implement all loading
end

function save(path, sim::ModelChains)
  s = size(sim.value)
  s[1] < 1     && error("cannot save empty chain")
  isfile(path) && error("$path already exists. Use append! or delete it.")

  mkpath(dirname(path))
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((s[1],s[2],s[3]), (-1,s[2],-1)), "chunk", (1000, s[2], 1))
    j["chains"][:,:,:] = sim.value
    j["modelchains"] = serialize_mc(sim[end:end,:,:])
  end
end

function serialize_mc(mc::ModelChains)
  io = IOBuffer()
  serialize(io, mc)
  io.data
end

deserialize_mc(ser) = deserialize(IOBuffer(ser))

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

function batch(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, mcmcargs...)

  if !isfile(path)
    isa(config, ModelConfig) || Base.error("not given a config")
    sim = mcmc(config, batchiters; mcmcargs...)
    save(path, sim)
  end

  sim = load(path)

  # only verbose is allowed in mcmc(::ModelChains,...)
  mcmcargs = filter(x->x[1]==:verbose, mcmcargs)

  # TODO: might contain bug if step skips maxiter
  while (iters = min(batchiters, maxiters-last(sim))) >= step(sim)
    sim = Mamba.mcmc(sim, iters; mcmcargs...)
    append!(path, sim)
    sim = load(path)
  end
  sim
end
