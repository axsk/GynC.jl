function load(path; all::Bool=false)
  sim = JLD.load(path, "modelchains")
  # TODO: implement all loading
end

function save(path, sim)
  isfile(path) && error("$path already exists. Delete it first or use append!")
  mkpath(dirname(path))
  s = size(sim.value)
  jldopen(path, "w") do j
    d_create(j.plain, "chains", Float64, ((s[1],s[2],s[3]), (-1,s[2],-1)), "chunk", (1000, s[2], 1))
    j["chains"][:,:,:] = sim.value
    j["modelchains"] = serialize_mc(sim[end,:,:])
  end
end

function serialize_mc(mc::ModelChains)
  io = IOBuffer()
  serialize(io, mc)
  io.data
end

deserialize_mc(ser) = deserialize(IOBuffer(ser))

function append!(path, sim)
  jldopen(path, "r+") do j
    oldrows = size(j["chains"], 1)
    oldmc   = j["modelchains"]
    diffmc = sim[(last(oldmc) + step(oldmc)):end, :, :]
    s = size(diffmc.value)

    set_dims!(j.plain["chains"], (oldrows+s[1], s[2], s[3]))
    j["chains"][(oldrows+1):end, :, :] = sim.value 
    j["modelchains"] = serialize_mc(sim[end,:,:])
  end
end

function batch(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, thin=100)
  if !isfile(path)
    isa(config, ModelConfig) || Base.error("not given a config")
    sim = mcmc(config, batchiters, thin = thin)
    save(path, sim)
  end

  sim = load(path)

  # TODO: might contain bug if step skips maxiter
  while (iters = min(batchiters, maxiters-last(sim))) > 0
    continuemcmc(path, iters)
    save(path, sim)
    sim = load(path)
  end
end
