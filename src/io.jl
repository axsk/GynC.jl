type ModelSerializer
  serializedmodel
  ModelSerializer(m::Mamba.Model) = new(serialize(m))
end

JLD.writeas(m::Mamba.Model) = ModelSerializer(m)
JLD.readas(ms::ModelSerializer) = deserialize(ms.serializedmodel)

" Serialize the ModelChain `mc`. Used for storage via JLD " 
function serialize(m::Mamba.Model)
  io = IOBuffer()
  Base.serialize(io, m)
  io.data
end

" Deserialize a serialized ModelChain "
deserialize(ser) = Base.deserialize(IOBuffer(ser)) :: Mamba.Model


load(path; all::Bool=true) = JLD.load(path, "sampling")

function save(path, s::Sampling)
  @assert path[end-3:end] == ".jld"
  JLD.save(path, "sampling", s)
end

" Append the surplus iterations from `sim` to the file specified by `path`."
function append!(path, s::Sampling)
  #=
  jldopen(path, "r+") do j
    oldrows = size(j["chains"], 1)
    oldmc   = deserialize_mc(read(j["modelchains"]))
    diffmc = sim[(last(oldmc) + step(oldmc)):end, :, :]
    s = size(diffmc.value)

    set_dims!(j.plain["chains"], (oldrows+s[1], s[2], s[3]))
    j["chains"][(oldrows+1):end, :, :] = sim.value 
    delete!(j["modelchains"])
    j["modelchains"] = serialize_mc(sim[end:end,:,:])
  end=#
end

" If the file speciefied in `path` exists, continue mcmc simulation of that file, otherwise start a new one with the given `config`.
Saves the result every `batchiters` to the file until `maxiters` is reached."
function batch(path::AbstractString; batchiters=100_000, maxiters=10_000_000, config::Union{ModelConfig, Void}=nothing, force=false, thin=1)

  local s

  if !isfile(path) || force
    isa(config, ModelConfig) || Base.error("Need to give a config")
    s = mcmc(config, batchiters; thin=thin)
    save(path, s)
  else
    s = load(path, all=false)
  end


  while (iters = min(batchiters, maxiters-size(s.samples, 1))) >= thin
    s = Mamba.mcmc(s, iters; thin=thin)
    append!(path, s)
    s = load(path)
  end
  s
end
