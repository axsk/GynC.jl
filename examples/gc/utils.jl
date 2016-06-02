# construct the WeightedChain corresponding to the concatenated samples and respective data
function WeightedChain(samplings::Sampling...; sigma::Real=.1)

  samples = vcat([s.samples for s in samplings]...)

  prior   = Float64[
              logprior(samplings[1].config, samples[i,:] |> vec)
              for i in 1:size(samples,1)]

  lhs     = Float64[
              llh(s.config, samples[i,:] |> vec)
              for i in 1:size(samples,1), s in samplings]

  prior = (prior - maximum(prior))  |> exp
  lhs   = (lhs  .- maximum(lhs, 1)) |> exp
  
  WeightedChain(samples, lhs, prior)
end


### Cache for likelihood evaluation ###

""" return memoized version of f, caching the last n calls' results (overwriting on same-argument calls) """
function cache(fn::Function, n::Int)
    cin  = fill!(Vector{Any}(n), nothing)
    cout = fill!(Vector{Any}(n), nothing)
    c = 1
    function (args...)
        i = any(cin.==nothing) ? 0 : findfirst(cargs->myisapprox(cargs,args), cin)
        res = i == 0 ? fn(args...) : cout[i]
        cin[c] = deepcopy(args)
        cout[c] = res
        c = c % n + 1
        res
    end
end

myisapprox(x::Number, y::Number) = (isapprox(x,y) || isequal(x,y))
myisapprox(x,y) = (r=map(myisapprox, x,y); all(r))


### Serializer to work around JLD bug not permitting storing Mamba models ###

type Serializer
  serialized
  Serializer(s) = new(serialize(s))
end

JLD.writeas(s::Mamba.SamplerVariate) = Serializer(s)
JLD.readas(s::Serializer) = deserialize(s)

function serialize(s)
  io = IOBuffer()
  Base.serialize(io, s)
  io.data
end

deserialize(s::Serializer) = Base.deserialize(IOBuffer(s.serialized))


### load / save samplings ###

load(path) = JLD.load(path, "sampling")

function save(path, s::Sampling)
  @assert path[end-3:end] == ".jld"
  JLD.save(path, "sampling", s)
end
