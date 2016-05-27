# construct the WeightedChain corresponding to the concatenated samples and respective data
function WeightedChain(samplings::Sampling...; sigma::Real=.1)
  samples = vcat([s.samples for s in samplings]...)
  prior   = vcat([exp(s.logprior) for s in samplings]...)
  lhs     = likelihoods(samples, Matrix[s.config.data for s in samplings], sigma)
  WeightedChain(samples, lhs, prior)
end

### model tools ###

allparms(parms::Vector) = (p = copy(refallparms); p[sampledinds] = parms; p)

" given a sample, extend to all model parameters " 
function sampletoparms(sample::Vector)
  np = length(sampledinds)
  parms = allparms(sample[1:np])
  y0 = sample[np+1:end]
  parms, y0
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

type ModelSerializer
  serializedmodel
  ModelSerializer(m::Mamba.Model) = new(serialize(m))
end

JLD.writeas(m::Mamba.Model) = ModelSerializer(m)
JLD.readas(ms::ModelSerializer) = deserialize(ms.serializedmodel)

function serialize(m::Mamba.Model)
  io = IOBuffer()
  Base.serialize(io, m)
  io.data
end

deserialize(ser) = Base.deserialize(IOBuffer(ser)) :: Mamba.Model


### load / save samplings ###

load(path) = JLD.load(path, "sampling")

function save(path, s::Sampling)
  @assert path[end-3:end] == ".jld"
  JLD.save(path, "sampling", s)
end


### MergedChain

""" memory efficient structure to represent the merged chain """
type MergedChain{T<:Real} <: AbstractMatrix{T}
  chains::Vector{Matrix{T}}

  function MergedChain{T}(chains::Vector{Matrix{T}})
    all([size(c) for c in chains] .== size(chains[1])) || warn("chains dont have same size")
    new(chains)
  end
end

mergedchain(chains...) = MergedChain{Float64}(Vector{Matrix{Float64}}(chains...))

chainlength(mc::MergedChain) = size(mc.chains[1], 1)
nchains(mc::MergedChain) = length(mc.chains)

Base.size(mc::MergedChain) = (nchains(mc) * chainlength(mc), size(mc.chains[1], 2))

function Base.getindex(mc::MergedChain, i::Int, j::Int)
  chain = floor((i-1) / chainlength(mc)) + 1 |> Int
  index = ((i-1) % chainlength(mc) + 1)
  mc.chains[chain][index, j]
end
