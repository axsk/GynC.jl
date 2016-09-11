using Iterators: product

# compute a proposal density based on the covariance of given samples
proposal(s::Matrix)   = cov(log(s)) * 2.38^2 / size(s,2)
proposal(s::Sampling) = proposal(s.samples)
proposal(ss::Vector{Sampling}) = proposal(vcat([s.samples for s in ss]...))


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
save(path, s::Sampling) = JLD.save(path, "sampling", s)

function readsamples(dir::AbstractString)
  cd(dir) do
    fs = filter(n->contains(n, ".jld"), readdir())
    ss = map(f->load(f), fs)
  end
end

# read all samplings from directory and create dataframe overview
function readdir(dir)
  ss = readsamples(dir)
end

function dataframe(ss::Vector{Sampling})
  dfs = [DataFrames.DataFrame(
    sample=s,
    data=s.config.patient.id, # ugly, use methods
    sigma=s.config.sigma_rho,
    adapt=s.config.adapt,
    thin=s.config.thin,
    length=length(s),
    unique=uniques(s),
    tracepropinit=trace(propinit(s)),
    tracepropadapt=trace(propadapt(s))) 
    for s in ss]
  vcat(dfs...)
end
