using Iterators: product

proposal(s::Sampling) = cov(log(s.samples)) * 2.38^2 / size(s.samples,2)
proposal(ss::Vector{Sampling}) = cov(log(vcat([s.samples for s in ss]...))) * 2.38^2 / size(ss[1].samples,2)

# construct the WeightedChain corresponding to the concatenated samples and respective data
function WeightedChain(samplings::Vector{Sampling}, burnin=0)


  # remove repeating samples
  curr = zeros(samplings[1].samples[1,:])
  samplevec = Matrix{Float64}[]
  counts  = Int[]

  for s in samplings
    for i in (1+burnin):size(s.samples, 1)
      if s.samples[i,:] == curr
        counts[end] += 1
      else
        curr = s.samples[i,:]
        push!(samplevec, curr)
        push!(counts, 1)
      end
    end
  end


  # compute prior and likelihoods

  prior   = map(samplevec) do s
              logprior(samplings[1].config, s |> vec)
            end


  lhs     = pmap(product(Vector[samplevec], [s.config for s in samplings])) do t
              map(s->llh(t[2], s |> vec), t[1])
            end 

  lhs     = hcat(lhs...)

  #prior   = Float64[
  #           logprior(samplings[1].config, samples[i,:] |> vec)
  #           for i in 1:size(samples,1)]

  #lhs     = Float64[
  #            llh(s.config, samples[i,:] |> vec)
  #            for i in 1:size(samples,1), s in samplings]

  # normalize for stability
  prior = (prior - maximum(prior))  |> exp
  lhs   = (lhs  .- maximum(lhs, 1)) |> exp
  
  samples = vcat(samplevec...)

  
  lhs   = lhs ./ sum(lhs, 1)
  weights = counts / sum(counts)
  density = Base.mean(lhs, 2) .* prior |> vec
  density = density / sum(density)

  # create weighted chain
  WeightedChain(samples, lhs, weights, density)
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
save(path, s::Sampling) = JLD.save(path, "sampling", s)
