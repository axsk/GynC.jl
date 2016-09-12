##### Data structures #####

# WeightedChain represents a weighted sampling of a distribution
type WeightedChain
  samples::Matrix     # sample coordinates in each row
  likelihoods::Matrix # likelihoods for the different datas in each column 
  weights::Vector     # the corresponding weights 
  upd::Vector         # unweighted density, used to calculate the density for e.g. entropy calculation

  function WeightedChain(s, l, w=ones(size(s, 1)), d=ones(size(s, 1)))
    #w = w / sum(w)
    #l = l ./ sum(l, 1)
    u = d ./ w # TODO: think ab out this
    new(s,l,w,u)
  end
end

function sample(s::WeightedChain, n=1)
  cumdens = cumsum(s.weights)
  total   = cumdens[end]

  i = map(rand(n) * total) do target
    findfirst(x->x>=target, cumdens)
  end
  s.samples[i,:]
end

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


density(c::WeightedChain) = c.upd .* c.weights

import Base.getindex

### fix upd scaling
getindex(c::WeightedChain, i) = WeightedChain(c.samples[i, :], c.likelihoods[i, :], c.weights[i, :] / sum(c.weights[i, :]), c.upd[i, :])

