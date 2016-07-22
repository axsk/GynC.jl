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


#=function sample(s::WeightedChain, n=1)
  norm = sum(s.weights)
  N    = size(s.samples, 1)
  
  S = Array{Float64}(n, size(s.samples, 2))

  for i = 1:n
    j = rand(1:N)
    while rand() > (s.weights[j] / norm)
      j = rand(1:N)
    end
    S[i, :] = s.samples[j, :]
  end
  S
end=#

density(c::WeightedChain) = c.upd .* c.weights

import Base.getindex

### fix upd scaling
getindex(c::WeightedChain, i) = WeightedChain(c.samples[i, :], c.likelihoods[i, :], c.weights[i, :] / sum(c.weights[i, :]), c.upd[i, :])

