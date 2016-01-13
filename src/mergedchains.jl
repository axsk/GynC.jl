### WeightedChain 

type WeightedChain
  chain::AbstractMatrix
  weights::AbstractVector
  likelihoods::AbstractMatrix
end

function WeightedChain(chain::Matrix, data::Vector{Matrix}, sigma::Real)
  WeightedChain(chain, ones(size(chain, 1)), likelihoods(chain, data, sigma))
end

function likelihood(data::Matrix, sample::Vector, sigma::Real)
  parms, y0 = sampletoparms(sample)
  exp(llh(data, parms, y0, sigma))
end

function sampletoparms(sample::Vector)
  parms = copy(mleparms)
  parms[sampleparms] = sample[1:length(sampleparms)]
  y0 = sample[length(sampleparms)+1:end]
  parms, y0
end

""" compute the likelihood matrix for given chains, data, sigma) """
function likelihoods(chain::Matrix, data::Vector{Matrix}, sigma::Real)
  K = size(chain, 1)
  M = length(data)
  likelihoods = Matrix(K,M)
  for k = 1:K, m = 1:M
    likelihoods[k,m] = likelihood(data[m], chain[k,:], sigma)
  end
end

function reweight!(c::WeightedChain)
  w = c.weights
  L = c.likelihoods
  K = size(L,1)
  M = size(L,2)
  for k=1:K
    w[k] = w[k] / M * sum([L[k,m] / sum([w[j] * L[j,m] for j=1:K]) for m=1:M])
  end
end

### MergedChain

type MergedChain{T<:Real} <: AbstractMatrix{T}
  chains::Vector{Matrix{T}}
  
  function MergedChain{T}(chains::Vector{Matrix{T}})
    all([size(c) for c in chains] .== size(chains[1])) || warn("chains dont have same size")
    new(chains)
  end
end

chainlength(mc::MergedChain) = size(mc.chains[1], 1)
nchains(mc::MergedChain) = length(mc.chains)

Base.size(mc::MergedChain) = (nchains(mc) * chainlength(mc), size(mc.chains[1], 2))

function Base.getindex(mc::MergedChain, i::Int, j::Int) 
  chain = floor((i-1) / chainlength(mc)) + 1 |> Int
  index = ((i-1) % chainlength(mc) + 1)
  mc.chains[chain][index, j]
end
